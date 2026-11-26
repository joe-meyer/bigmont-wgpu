//! Device-loss handling: a lost device, a GPU wait longer than the poll timeout and an uncaptured
//! wgpu error must come back as `Err` from the public entry points, never as a panic or a hang.
//!
//! `device.destroy()` makes wgpu report the device lost (reason `Destroyed`) deterministically,
//! which stands in for the rare NVIDIA Xid 32 losses (see
//! docs/superpowers/results/2026-11-24-device-loss-investigation.md).

#![allow(clippy::err_expect)] // Run and Runner are not Debug

mod common;
use bigmont_core::gpu::{Batch, Gpu, run_async};
use bigmont_core::kernel::{KernelConfig, Variant};
use bigmont_core::runner::Runner;
use bigmont_oracle::{XorShift64, fermat_base2, random_odd_modulus};
use num_bigint::BigUint;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::{Duration, Instant};

const BITS: u32 = 256;

fn cfg() -> KernelConfig {
    KernelConfig::for_bits(BITS, 8, 64, Variant::Unrolled)
}

/// A squaring-mode batch for `cfg` (moduli with R > 4n, as the self-test uses).
fn squaring_batch(cfg: KernelConfig, count: u32, seed: u64) -> Batch {
    let values = Batch::random_values(count, cfg.limbs * 13 - 3, seed);
    Batch::from_values_n(&values, cfg.limbs as usize)
}

fn candidates(count: usize, seed: u64) -> Vec<BigUint> {
    let mut rng = XorShift64::new(seed);
    (0..count)
        .map(|_| random_odd_modulus(&mut rng, BITS))
        .collect()
}

/// Runs `f`, failing the test if it panics; returns its value.
fn no_panic<T>(what: &str, f: impl FnOnce() -> T) -> T {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(v) => v,
        Err(p) => {
            let msg = p
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_default();
            panic!("{what} panicked instead of returning Err: {msg}");
        }
    }
}

#[test]
fn a_destroyed_device_gives_device_lost_errors_not_panics() {
    let cfg = cfg();
    let Some(gpu) = common::open_gpu(cfg) else {
        return;
    };
    let batch = squaring_batch(cfg, 64, 1);
    let ns = candidates(40, 2);
    let want: Vec<bool> = ns.iter().map(fermat_base2).collect();

    // Healthy first: both entry points work and the device reports no problem.
    pollster::block_on(run_async(&gpu, cfg, &batch, 3, 1)).expect("healthy run");
    let mut runner =
        pollster::block_on(Runner::new_async(&gpu, cfg, BITS, 16)).expect("healthy runner");
    assert_eq!(
        pollster::block_on(runner.fermat_async(&gpu, &ns)).expect("healthy fermat"),
        want
    );
    assert!(!gpu.is_lost());
    gpu.check_health().expect("healthy device");

    gpu.device.destroy();
    let t = Instant::now();
    let e = no_panic("run on a destroyed device", || {
        pollster::block_on(run_async(&gpu, cfg, &batch, 3, 1))
    })
    .err()
    .expect("run on a destroyed device must fail");
    assert!(e.contains("device lost"), "{e}");
    let e = no_panic("Runner::fermat on a destroyed device", || {
        pollster::block_on(runner.fermat_async(&gpu, &ns))
    })
    .expect_err("fermat on a destroyed device must fail");
    assert!(e.contains("device lost"), "{e}");
    let e = no_panic("Runner::new on a destroyed device", || {
        pollster::block_on(Runner::new_async(&gpu, cfg, BITS, 16))
    })
    .err()
    .expect("Runner::new on a destroyed device must fail");
    assert!(e.contains("device lost"), "{e}");
    let e = no_panic("Gpu::wait on a destroyed device", || gpu.wait())
        .expect_err("wait on a destroyed device must fail");
    assert!(e.contains("device lost"), "{e}");
    assert!(
        t.elapsed() < Duration::from_secs(10),
        "the errors took {:?}",
        t.elapsed()
    );
    assert!(gpu.is_lost());
    assert!(gpu.check_health().unwrap_err().contains("device lost"));
}

#[test]
fn a_wait_longer_than_the_poll_timeout_is_a_timed_out_error() {
    let cfg = KernelConfig::for_bits(2029, 8, 64, Variant::Unrolled);
    let Some(mut gpu) = common::open_gpu(cfg) else {
        return;
    };
    gpu.set_poll_timeout(Duration::from_millis(1));
    assert_eq!(gpu.poll_timeout, Duration::from_millis(1));
    let batch = squaring_batch(cfg, 4096, 3);
    // Grow the dispatch until one takes longer than the 1 ms timeout (a fast GPU finishes the small ones).
    let mut squarings = 4u32;
    let e = loop {
        match no_panic("a run with a 1 ms poll timeout", || {
            pollster::block_on(run_async(&gpu, cfg, &batch, squarings, 1))
        }) {
            Ok(r) => {
                assert!(
                    r.best_secs < 0.5,
                    "a {:.3} s dispatch ({squarings} squarings) finished without a timeout under a 1 ms poll timeout",
                    r.best_secs
                );
                squarings *= 2;
            }
            Err(e) => break e,
        }
    };
    eprintln!("timed out at {squarings} squarings: {e}");
    assert!(e.contains("timed out"), "{e}");
    assert!(gpu.is_poisoned(), "a timeout poisons the Gpu");
    assert!(gpu.check_health().unwrap_err().contains("timed out"));
    // Every later call fails at once (the device may never finish its work).
    let t = Instant::now();
    let e = no_panic("a run on a poisoned Gpu", || {
        pollster::block_on(run_async(&gpu, cfg, &batch, 1, 1))
    })
    .err()
    .expect("a poisoned Gpu must refuse work");
    assert!(e.contains("timed out"), "{e}");
    assert!(t.elapsed() < Duration::from_secs(1));
    // Dropping the poisoned Gpu must not block on the outstanding work (it is leaked instead).
    let t = Instant::now();
    drop(gpu);
    assert!(
        t.elapsed() < Duration::from_secs(5),
        "dropping a poisoned Gpu took {:?}",
        t.elapsed()
    );
}

#[test]
fn an_uncaptured_wgpu_error_is_recorded_instead_of_panicking() {
    let cfg = cfg();
    let Some(gpu) = common::open_gpu(cfg) else {
        return;
    };
    gpu.check_health().expect("healthy device");
    // MAP_READ together with STORAGE is invalid without MAPPABLE_PRIMARY_BUFFERS: a validation error
    // raised outside any error scope, which wgpu's default handler turns into a panic.
    let buffer = no_panic("an invalid create_buffer", || {
        gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("invalid on purpose"),
            size: 16,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        })
    });
    drop(buffer);
    let e = gpu
        .check_health()
        .expect_err("the uncaptured error must be reported");
    assert!(e.contains("uncaptured wgpu error"), "{e}");
    assert!(
        gpu.uncaptured_errors().len() == 1,
        "{:?}",
        gpu.uncaptured_errors()
    );
    assert!(!gpu.is_lost(), "a validation error is not a lost device");
    // The entry points refuse to run on a device that reported an uncaptured error.
    let e = no_panic("run after an uncaptured error", || {
        pollster::block_on(run_async(&gpu, cfg, &squaring_batch(cfg, 8, 4), 3, 1))
    })
    .err()
    .expect("run must fail");
    assert!(e.contains("uncaptured wgpu error"), "{e}");
}

#[test]
fn a_panicking_poll_is_a_driver_loss_and_the_device_is_leaked_on_drop() {
    // wgpu 30's Device::poll panics ("Error in Device::poll: Parent device is lost") on a real
    // driver-side loss; destroy() never takes this path, so inject the panic directly.
    let Some(gpu) = common::open_gpu(cfg()) else {
        return;
    };
    let e = no_panic("a wait whose poll panics", || {
        gpu.wait_with(|_, _| panic!("Error in Device::poll: Parent device is lost (test)"))
    })
    .expect_err("a panicking poll must be an Err");
    assert!(
        e.contains("device lost")
            && e.contains("poll panicked")
            && e.contains("Parent device is lost"),
        "{e}"
    );
    assert!(gpu.is_lost());
    assert!(!gpu.is_poisoned(), "a lost device is not a timeout");
    assert!(
        gpu.leaks_on_drop(),
        "a driver loss leaks the device instead of a possibly blocking drop"
    );
    assert!(gpu.check_health().unwrap_err().contains("poll panicked"));
    // A later destroy() does not turn the driver loss into an idle, normally dropped device.
    gpu.device.destroy();
    assert!(gpu.check_health().unwrap_err().contains("poll panicked"));
    assert!(gpu.leaks_on_drop());
    let t = Instant::now();
    drop(gpu);
    assert!(t.elapsed() < Duration::from_secs(1));
}

#[test]
fn the_poll_panic_hook_reaches_the_entry_points() {
    let cfg = cfg();
    let Some(gpu) = common::open_gpu(cfg) else {
        return;
    };
    let batch = squaring_batch(cfg, 64, 5);
    pollster::block_on(run_async(&gpu, cfg, &batch, 3, 1)).expect("healthy run");
    gpu.inject_poll_panic();
    let e = no_panic("run with an injected poll panic", || {
        pollster::block_on(run_async(&gpu, cfg, &batch, 3, 1))
    })
    .err()
    .expect("run must fail");
    assert!(
        e.contains("device lost") && e.contains("poll panicked"),
        "{e}"
    );
    assert!(gpu.is_lost() && !gpu.is_poisoned() && gpu.leaks_on_drop());
}

#[test]
fn a_poll_that_never_returns_is_abandoned_after_the_grace_period() {
    // The hang mode seen on the laptop: the poll ignores its own timeout. The wait gives up after
    // the timeout plus a 2 s grace, poisons the Gpu and leaves the helper thread behind.
    let Some(mut gpu) = common::open_gpu(cfg()) else {
        return;
    };
    gpu.set_poll_timeout(Duration::from_millis(1));
    let t = Instant::now();
    let e = no_panic("a wait whose poll hangs", || {
        gpu.wait_with(|_, _| {
            std::thread::sleep(Duration::from_secs(4));
            Ok(wgpu::PollStatus::QueueEmpty)
        })
    })
    .expect_err("a hung poll must be an Err");
    let waited = t.elapsed();
    assert!(e.contains("timed out") && e.contains("grace"), "{e}");
    assert!(
        waited >= Duration::from_secs(2) && waited < Duration::from_millis(3500),
        "gave up after {waited:?}"
    );
    assert!(gpu.is_poisoned());
    assert!(!gpu.is_lost());
    assert!(gpu.leaks_on_drop());
    assert!(gpu.check_health().unwrap_err().contains("timed out"));
}

#[test]
fn the_device_lost_callback_and_handlers_survive_from_parts() {
    // Gpu::from_parts (used by diagnostics that build their own device) installs the same handlers.
    let Some(opened) = common::open_gpu(cfg()) else {
        return;
    };
    let gpu = Gpu::from_parts(
        opened.info.clone(),
        opened.has_subgroups,
        opened.device.clone(),
        opened.queue.clone(),
    );
    gpu.device.destroy();
    assert!(gpu.check_health().unwrap_err().contains("device lost"));
    assert!(gpu.is_lost());
}
