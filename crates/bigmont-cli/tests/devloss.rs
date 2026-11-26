//! Engine recovery from device-level GPU failures: a lost device is re-opened and the bucket re-run;
//! after `device_retries` failed re-opens the engine falls back to the CPU for good; a GPU wait
//! longer than the poll timeout is handled the same way. Verdicts must equal the oracle in every
//! path. The device loss is simulated with `device.destroy()` through the engine's test hooks.

use bigmont_cli::engine::{Backend, Engine, EngineReport, Options};
use bigmont_oracle::*;
use num_bigint::BigUint;
use std::time::Duration;

mod common;

/// Two GPU buckets (1019 bits: 10 random + 2 PRPs, 777 bits: 9 random) and a lone 3 (CPU).
fn batch() -> (Vec<BigUint>, Vec<u8>, u64) {
    let mut rng = XorShift64::new(9);
    let mut ns: Vec<BigUint> = (0..10)
        .map(|_| random_odd_modulus(&mut rng, 1019))
        .collect();
    ns.extend((0..2).map(|_| random_prp(&mut rng, 1019)));
    ns.extend((0..9).map(|_| random_odd_modulus(&mut rng, 777)));
    ns.push(BigUint::from(3u32));
    let want = ns.iter().map(|n| u8::from(fermat_base2(n))).collect();
    (ns, want, 12 + 9)
}

struct RemoveOnDrop(std::path::PathBuf);
impl Drop for RemoveOnDrop {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn opts(tag: &str) -> (Options, RemoveOnDrop) {
    let cache = std::env::temp_dir().join(format!(
        "bigmont-devloss-{tag}-{}.cache",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&cache);
    let o = Options {
        backend: Backend::Gpu,
        min_gpu_batch: 4,
        capacity: 32,
        verify_rate: 0,
        cache: Some(cache.clone()),
        ..Options::default()
    };
    (o, RemoveOnDrop(cache))
}

fn show(label: &str, r: &EngineReport) {
    eprintln!("{label}: {r:?}");
}

#[test]
fn the_default_allows_two_device_retries() {
    let o = Options::default();
    assert_eq!(o.device_retries, 2);
    assert_eq!(o.device_retries_total, 8);
    assert_eq!(o.inject_loss_on_open, 0);
    assert_eq!(o.poll_timeout, None);
    let r = EngineReport::default();
    assert_eq!(
        (r.device_losses, r.device_recoveries, r.poll_timeouts),
        (0, 0, 0)
    );
}

#[test]
fn a_lost_device_between_calls_is_reopened_and_the_buckets_rerun_on_the_gpu() {
    if !common::usable_gpu() {
        return;
    }
    let (ns, want, gpu_candidates) = batch();
    let (o, _cleanup) = opts("reopen");
    let mut e = Engine::open(o);
    eprintln!("{}", e.describe());
    assert_eq!(e.verdicts(&ns).unwrap(), want);
    let before = e.report();
    show("before", &before);
    assert_eq!(before.gpu_tests, gpu_candidates);
    assert_eq!(
        (
            before.device_losses,
            before.device_recoveries,
            before.gpu_errors
        ),
        (0, 0, 0)
    );

    e.inject_device_loss();
    assert_eq!(
        e.verdicts(&ns).unwrap(),
        want,
        "verdicts after a device loss must equal the oracle"
    );
    let after = e.report();
    show("after", &after);
    assert_eq!(after.device_losses, 1);
    assert_eq!(after.device_recoveries, 1);
    assert_eq!(after.poll_timeouts, 0);
    assert_eq!(
        after.gpu_errors, 0,
        "a recovered loss is not a CPU fallback"
    );
    assert_eq!(
        after.gpu_tests - before.gpu_tests,
        gpu_candidates,
        "both buckets ran on the re-opened GPU"
    );
    assert_eq!(
        after.cpu_tests - before.cpu_tests,
        1,
        "only the lone 3 ran on the CPU"
    );
    assert_eq!(
        after.tuned_classes, before.tuned_classes,
        "the tuned configurations are kept: no re-tune"
    );
    assert!(e.gpu_active());
    assert_eq!(
        (after.prp_disagreements, after.composite_disagreements),
        (0, 0)
    );
}

#[test]
fn a_lost_device_with_the_gerbicz_check_on_is_reopened_and_the_checked_runners_rebuilt() {
    if !common::usable_gpu() {
        return;
    }
    let (ns, want, gpu_candidates) = batch();
    let (o, _cleanup) = opts("reopen-gec");
    let mut e = Engine::open(Options { gec: true, ..o });
    assert_eq!(e.verdicts(&ns).unwrap(), want);
    e.inject_device_loss();
    assert_eq!(
        e.verdicts(&ns).unwrap(),
        want,
        "verdicts after a device loss must equal the oracle"
    );
    let r = e.report();
    show("gec after", &r);
    assert_eq!(
        (r.device_losses, r.device_recoveries, r.gpu_errors),
        (1, 1, 0)
    );
    assert_eq!(r.gpu_tests, 2 * gpu_candidates, "both calls ran on the GPU");
    assert_eq!(r.gec_alarms, 0, "a lost device is not a Gerbicz alarm");
    assert!(e.gpu_active());
}

#[test]
fn more_losses_than_retries_fall_back_to_the_cpu_for_the_rest_of_the_process() {
    if !common::usable_gpu() {
        return;
    }
    let (ns, want, gpu_candidates) = batch();
    let (o, _cleanup) = opts("exhaust");
    // The first device and both re-opened ones are destroyed right after opening.
    let o = Options {
        inject_loss_on_open: 3,
        device_retries: 2,
        ..o
    };
    let mut e = Engine::open(o);
    assert_eq!(
        e.verdicts(&ns).unwrap(),
        want,
        "the CPU fallback must give the oracle's verdicts"
    );
    let first = e.report();
    show("first", &first);
    // One loss (the first device, found dead by the first GPU bucket); both re-opened devices were
    // dead too, which is a failed re-open, not another loss; the bucket fell back once.
    assert_eq!(first.device_losses, 1);
    assert_eq!(first.gpu_errors, 1);
    assert_eq!(first.poll_timeouts, 0);
    assert_eq!(
        first.device_recoveries, 0,
        "every re-opened device was dead"
    );
    assert_eq!(first.gpu_tests, 0);
    assert_eq!(
        first.cpu_tests,
        gpu_candidates + 1,
        "every candidate ran on the CPU"
    );
    assert!(
        !e.gpu_active(),
        "the engine is CPU-only after the retries are used up"
    );

    assert_eq!(e.verdicts(&ns).unwrap(), want);
    let second = e.report();
    show("second", &second);
    assert_eq!(
        second.device_losses, first.device_losses,
        "the second call does not touch the GPU"
    );
    assert_eq!(second.device_recoveries, first.device_recoveries);
    assert_eq!(second.gpu_tests, 0);
    assert_eq!(second.gpu_errors, first.gpu_errors);
    assert_eq!(second.cpu_tests - first.cpu_tests, gpu_candidates + 1);
}

#[test]
fn a_poll_timeout_is_counted_and_the_verdicts_come_from_the_cpu() {
    if !common::usable_gpu() {
        return;
    }
    let (ns, want, _) = batch();
    let (o, _cleanup) = opts("timeout");
    // Far shorter than any real dispatch: the first wait that does not finish at once times out.
    let o = Options {
        poll_timeout: Some(Duration::from_micros(1)),
        ..o
    };
    let mut e = Engine::open(o);
    assert_eq!(e.verdicts(&ns).unwrap(), want);
    let r = e.report();
    show("timeout", &r);
    // The first GPU bucket times out on the first device and on both re-opened ones (the budget of
    // 2 re-opens per call), then everything runs on the CPU.
    assert_eq!(r.poll_timeouts, 3, "{r:?}");
    assert_eq!(
        r.device_losses, 3,
        "a timeout is a device-level failure: {r:?}"
    );
    assert_eq!(r.device_recoveries, 2, "{r:?}");
    assert_eq!(r.gpu_errors, 1, "{r:?}");
    assert_eq!(r.gpu_tests, 0, "{r:?}");
    assert_eq!(r.cpu_tests, 22, "every candidate ran on the CPU: {r:?}");
    assert!(!e.gpu_active());
}

#[test]
fn a_panicking_poll_is_recovered_like_a_lost_device() {
    if !common::usable_gpu() {
        return;
    }
    // The path a real NVIDIA Xid takes: wgpu 30's Device::poll panics. The engine must classify it
    // as a device failure, re-open, and re-run the bucket on the GPU.
    let (ns, want, gpu_candidates) = batch();
    let (o, _cleanup) = opts("pollpanic");
    let mut e = Engine::open(o);
    assert_eq!(e.verdicts(&ns).unwrap(), want);
    let before = e.report();
    e.inject_poll_panic();
    assert_eq!(e.verdicts(&ns).unwrap(), want);
    let after = e.report();
    show("poll panic", &after);
    assert_eq!(
        (
            after.device_losses,
            after.device_recoveries,
            after.poll_timeouts,
            after.gpu_errors
        ),
        (1, 1, 0, 0)
    );
    assert_eq!(
        after.gpu_tests - before.gpu_tests,
        gpu_candidates,
        "both buckets ran on the re-opened GPU"
    );
    assert_eq!(
        after.tuned_classes, before.tuned_classes,
        "same adapter: the tuning is kept"
    );
    assert!(e.gpu_active());
}

#[test]
fn a_different_adapter_after_a_reopen_is_tuned_again() {
    if !common::usable_gpu() {
        return;
    }
    let (ns, want, gpu_candidates) = batch();
    let (o, _cleanup) = opts("adapter");
    let mut e = Engine::open(o);
    assert_eq!(e.verdicts(&ns).unwrap(), want);
    let before = e.report();
    assert_eq!(
        before.tuned_classes, 2,
        "1019 and 777 bits are two limb classes"
    );
    e.inject_device_loss();
    e.inject_adapter_change();
    assert_eq!(
        e.verdicts(&ns).unwrap(),
        want,
        "verdicts after re-tuning on the 'new' adapter"
    );
    let after = e.report();
    show("adapter change", &after);
    assert_eq!(
        (
            after.device_losses,
            after.device_recoveries,
            after.gpu_errors
        ),
        (1, 1, 0)
    );
    assert_eq!(
        after.tuned_classes,
        before.tuned_classes + 2,
        "both classes are tuned (self-tested) again on the new adapter"
    );
    assert_eq!(after.gpu_tests - before.gpu_tests, gpu_candidates);
}

#[test]
fn the_lifetime_reopen_cap_makes_the_engine_cpu_only() {
    if !common::usable_gpu() {
        return;
    }
    // A device that dies on every call: the per-call budget resets each call, the lifetime cap does not.
    let (ns, want, gpu_candidates) = batch();
    let (o, _cleanup) = opts("cap");
    let mut e = Engine::open(Options {
        device_retries: 2,
        device_retries_total: 2,
        ..o
    });
    assert_eq!(e.verdicts(&ns).unwrap(), want);
    for call in 1..=2u64 {
        e.inject_device_loss();
        assert_eq!(e.verdicts(&ns).unwrap(), want);
        let r = e.report();
        assert_eq!(
            (r.device_losses, r.device_recoveries, r.gpu_errors),
            (call, call, 0),
            "call {call}: {r:?}"
        );
        assert!(e.gpu_active());
    }
    let r2 = e.report();
    e.inject_device_loss();
    assert_eq!(e.verdicts(&ns).unwrap(), want);
    let r = e.report();
    show("cap", &r);
    assert_eq!(
        (r.device_losses, r.device_recoveries, r.gpu_errors),
        (3, 2, 1),
        "the third loss finds the lifetime cap used up: {r:?}"
    );
    assert_eq!(r.gpu_tests, r2.gpu_tests, "nothing more ran on the GPU");
    assert_eq!(r.cpu_tests - r2.cpu_tests, gpu_candidates + 1);
    assert!(!e.gpu_active());
    assert!(
        e.describe().contains("device_retries_total"),
        "{}",
        e.describe()
    );
}
