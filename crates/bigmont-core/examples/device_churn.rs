//! Diagnostic: which part of "create a wgpu device, run a kernel, drop it" triggers the
//! intermittent failures seen in the test binaries (NVIDIA Xid 32 / hangs on Linux, a 0xC0000409
//! crash inside the AMD user-mode driver at process exit on Windows)? The CLI binary (one device per
//! process, main thread) has never shown them; the test binaries (a new device per check, each test
//! on its own spawned thread) have.
//!
//!   device_churn MODE N [tpi] [squarings] [flags...]
//!
//! MODE: `new` (fresh Gpu per cycle, dropped each cycle), `reuse` (one Gpu), `keep` (fresh Gpu per
//! cycle, all alive until exit). Flags (any order): `thread` runs every cycle on its own spawned
//! thread that exits before the next cycle (like the test harness); `vkonly` builds the wgpu
//! Instance with the Vulkan backend only (the library default is now Vulkan + Dx12 + Metal; set
//! BIGMONT_INSTANCE_BACKENDS=all to get every backend, GL included, back).
//!
//! A clean run exits 0; the interesting outcomes are a hang, a crash at exit, or a device-lost error.

use bigmont_core::gpu::{Batch, Gpu, run};
use bigmont_core::kernel::{KernelConfig, Variant};

fn make_gpu(vkonly: bool) -> Gpu {
    if !vkonly {
        return Gpu::new().expect("gpu");
    }
    pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .expect("vulkan adapter");
        let info = adapter.get_info();
        let has_subgroups = adapter.features().contains(wgpu::Features::SUBGROUP);
        let features = if has_subgroups {
            wgpu::Features::SUBGROUP
        } else {
            wgpu::Features::empty()
        };
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_features: features,
                ..Default::default()
            })
            .await
            .expect("device");
        Gpu::from_parts(info, has_subgroups, device, queue)
    })
}

/// Open file descriptors and threads of this process (Linux only; empty elsewhere): a per-cycle
/// growth shows that dropping a `Gpu` does not release everything it acquired.
fn proc_counts() -> String {
    let fds = std::fs::read_dir("/proc/self/fd").map(|d| d.count().to_string());
    let threads = std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("Threads:"))
                .map(|l| l.trim_start_matches("Threads:").trim().to_string())
        });
    match (fds, threads) {
        (Ok(f), Some(t)) => format!("fds {f} threads {t}"),
        _ => String::new(),
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mode = args.next().expect("mode: new | reuse | keep");
    let cycles: u32 = args.next().map_or(12, |s| s.parse().expect("N"));
    let tpi: u32 = args.next().map_or(8, |s| s.parse().expect("tpi"));
    let squarings: u32 = args.next().map_or(128, |s| s.parse().expect("squarings"));
    let rest: Vec<String> = args.collect();
    let (threaded, vkonly) = (
        rest.iter().any(|a| a == "thread"),
        rest.iter().any(|a| a == "vkonly"),
    );
    let cfg = KernelConfig::new(tpi, 64, Variant::Unrolled);
    let batch = Batch::from_values(&Batch::random_values(70, 2029, 7));

    let cycle = |i: u32, gpu: &Gpu| {
        let r = run(gpu, cfg, &batch, squarings, 1).expect("run");
        eprintln!(
            "cycle {i}: {} limbs back, {:.4} s | {} [{:?}] | {}",
            r.x.len(),
            r.best_secs,
            gpu.info.name,
            gpu.info.backend,
            proc_counts()
        );
    };
    let shared = (mode == "reuse").then(|| make_gpu(vkonly));
    let mut kept = Vec::new();
    for i in 0..cycles {
        if mode == "keep" {
            let gpu = make_gpu(vkonly);
            cycle(i, &gpu);
            kept.push(gpu);
        } else if threaded {
            std::thread::scope(|s| {
                s.spawn(|| match &shared {
                    Some(g) => cycle(i, g),
                    None => cycle(i, &make_gpu(vkonly)),
                });
            });
        } else {
            match &shared {
                Some(g) => cycle(i, g),
                None => cycle(i, &make_gpu(vkonly)),
            }
        }
    }
    eprintln!(
        "exiting normally after {cycles} cycles ({mode}{}{})",
        if threaded { ", thread" } else { "" },
        if vkonly { ", vkonly" } else { "" }
    );
}
