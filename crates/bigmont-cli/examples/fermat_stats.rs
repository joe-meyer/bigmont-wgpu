//! Measures the raw GPU runner on a candidate file and prints `RunStats` (host-side Montgomery
//! preparation against GPU dispatch time), without the engine's CPU re-checks.
//!
//!   BIGMONT_ADAPTER=... cargo run --release -p bigmont-cli --example fermat_stats -- CAND [CAPACITY]

use bigmont_cli::tune::SchemeChoice;
use bigmont_cli::{formats, tune};
use bigmont_core::gpu::Gpu;
use bigmont_core::runner::Runner;
use std::path::Path;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(cand) = args.first() else {
        eprintln!("usage: fermat_stats CAND [CAPACITY]");
        std::process::exit(2);
    };
    let capacity: u32 = args
        .get(1)
        .map_or(16384, |c| c.parse().expect("CAPACITY must be a number"));
    let file = formats::read_candidates(Path::new(cand)).expect("read candidates");
    let bits = file
        .numbers
        .iter()
        .map(|n| n.bits())
        .max()
        .expect("no candidates") as u32;
    let gpu = Gpu::open(&bigmont_core::gpu::AdapterSelector::from_env()).expect("no GPU");
    println!("adapter: {}", gpu.describe());
    let tuned = tune::autotune(&gpu, bits, None, true, SchemeChoice::Auto).expect("autotune");
    println!(
        "config: {:?} ({:.0} squarings/s measured by autotune)",
        tuned.cfg, tuned.squarings_per_sec
    );
    // The tuned shape at this exact length (a 32-bit limb count can differ within a tuning class).
    let cfg = tuned.cfg.retarget(bits);
    let mut runner =
        pollster::block_on(Runner::new_async(&gpu, cfg, bits, capacity)).expect("runner");
    let cap = runner.capacity() as usize;
    let t = Instant::now();
    let mut prps = 0usize;
    for chunk in file.numbers.chunks(cap) {
        prps += pollster::block_on(runner.fermat_async(&gpu, chunk))
            .expect("fermat")
            .iter()
            .filter(|&&v| v)
            .count();
    }
    let wall = t.elapsed().as_secs_f64();
    let s = runner.stats;
    println!(
        "{} tests, {bits} bits, capacity {cap}: wall {wall:.3} s ({:.1} tests/s) | gpu {:.3} s ({:.1} tests/s) | host prep {:.3} s ({:.1}% of wall) | PRP {prps}",
        s.tests,
        s.tests as f64 / wall,
        s.gpu_secs,
        s.tests as f64 / s.gpu_secs,
        s.host_prep_secs,
        100.0 * s.host_prep_secs / wall,
    );
}
