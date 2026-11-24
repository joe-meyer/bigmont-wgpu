//! Phase 0 speed spike: validates each kernel configuration against the oracle, then measures
//! Montgomery squarings per second.
//!
//!   cargo run --release -p bigmont-core --example spike
//!   cargo run --release -p bigmont-core --example spike -- --list
//!   BIGMONT_ADAPTER=6700 BIGMONT_BACKEND=vulkan cargo run --release -p bigmont-core --example spike
//!   cargo run --release -p bigmont-core --example spike -- --quick     (fewer configurations)
//!   cargo run --release -p bigmont-core --example spike -- --max-count 131072   (raise the batch cap)
//!
//! The test suite (`cargo test`) is strict by default: a missing adapter or an unsupported
//! configuration fails the test. `BIGMONT_ALLOW_SKIP=1` turns those cases into loud skips.
//! Run benchmarks in a single process (this example does). The exit status is nonzero if any
//! configuration failed validation, was rejected, or failed to benchmark.

use bigmont_core::gpu::{Batch, Gpu, run};
use bigmont_core::kernel::KernelConfig;
use bigmont_oracle::{LIMB_MASK, LIMBS, mont_square_chain, unpack_limbs};

const BITS: u32 = 2029;
const MAD_PER_SQUARING: f64 = 2.0 * (LIMBS * LIMBS) as f64;

/// Why a configuration did not get a speed.
enum Failure {
    /// The shader, pipeline, buffers or dispatch were rejected or errored (`run` returned Err).
    Rejected(String),
    /// The kernel ran but produced a wrong result.
    Wrong(String),
}

/// Validates one configuration on 50 random instances (3 squarings). Returns Err with details.
fn validate(gpu: &Gpu, cfg: KernelConfig) -> Result<(), Failure> {
    let values = Batch::random_values(50, BITS, 11);
    let batch = Batch::from_values(&values);
    let out = run(gpu, cfg, &batch, 3, 1).map_err(Failure::Rejected)?;
    for (i, (n, x0)) in values.iter().enumerate() {
        let limbs = &out.x[i * LIMBS..(i + 1) * LIMBS];
        if limbs.iter().any(|&l| l > LIMB_MASK) {
            return Err(Failure::Wrong(format!("instance {i}: limbs not normalised")));
        }
        let got = unpack_limbs(limbs);
        if &got % n != mont_square_chain(n, x0, 3) {
            return Err(Failure::Wrong(format!("instance {i}: wrong value")));
        }
    }
    Ok(())
}

struct Row {
    cfg: KernelConfig,
    count: u32,
    squarings: u32,
    best: f64,
}

/// Grows the batch until one dispatch takes at least 50 ms (staying far below the plan's 0.5 s
/// per-dispatch budget) or the batch reaches `max_count`, then takes the best of 5 dispatches.
fn bench(gpu: &Gpu, cfg: KernelConfig, max_count: u32) -> Result<Row, String> {
    let squarings = 2;
    let per_wg = cfg.instances_per_workgroup();
    let cap = (max_count / per_wg).max(1) * per_wg;
    let mut count = ((1024 / per_wg).max(1) * per_wg).min(cap);
    loop {
        let batch = Batch::from_values(&Batch::random_values(count, BITS, 99));
        let probe = run(gpu, cfg, &batch, squarings, 1)?;
        if probe.best_secs >= 0.05 || count >= cap {
            let timed = run(gpu, cfg, &batch, squarings, 5)?;
            return Ok(Row { cfg, count, squarings, best: timed.best_secs });
        }
        count = (count * 2).min(cap);
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--list") {
        for a in Gpu::list_adapters() {
            println!("{a}");
        }
        return;
    }
    let quick = args.iter().any(|a| a == "--quick");
    let max_count: u32 = match args.iter().position(|a| a == "--max-count") {
        Some(i) => args.get(i + 1).and_then(|v| v.parse().ok()).filter(|&n| (1024..=262_144).contains(&n)).expect("--max-count needs a number from 1024 to 262144"),
        None => 65536,
    };
    let gpu = Gpu::new().expect("no usable adapter");
    // Each instance needs LIMBS * 4 bytes in each storage buffer; keep the largest buffer bindable.
    let bytes_per_instance = (LIMBS * 4) as u64;
    let bind_limit = (gpu.device.limits().max_storage_buffer_binding_size / bytes_per_instance).min(u64::from(u32::MAX)) as u32;
    let requested = max_count;
    let max_count = max_count.min(bind_limit);
    println!("adapter: {}", gpu.describe());
    println!("subgroups feature: {}", gpu.has_subgroups);
    println!("max batch (instances): {max_count}");
    if max_count < requested {
        println!("note: --max-count {requested} clamped to {max_count} (max_storage_buffer_binding_size {} / {bytes_per_instance} bytes per instance)", gpu.device.limits().max_storage_buffer_binding_size);
    }
    println!();

    let tpis = [1u32, 8, 16, 32];
    let wgs: &[u32] = if quick { &[64] } else { &[64, 128, 256] };
    let mut failures = 0u32;
    println!("{:>4} {:>4} {:>9} {:>5} {:>9} {:>12} {:>10} {:>12}  status", "tpi", "wg", "instances", "sq/d", "best ms", "squarings/s", "Gmad/s", "tests/s*");
    for &tpi in &tpis {
        for &wg in wgs {
            let cfg = KernelConfig { tpi, wg };
            if let Err(e) = cfg.validate() {
                println!("{tpi:>4} {wg:>4}  skipped: {e}");
                continue;
            }
            // The batch can grow to max_count, rounded down to whole workgroups; a dispatch is limited to
            // 65535 workgroups in x.
            let per_wg = cfg.instances_per_workgroup();
            if max_count / per_wg > 65535 {
                println!("{tpi:>4} {wg:>4}  skipped: {} workgroups at --max-count {max_count} exceeds 65535 per dispatch", max_count / per_wg);
                continue;
            }
            if cfg.needs_subgroups() {
                if let Err(e) = gpu.check_subgroups(tpi) {
                    println!("{tpi:>4} {wg:>4}  skipped: {e}");
                    continue;
                }
            }
            match validate(&gpu, cfg) {
                Ok(()) => {}
                Err(Failure::Rejected(e)) => {
                    println!("{tpi:>4} {wg:>4}  REJECTED: {e} (no speed reported)");
                    failures += 1;
                    continue;
                }
                Err(Failure::Wrong(e)) => {
                    println!("{tpi:>4} {wg:>4}  VALIDATION FAILED: {e} (no speed reported)");
                    failures += 1;
                    continue;
                }
            }
            match bench(&gpu, cfg, max_count) {
                Ok(r) => {
                    let sps = f64::from(r.count) * f64::from(r.squarings) / r.best;
                    println!(
                        "{:>4} {:>4} {:>9} {:>5} {:>9.2} {:>12.0} {:>10.1} {:>12.0}  ok",
                        r.cfg.tpi,
                        r.cfg.wg,
                        r.count,
                        r.squarings,
                        r.best * 1e3,
                        sps,
                        sps * MAD_PER_SQUARING / 1e9,
                        sps / f64::from(BITS)
                    );
                }
                Err(e) => {
                    println!("{tpi:>4} {wg:>4}  bench failed: {e}");
                    failures += 1;
                }
            }
        }
    }
    println!();
    println!("* tests/s = squarings/s / {BITS}: one squaring per exponent bit of a base-2 Fermat test (doubling folded into the reduction). A projection, not a measurement of a full test.");
    if failures > 0 {
        eprintln!("{failures} configuration(s) failed validation, were rejected, or failed to benchmark");
        std::process::exit(1);
    }
}
