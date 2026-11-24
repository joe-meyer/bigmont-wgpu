//! Phase 0 speed spike: validates each kernel configuration against the oracle, then measures
//! Montgomery squarings per second.
//!
//!   cargo run --release -p bigmont-core --example spike
//!   cargo run --release -p bigmont-core --example spike -- --list
//!   BIGMONT_ADAPTER=6700 BIGMONT_BACKEND=vulkan cargo run --release -p bigmont-core --example spike
//!   cargo run --release -p bigmont-core --example spike -- --quick     (fewer configurations)
//!
//! Any run that is meant to count as validation should set `BIGMONT_REQUIRE_GPU=1` for the test
//! suite (`cargo test` skips silently without a GPU otherwise). Run benchmarks in a single process
//! (this example does).

use bigmont_core::gpu::{Batch, Gpu, run};
use bigmont_core::kernel::KernelConfig;
use bigmont_oracle::{LIMB_MASK, LIMBS, mont_square_chain, unpack_limbs};

const BITS: u32 = 2029;
const MAD_PER_SQUARING: f64 = 2.0 * (LIMBS * LIMBS) as f64;

/// Validates one configuration on 50 random instances (3 squarings). Returns Err with details.
fn validate(gpu: &Gpu, cfg: KernelConfig) -> Result<(), String> {
    let values = Batch::random_values(50, BITS, 11);
    let batch = Batch::from_values(&values);
    let out = run(gpu, cfg, &batch, 3, 1)?;
    for (i, (n, x0)) in values.iter().enumerate() {
        let limbs = &out.x[i * LIMBS..(i + 1) * LIMBS];
        if limbs.iter().any(|&l| l > LIMB_MASK) {
            return Err(format!("instance {i}: limbs not normalised"));
        }
        let got = unpack_limbs(limbs);
        if &got % n != mont_square_chain(n, x0, 3) {
            return Err(format!("instance {i}: wrong value"));
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

/// Grows the batch until one dispatch takes at least 50 ms (staying far below the ~2 s Windows
/// GPU timeout), then takes the best of 5 dispatches.
fn bench(gpu: &Gpu, cfg: KernelConfig) -> Result<Row, String> {
    let squarings = 2;
    let per_wg = cfg.instances_per_workgroup();
    let mut count = (1024 / per_wg).max(1) * per_wg;
    loop {
        let batch = Batch::from_values(&Batch::random_values(count, BITS, 99));
        let probe = run(gpu, cfg, &batch, squarings, 1)?;
        if probe.best_secs >= 0.05 || count >= 65536 {
            let timed = run(gpu, cfg, &batch, squarings, 5)?;
            return Ok(Row { cfg, count, squarings, best: timed.best_secs });
        }
        count *= 2;
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
    let gpu = Gpu::new().expect("no usable adapter");
    println!("adapter: {}", gpu.describe());
    println!("subgroups feature: {}", gpu.has_subgroups);
    println!();

    let tpis = [1u32, 8, 16, 32];
    let wgs: &[u32] = if quick { &[64] } else { &[64, 128, 256] };
    println!("{:>4} {:>4} {:>9} {:>5} {:>9} {:>12} {:>10} {:>12}  status", "tpi", "wg", "instances", "sq/d", "best ms", "squarings/s", "Gmad/s", "tests/s*");
    for &tpi in &tpis {
        for &wg in wgs {
            let cfg = KernelConfig { tpi, wg };
            if let Err(e) = cfg.validate() {
                println!("{tpi:>4} {wg:>4}  skipped: {e}");
                continue;
            }
            if cfg.needs_subgroups() {
                if let Err(e) = gpu.check_subgroups(tpi) {
                    println!("{tpi:>4} {wg:>4}  skipped: {e}");
                    continue;
                }
            }
            if let Err(e) = validate(&gpu, cfg) {
                println!("{tpi:>4} {wg:>4}  VALIDATION FAILED: {e} (no speed reported)");
                continue;
            }
            match bench(&gpu, cfg) {
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
                Err(e) => println!("{tpi:>4} {wg:>4}  bench failed: {e}"),
            }
        }
    }
    println!();
    println!("* tests/s = squarings/s / {BITS}: one squaring per exponent bit of a base-2 Fermat test (doubling folded into the reduction). A projection, not a measurement of a full test.");
}
