//! Phase 0 speed spike: validates each kernel configuration against the oracle, then measures
//! Montgomery squarings per second.
//!
//!   cargo run --release -p bigmont-core --example spike
//!   cargo run --release -p bigmont-core --example spike -- --list
//!   BIGMONT_ADAPTER=6700 BIGMONT_BACKEND=vulkan cargo run --release -p bigmont-core --example spike
//!   cargo run --release -p bigmont-core --example spike -- --quick     (fewer configurations)
//!   cargo run --release -p bigmont-core --example spike -- --max-count 131072   (raise the batch cap)
//!   cargo run --release -p bigmont-core --example spike -- --max-squarings 64    (cap the amortisation, default 256)
//!   cargo run --release -p bigmont-core --example spike -- --fixed-count 65536   (same batch size for every row; needs --max-count >= it to avoid clamping)
//!
//! Each configuration is timed at 2 squarings per dispatch and at progressively longer dispatches
//! (up to a predicted 0.4 s, stopping once one takes 0.2 s), so the fixed per-dispatch cost can be
//! separated from the real throughput.
//!
//! The test suite (`cargo test`) is strict by default: a missing adapter or an unsupported
//! configuration fails the test. `BIGMONT_ALLOW_SKIP=1` turns those cases into loud skips.
//! Run benchmarks in a single process (this example does). The exit status is nonzero if any
//! configuration failed validation, was rejected, or failed to benchmark.

use bigmont_core::gpu::{Batch, Gpu, run};
use bigmont_core::kernel::{KernelConfig, Variant};
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

/// Longest dispatch the amortisation loop may plan for (the plan's budget is 0.5 s per dispatch).
const LONG_DISPATCH_BUDGET: f64 = 0.4;
/// Stop lengthening once a dispatch takes this long: the fixed cost is then a small fraction.
const LONG_DISPATCH_TARGET: f64 = 0.2;

struct Row {
    cfg: KernelConfig,
    count: u32,
    /// Best time of a dispatch of 2 squarings (the original measurement).
    short_secs: f64,
    /// Squarings per dispatch and best time of the longest dispatch measured.
    long_squarings: u32,
    long_secs: f64,
}

impl Row {
    /// Squarings per second at 2 squarings per dispatch (includes the fixed per-dispatch cost).
    fn rate_short(&self) -> f64 {
        f64::from(self.count) * 2.0 / self.short_secs
    }
    /// Squarings per second at the longest dispatch (fixed cost amortised over many squarings).
    fn rate_long(&self) -> f64 {
        f64::from(self.count) * f64::from(self.long_squarings) / self.long_secs
    }
    /// Marginal squarings per second between the short and the long dispatch (fixed cost removed),
    /// or None when the long dispatch was not measurably longer.
    fn rate_slope(&self) -> Option<f64> {
        let extra = self.long_secs - self.short_secs;
        (self.long_squarings > 2 && extra > 0.02 * self.short_secs).then(|| f64::from(self.count) * f64::from(self.long_squarings - 2) / extra)
    }
    /// Fixed cost per dispatch implied by the slope, in ms (clamped at 0).
    fn fixed_ms(&self) -> Option<f64> {
        self.rate_slope().map(|s| ((self.short_secs - 2.0 * f64::from(self.count) / s) * 1e3).max(0.0))
    }
}

/// Grows the batch until one dispatch of 2 squarings takes at least 50 ms (staying far below the
/// plan's 0.5 s per-dispatch budget) or the batch reaches `max_count`, takes the best of 5
/// dispatches, then doubles the squarings per dispatch (up to `max_squarings`) while the next
/// dispatch is predicted to stay within LONG_DISPATCH_BUDGET (assuming linear scaling, the worst
/// case) and the current one is shorter than LONG_DISPATCH_TARGET. Each longer dispatch is the best
/// of 3.
fn bench(gpu: &Gpu, cfg: KernelConfig, max_count: u32, max_squarings: u32, fixed_count: Option<u32>) -> Result<Row, String> {
    let per_wg = cfg.instances_per_workgroup();
    let cap = (max_count / per_wg).max(1) * per_wg;
    // With --fixed-count the batch is exactly that size (rounded down to whole workgroups) and the
    // growth loop is skipped, so rows can be compared at the same instance count.
    let mut count = match fixed_count {
        Some(n) => (n.min(max_count) / per_wg).max(1) * per_wg,
        None => ((1024 / per_wg).max(1) * per_wg).min(cap),
    };
    loop {
        let batch = Batch::from_values(&Batch::random_values(count, BITS, 99));
        let probe = run(gpu, cfg, &batch, 2, 1)?;
        if fixed_count.is_some() || probe.best_secs >= 0.05 || count >= cap {
            let short_secs = run(gpu, cfg, &batch, 2, 5)?.best_secs;
            let (mut long_squarings, mut long_secs) = (2u32, short_secs);
            while long_secs < LONG_DISPATCH_TARGET {
                let next = long_squarings * 2;
                if next > max_squarings || long_secs * f64::from(next) / f64::from(long_squarings) > LONG_DISPATCH_BUDGET {
                    break;
                }
                long_secs = run(gpu, cfg, &batch, next, 3)?.best_secs;
                long_squarings = next;
            }
            return Ok(Row { cfg, count, short_secs, long_squarings, long_secs });
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
    let max_squarings: u32 = match args.iter().position(|a| a == "--max-squarings") {
        Some(i) => args.get(i + 1).and_then(|v| v.parse().ok()).filter(|&n| (2..=1024).contains(&n)).expect("--max-squarings needs a number from 2 to 1024"),
        None => 256,
    };
    let fixed_count: Option<u32> = args.iter().position(|a| a == "--fixed-count").map(|i| {
        args.get(i + 1).and_then(|v| v.parse().ok()).filter(|&n| (64..=262_144).contains(&n)).expect("--fixed-count needs a number from 64 to 262144")
    });
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

    // --tpi 8,32 restricts the widths (useful with --fixed-count: a large batch of one-thread
    // instances would exceed the per-dispatch time budget).
    let tpis: Vec<u32> = match args.iter().position(|a| a == "--tpi") {
        Some(i) => args.get(i + 1).map(|v| v.split(',').map(|t| t.parse().expect("--tpi needs a comma list like 8,32")).collect()).expect("--tpi needs a list"),
        None => vec![1, 8, 16, 32],
    };
    let wgs: &[u32] = if quick { &[64] } else { &[64, 128, 256] };
    let mut failures = 0u32;
    println!(
        "{:>4} {:>4} {:>9} {:>5} {:>9} {:>12} {:>12} {:>12} {:>8} {:>9} {:>11}  status",
        "tpi", "wg", "instances", "sq/d", "long ms", "sq/s @2", "sq/s @long", "sq/s slope", "fixed ms", "Gmad/s@L", "tests/s*@L"
    );
    for &tpi in tpis.iter() {
        for &wg in wgs {
            let cfg = KernelConfig { tpi, wg, variant: Variant::Rolled };
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
            match bench(&gpu, cfg, max_count, max_squarings, fixed_count) {
                Ok(r) => {
                    let long = r.rate_long();
                    let slope = r.rate_slope().map_or("n/a".to_string(), |s| format!("{s:.0}"));
                    let fixed = r.fixed_ms().map_or("n/a".to_string(), |f| format!("{f:.1}"));
                    println!(
                        "{:>4} {:>4} {:>9} {:>5} {:>9.2} {:>12.0} {:>12.0} {:>12} {:>8} {:>9.1} {:>11.0}  ok",
                        r.cfg.tpi,
                        r.cfg.wg,
                        r.count,
                        r.long_squarings,
                        r.long_secs * 1e3,
                        r.rate_short(),
                        long,
                        slope,
                        fixed,
                        long * MAD_PER_SQUARING / 1e9,
                        long / f64::from(BITS)
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
    println!("sq/d = squarings per dispatch of the longest dispatch measured (its time is 'long ms'); 'sq/s @2' is the original 2-squaring figure (includes the fixed per-dispatch cost).");
    println!("'sq/s @long' = instances x sq/d / long time. 'sq/s slope' = marginal rate between the 2-squaring and the long dispatch (fixed cost removed); 'fixed ms' = the per-dispatch cost that slope implies (n/a when the long dispatch was not measurably longer). Gmad/s@L is derived from sq/s @long.");
    println!("* tests/s*@L = sq/s @long / {BITS}: one squaring per exponent bit of a base-2 Fermat test (assumes the base-2 doubling is nearly free; not implemented here). A projection, not a measurement of a full test.");
    if failures > 0 {
        eprintln!("{failures} configuration(s) failed validation, were rejected, or failed to benchmark");
        std::process::exit(1);
    }
}
