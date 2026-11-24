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
//!   cargo run --release -p bigmont-core --example spike -- --variant unrolled    (rolled (default) or unrolled kernel; unrolled skips tpi 1)
//!
//! Each configuration is timed at 2 squarings per dispatch and at progressively longer dispatches
//! (up to a predicted 0.4 s, stopping once one takes 0.2 s), so the fixed per-dispatch cost can be
//! separated from the real throughput.
//!
//! The test suite (`cargo test`) is strict by default: a missing adapter or an unsupported
//! configuration fails the test. `BIGMONT_ALLOW_SKIP=1` turns those cases into loud skips.
//! Run benchmarks in a single process (this example does). The exit status is nonzero if any
//! configuration failed validation, was rejected, or failed to benchmark.

use bigmont_core::gpu::{Batch, Gpu, Run, run};
use bigmont_core::kernel::{KernelConfig, Variant};
use bigmont_oracle::{LIMB_MASK, LIMBS, mont_square_chain, unpack_limbs};
use num_bigint::BigUint;

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

/// Instances checked against the oracle after a timed run: the first 4, the last 4 (all below
/// `count`), 16 spread across the batch, and the first instance of the second and of the last
/// workgroup and the middle one. Always includes indices of the first and the last workgroup.
fn sample_indices(count: u32, per_wg: u32) -> Vec<usize> {
    let count = count as usize;
    let mut set = std::collections::BTreeSet::new();
    set.extend(0..4.min(count));
    set.extend(count.saturating_sub(4)..count);
    set.extend((1..=16).map(|i| i * (count - 1) / 17));
    set.extend([per_wg as usize, count - per_wg as usize, count / 2].into_iter().filter(|&i| i < count));
    set.into_iter().collect()
}

/// Compares the output of a timed run (`s` squarings per dispatch, `r` chained dispatches, so
/// `s * r` squarings per instance) with the oracle for a sample of instances. Every sampled
/// instance must have all limbs below 2^13, a value below 2n, and the right value modulo n. All
/// failing sampled instances are listed. Returns the number of instances checked.
fn verify_timed(values: &[(BigUint, BigUint)], per_wg: u32, out: &Run, s: u32, r: u32, what: &str) -> Result<usize, Failure> {
    let squarings = s * r;
    let sample = sample_indices(values.len() as u32, per_wg);
    let mut bad = Vec::new();
    for &i in &sample {
        let (n, x0) = &values[i];
        let limbs = &out.x[i * LIMBS..(i + 1) * LIMBS];
        let got = unpack_limbs(limbs);
        let why = if limbs.iter().any(|&l| l > LIMB_MASK) {
            Some("limbs not normalised")
        } else if got >= n * 2u32 {
            Some("value not below 2n")
        } else if &got % n != mont_square_chain(n, x0, squarings) {
            Some("wrong value")
        } else {
            None
        };
        if let Some(why) = why {
            bad.push(format!("{i} ({why})"));
        }
    }
    if bad.is_empty() {
        Ok(sample.len())
    } else {
        Err(Failure::Wrong(format!(
            "{what}: {} of {} sampled instances (of {}) failed the oracle after {squarings} squarings ({s} per dispatch x {r}); failing instance indices: {}",
            bad.len(),
            sample.len(),
            values.len(),
            bad.join(", ")
        )))
    }
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
    /// Instances compared with the oracle after the short run and after the final long run, and the
    /// squarings per instance those outputs had (squarings per dispatch x chained dispatches).
    verified_short: usize,
    verified_long: usize,
    verified_short_sq: u32,
    verified_long_sq: u32,
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
fn bench(gpu: &Gpu, cfg: KernelConfig, max_count: u32, max_squarings: u32, fixed_count: Option<u32>) -> Result<Row, Failure> {
    let per_wg = cfg.instances_per_workgroup();
    let cap = (max_count / per_wg).max(1) * per_wg;
    // With --fixed-count the batch is exactly that size (rounded down to whole workgroups) and the
    // growth loop is skipped, so rows can be compared at the same instance count.
    let mut count = match fixed_count {
        Some(n) => (n.min(max_count) / per_wg).max(1) * per_wg,
        None => ((1024 / per_wg).max(1) * per_wg).min(cap),
    };
    loop {
        let values = Batch::random_values(count, BITS, 99);
        let batch = Batch::from_values(&values);
        let probe = run(gpu, cfg, &batch, 2, 1).map_err(Failure::Rejected)?;
        if fixed_count.is_some() || probe.best_secs >= 0.05 || count >= cap {
            // `run` chains `repeats` dispatches from the previous output, so the output of
            // run(.., s, r) has had s * r squarings.
            let short = run(gpu, cfg, &batch, 2, 5).map_err(Failure::Rejected)?;
            let short_secs = short.best_secs;
            let verified_short = verify_timed(&values, per_wg, &short, 2, 5, "short run")?;
            let (mut long_squarings, mut long_secs) = (2u32, short_secs);
            let (mut long_run, mut long_repeats) = (short, 5u32);
            while long_secs < LONG_DISPATCH_TARGET {
                let next = long_squarings * 2;
                if next > max_squarings || long_secs * f64::from(next) / f64::from(long_squarings) > LONG_DISPATCH_BUDGET {
                    break;
                }
                long_run = run(gpu, cfg, &batch, next, 3).map_err(Failure::Rejected)?;
                long_secs = long_run.best_secs;
                long_squarings = next;
                long_repeats = 3;
            }
            // The run whose time is reported as `long ms` is `long_run` (the short run when the
            // loop never lengthened it).
            let verified_long = if long_squarings == 2 { verified_short } else { verify_timed(&values, per_wg, &long_run, long_squarings, long_repeats, "long run")? };
            return Ok(Row {
                cfg,
                count,
                short_secs,
                long_squarings,
                long_secs,
                verified_short,
                verified_long,
                verified_short_sq: 2 * 5,
                verified_long_sq: long_squarings * long_repeats,
            });
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
    let variant = match args.iter().position(|a| a == "--variant").map(|i| args.get(i + 1).map(String::as_str)) {
        None => Variant::Rolled,
        Some(Some("rolled")) => Variant::Rolled,
        Some(Some("unrolled")) => Variant::Unrolled,
        Some(_) => panic!("--variant needs rolled or unrolled"),
    };
    let variant_name = match variant {
        Variant::Rolled => "rolled",
        Variant::Unrolled => "unrolled",
    };
    let gpu = Gpu::new().expect("no usable adapter");
    // Each instance needs LIMBS * 4 bytes in each storage buffer; keep the largest buffer bindable.
    let bytes_per_instance = (LIMBS * 4) as u64;
    let bind_limit = (gpu.device.limits().max_storage_buffer_binding_size / bytes_per_instance).min(u64::from(u32::MAX)) as u32;
    let requested = max_count;
    let max_count = max_count.min(bind_limit);
    println!("adapter: {}", gpu.describe());
    println!("variant: {variant_name}");
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
        None => vec![1, 4, 8, 16, 32],
    };
    let wgs: &[u32] = if quick { &[64] } else { &[64, 128, 256] };
    let mut failures = 0u32;
    println!(
        "{:>8} {:>4} {:>4} {:>9} {:>5} {:>9} {:>12} {:>12} {:>12} {:>8} {:>9} {:>11}  status",
        "variant", "tpi", "wg", "instances", "sq/d", "long ms", "sq/s @2", "sq/s @long", "sq/s slope", "fixed ms", "Gmad/s@L", "tests/s*@L"
    );
    for &tpi in tpis.iter() {
        for &wg in wgs {
            let cfg = KernelConfig { tpi, wg, variant };
            if let Err(e) = cfg.validate() {
                println!("{variant_name:>8} {tpi:>4} {wg:>4}  skipped: {e}");
                continue;
            }
            // The batch can grow to max_count, rounded down to whole workgroups; a dispatch is limited to
            // 65535 workgroups in x.
            let per_wg = cfg.instances_per_workgroup();
            if max_count / per_wg > 65535 {
                println!("{variant_name:>8} {tpi:>4} {wg:>4}  skipped: {} workgroups at --max-count {max_count} exceeds 65535 per dispatch", max_count / per_wg);
                continue;
            }
            if cfg.needs_subgroups() {
                if let Err(e) = gpu.check_subgroups(tpi) {
                    println!("{variant_name:>8} {tpi:>4} {wg:>4}  skipped: {e}");
                    continue;
                }
            }
            match validate(&gpu, cfg) {
                Ok(()) => {}
                Err(Failure::Rejected(e)) => {
                    println!("{variant_name:>8} {tpi:>4} {wg:>4}  REJECTED: {e} (no speed reported)");
                    failures += 1;
                    continue;
                }
                Err(Failure::Wrong(e)) => {
                    println!("{variant_name:>8} {tpi:>4} {wg:>4}  VALIDATION FAILED: {e} (no speed reported)");
                    failures += 1;
                    continue;
                }
            }
            match bench(&gpu, cfg, max_count, max_squarings, fixed_count) {
                Ok(r) => {
                    let verified = format!("ok, verified {} inst x {} sq (long), {} inst x {} sq (short)", r.verified_long, r.verified_long_sq, r.verified_short, r.verified_short_sq);
                    let long = r.rate_long();
                    let slope = r.rate_slope().map_or("n/a".to_string(), |s| format!("{s:.0}"));
                    let fixed = r.fixed_ms().map_or("n/a".to_string(), |f| format!("{f:.1}"));
                    println!(
                        "{:>8} {:>4} {:>4} {:>9} {:>5} {:>9.2} {:>12.0} {:>12.0} {:>12} {:>8} {:>9.1} {:>11.0}  {}",
                        variant_name,
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
                        long / f64::from(BITS),
                        verified
                    );
                }
                Err(Failure::Rejected(e)) => {
                    println!("{variant_name:>8} {tpi:>4} {wg:>4}  bench failed: {e}");
                    failures += 1;
                }
                Err(Failure::Wrong(e)) => {
                    println!("{variant_name:>8} {tpi:>4} {wg:>4}  TIMED OUTPUT FAILED ORACLE: {e} (no speed reported)");
                    failures += 1;
                }
            }
        }
    }
    println!();
    println!("Each row's timed outputs were compared with the oracle (first 4, last 4 and spread instances; limbs < 2^13, value < 2n, value mod n == mont_square_chain): 'verified K inst x S sq' = K instances checked after S squarings per instance, for the final long run and for the 2 squarings x 5 run. A row that fails prints no speed. Times are wall-clock around submit plus a blocking poll (no GPU timestamps); best of 3 (best of 5 for the short run).");
    println!("sq/d = squarings per dispatch of the longest dispatch measured (its time is 'long ms'); 'sq/s @2' is the original 2-squaring figure (includes the fixed per-dispatch cost).");
    println!("'sq/s @long' = instances x sq/d / long time. 'sq/s slope' = marginal rate between the 2-squaring and the long dispatch (fixed cost removed); 'fixed ms' = the per-dispatch cost that slope implies (n/a when the long dispatch was not measurably longer). Gmad/s@L is derived from sq/s @long.");
    println!("* tests/s*@L = sq/s @long / {BITS}: one squaring per exponent bit of a base-2 Fermat test (assumes the base-2 doubling is nearly free; not implemented here). A projection, not a measurement of a full test.");
    if failures > 0 {
        eprintln!("{failures} configuration(s) failed validation, were rejected, or failed to benchmark");
        std::process::exit(1);
    }
}
