//! Autotuning and per-adapter self-validation: every kernel configuration is checked against the
//! CPU references before it is timed, and the fastest verified one is cached per adapter and size.

use bigmont_core::gpu::{Batch, Gpu, run};
use bigmont_core::kernel::{KernelConfig, Variant};
use bigmont_core::runner::Runner;
use bigmont_oracle::{LIMB_MASK, XorShift64, limbs_for_bits, mont_square_chain_n, random_odd_modulus, unpack_limbs};
use num_bigint::BigUint;
use num_traits::{One, Zero};
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

const TPIS: [u32; 4] = [4, 8, 16, 32];
const WGS: [u32; 3] = [64, 128, 256];
/// Workgroups in the timing dispatch on a GPU: enough to saturate it (4 groups measured only latency
/// and ranked configurations by workgroup size). A CPU adapter gets `CPU_MEASURE_GROUPS`.
const MEASURE_GROUPS: u32 = 256;
const CPU_MEASURE_GROUPS: u32 = 4;
/// Grow the timed dispatch until it takes at least this long.
const MEASURE_TARGET_SECS: f64 = 0.05;

/// Unrolled configurations for `bits`-bit numbers that fit this adapter's subgroups. Empty when the
/// adapter has no subgroup support.
pub fn candidates(gpu: &Gpu, bits: u32) -> Vec<KernelConfig> {
    let mut out = Vec::new();
    for tpi in TPIS {
        if gpu.check_subgroups(tpi).is_err() {
            continue;
        }
        for wg in WGS {
            let cfg = KernelConfig::for_bits(bits, tpi, wg, Variant::Unrolled);
            if cfg.validate().is_ok() {
                out.push(cfg);
            }
        }
    }
    out
}

/// Fermat test inputs for `bits`-bit moduli with their reference verdicts (computed once per size).
struct FermatSet {
    ns: Vec<BigUint>,
    expect: Vec<bool>,
    /// Number of leading entries (the random numbers and the probable primes) whose verdicts count
    /// towards the "not vacuous" check; the trailing all-ones number's verdict is not assumed.
    counted: usize,
}

fn small_primes() -> Vec<u32> {
    (3..1000u32).filter(|p| (2..*p).take_while(|d| d * d <= *p).all(|d| p % d != 0)).collect()
}

fn fermat_set(bits: u32) -> Arc<FermatSet> {
    static CACHE: Mutex<Option<HashMap<u32, Arc<FermatSet>>>> = Mutex::new(None);
    let mut guard = CACHE.lock().unwrap_or_else(|e| e.into_inner());
    let cache = guard.get_or_insert_with(HashMap::new);
    if let Some(set) = cache.get(&bits) {
        return set.clone();
    }
    let mut rng = XorShift64::new(0xFE12_3A5E ^ u64::from(bits));
    let mut ns: Vec<BigUint> = (0..14).map(|_| random_odd_modulus(&mut rng, bits)).collect();
    // Probable primes found here (bigmont_oracle::random_prp is slow at 2,029 bits): small-prime
    // filter, then the CPU Fermat test.
    let primes = small_primes();
    let mut found = 0;
    while found < 3 {
        let n = random_odd_modulus(&mut rng, bits);
        if primes.iter().any(|&p| (&n % p).is_zero()) {
            continue;
        }
        if bigmont_cpu::fermat_base2(&n) {
            ns.push(n);
            found += 1;
        }
    }
    let counted = ns.len();
    ns.push((BigUint::one() << bits) - 1u32);
    let expect = ns.iter().map(bigmont_cpu::fermat_base2).collect();
    let set = Arc::new(FermatSet { ns, expect, counted });
    cache.insert(bits, set.clone());
    set
}

/// Checks `cfg` on this adapter against the CPU references: chained Montgomery squarings against
/// the oracle, then Fermat verdicts for `bits`-bit numbers against `bigmont_cpu::fermat_base2`.
pub fn self_test(gpu: &Gpu, cfg: KernelConfig, bits: u32) -> Result<(), String> {
    self_test_inner(gpu, cfg, bits).map_err(|e| format!("{cfg:?}: {e}"))
}

fn self_test_inner(gpu: &Gpu, cfg: KernelConfig, bits: u32) -> Result<(), String> {
    cfg.validate()?;
    if 13 * cfg.limbs < bits + 4 {
        return Err(format!("{} limbs are too few for {bits}-bit numbers (need R > 16n)", cfg.limbs));
    }
    let limbs = cfg.limbs as usize;

    // (a) squaring mode: R > 4n moduli, plus the all-ones modulus.
    let bits_sq = cfg.limbs * 13 - 3;
    let mut values = Batch::random_values(24, bits_sq, 0x5EED_0001);
    let all_ones = (BigUint::one() << bits_sq) - 1u32;
    values.push((all_ones.clone(), &all_ones - 1u32));
    let squarings = 3;
    let batch = Batch::from_values_n(&values, limbs);
    let result = run(gpu, cfg, &batch, squarings, 1)?;
    for (i, (n, x0)) in values.iter().enumerate() {
        let got_limbs = &result.x[i * limbs..(i + 1) * limbs];
        if got_limbs.iter().any(|&l| l > LIMB_MASK) {
            return Err(format!("squaring instance {i}: limbs are not normalised"));
        }
        let got = unpack_limbs(got_limbs);
        if got >= n * 2u32 {
            return Err(format!("squaring instance {i}: result is not below 2n"));
        }
        if &got % n != mont_square_chain_n(n, x0, squarings, limbs) {
            return Err(format!("squaring instance {i}: wrong value"));
        }
    }

    // (b) Fermat ladder: composites, probable primes and the all-ones number, in several chunks.
    let set = fermat_set(bits);
    let mut runner = Runner::new(gpu, cfg, bits, 8)?;
    let verdicts = runner.fermat(gpu, &set.ns)?;
    for (i, (got, want)) in verdicts.iter().zip(&set.expect).enumerate() {
        if got != want {
            return Err(format!("Fermat candidate {i}: GPU says {got}, CPU says {want}"));
        }
    }
    let counted = &verdicts[..set.counted];
    if !counted.iter().any(|&v| v) || counted.iter().all(|&v| v) {
        return Err("self-test is vacuous".into());
    }
    Ok(())
}

/// Verified squarings per second for `cfg` (one squaring = one Montgomery squaring of one number).
/// Work grows from a small dispatch, so no single submission runs long enough to trip a driver
/// watchdog.
pub fn measure(gpu: &Gpu, cfg: KernelConfig) -> Result<f64, String> {
    cfg.validate()?;
    let limbs = cfg.limbs as usize;
    let on_cpu = gpu.info.device_type == wgpu::DeviceType::Cpu;
    let count = cfg.instances_per_workgroup() * if on_cpu { CPU_MEASURE_GROUPS } else { MEASURE_GROUPS };
    let max_work: u64 = if on_cpu { 1 << 18 } else { 1 << 26 };
    let values = Batch::random_values(count, cfg.limbs * 13 - 3, 0x5EED_0002);
    let batch = Batch::from_values_n(&values, limbs);

    let mut squarings = 8u32;
    loop {
        let t = run(gpu, cfg, &batch, squarings, 1)?.best_secs;
        if t >= MEASURE_TARGET_SECS || u64::from(count) * u64::from(squarings) >= max_work {
            break;
        }
        squarings *= 2;
    }
    let mut best = f64::MAX;
    let mut last = None;
    for _ in 0..2 {
        let r = run(gpu, cfg, &batch, squarings, 1)?;
        best = best.min(r.best_secs);
        last = Some(r);
    }
    let last = last.expect("two timed runs");
    for i in (0..count as usize).step_by((count as usize / 4).max(1)).take(4) {
        let (n, x0) = &values[i];
        let got = unpack_limbs(&last.x[i * limbs..(i + 1) * limbs]);
        if &got % n != mont_square_chain_n(n, x0, squarings, limbs) {
            return Err(format!("timing run: instance {i} differs from the oracle"));
        }
    }
    Ok(f64::from(count) * f64::from(squarings) / best)
}

#[derive(Clone, Copy, Debug)]
pub struct Tuned {
    pub cfg: KernelConfig,
    pub squarings_per_sec: f64,
    pub from_cache: bool,
}

fn clean(s: &str) -> String {
    s.chars().map(|c| if matches!(c, '|' | '\n' | '\r') { '/' } else { c }).collect()
}

fn cache_key(gpu: &Gpu, bits: u32) -> String {
    let i = &gpu.info;
    format!("{}|{:?}|{} {}|{}", clean(&i.name), i.backend, clean(&i.driver), clean(&i.driver_info), limbs_for_bits(bits))
}

fn cache_value(cfg: &KernelConfig, rate: f64) -> String {
    format!("{} {} {}|{}", if cfg.variant == Variant::Unrolled { "unrolled" } else { "rolled" }, cfg.tpi, cfg.wg, rate)
}

/// Parses `unrolled <tpi> <wg>|<rate>`. The caller looks the pair up among the candidates, so
/// corrupt cache text never reaches `KernelConfig` arithmetic.
fn parse_value(v: &str) -> Option<(u32, u32, f64)> {
    let (c, rate) = v.split_once('|')?;
    let mut f = c.split_whitespace();
    if f.next()? != "unrolled" {
        return None;
    }
    let (tpi, wg) = (f.next()?.parse().ok()?, f.next()?.parse().ok()?);
    let rate: f64 = rate.trim().parse().ok()?;
    (rate.is_finite() && rate > 0.0).then_some((tpi, wg, rate))
}

fn read_cache(path: &Path) -> Vec<String> {
    std::fs::read_to_string(path).map(|s| s.lines().map(str::to_string).collect()).unwrap_or_default()
}

fn write_cache(path: &Path, key: &str, value: &str) {
    let prefix = format!("{key}|");
    let mut lines: Vec<String> = read_cache(path).into_iter().filter(|l| !l.starts_with(&prefix)).collect();
    lines.push(format!("{prefix}{value}"));
    let result = (|| {
        if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir)?;
        }
        std::fs::write(path, lines.join("\n") + "\n")
    })();
    if let Err(e) = result {
        eprintln!("cannot write tuning cache {}: {e}", path.display());
    }
}

/// Picks the fastest configuration that passes the self-test, reusing `cache` unless `retune`.
/// `Err` when none passes (the engine then falls back to the CPU).
pub fn autotune(gpu: &Gpu, bits: u32, cache: Option<&Path>, retune: bool) -> Result<Tuned, String> {
    let cands = candidates(gpu, bits);
    if cands.is_empty() {
        let why = gpu.check_subgroups(4).err().unwrap_or_else(|| "no configuration fits".into());
        return Err(format!("no usable kernel configuration on {} ({why})", gpu.describe()));
    }
    let key = cache_key(gpu, bits);
    if let (Some(path), false) = (cache, retune) {
        let prefix = format!("{key}|");
        let hit = read_cache(path).iter().find_map(|l| {
            let (tpi, wg, rate) = l.strip_prefix(&prefix).and_then(parse_value)?;
            cands.iter().find(|c| c.tpi == tpi && c.wg == wg).map(|&cfg| (cfg, rate))
        });
        if let Some((cfg, rate)) = hit {
            match self_test(gpu, cfg, bits) {
                Ok(()) => return Ok(Tuned { cfg, squarings_per_sec: rate, from_cache: true }),
                Err(e) => eprintln!("cached {cfg:?} failed its self-test ({e}); retuning"),
            }
        }
    }
    let mut best: Option<Tuned> = None;
    let mut reasons = Vec::new();
    for cfg in cands {
        match self_test(gpu, cfg, bits).and_then(|()| measure(gpu, cfg).map_err(|e| format!("{cfg:?}: {e}"))) {
            Ok(rate) => {
                eprintln!("tune {cfg:?}: {rate:.0} squarings/s");
                if best.is_none_or(|b| rate > b.squarings_per_sec) {
                    best = Some(Tuned { cfg, squarings_per_sec: rate, from_cache: false });
                }
            }
            Err(e) => {
                eprintln!("skip {e}");
                reasons.push(e);
            }
        }
    }
    let best = best.ok_or_else(|| format!("no kernel configuration passed the self-test on {}: {}", gpu.describe(), reasons.join("; ")))?;
    if let Some(path) = cache {
        write_cache(path, &key, &cache_value(&best.cfg, best.squarings_per_sec));
    }
    Ok(best)
}
