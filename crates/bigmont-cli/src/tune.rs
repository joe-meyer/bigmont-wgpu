//! Autotuning and per-adapter self-validation: every kernel configuration is checked against the
//! CPU references before it is timed, and the fastest verified one is cached per adapter and size.

use bigmont_core::gpu::{Batch, Gpu, run_async};
use bigmont_core::kernel::{KernelConfig, Variant};
use bigmont_core::runner::Runner;
use bigmont_core::scheme::LimbScheme;
use bigmont_oracle::{LIMB_BITS, XorShift64, limbs_for_bits, random_odd_modulus};
use num_bigint::BigUint;
use num_traits::{One, Zero};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

const TPIS: [u32; 4] = [4, 8, 16, 32];
const WGS: [u32; 3] = [64, 128, 256];
/// Workgroups in the timing dispatch on a GPU: enough to saturate it (4 groups measured only latency
/// and ranked configurations by workgroup size). A CPU adapter gets `CPU_MEASURE_GROUPS`.
const MEASURE_GROUPS: u32 = 256;
const CPU_MEASURE_GROUPS: u32 = 4;
/// Grow the timed dispatch until it takes at least this long.
const MEASURE_TARGET_SECS: f64 = 0.05;

/// The smallest limb class: the unrolled kernels need at least `tpi` limbs and the narrowest width
/// is 4, so every bit length that needs four limbs or fewer (up to 48 bits, down to the 2-bit number
/// 3) builds the same kernels.
const MIN_CLASS_LIMBS: u32 = 4;
/// Self-tests need random moduli that are sometimes prime and sometimes not, which requires room for
/// them: a 2-bit modulus has exactly one value and 8-bit ones are mostly prime or tiny.
const MIN_SELF_TEST_BITS: u32 = 32;

/// The tuning class of `bits`-bit numbers: the minimum 13-bit limb count `limbs_for_bits(bits)`
/// (R > 16n), at least 4. For the 13-bit scheme every bit length of a class builds the same kernel for
/// each cooperative width (the limb count is that minimum rounded up to a multiple of `tpi`), so the
/// tuned configuration, the cache entry and the self-test are shared by the whole class. A 32-bit
/// kernel's limb count can differ between lengths of one class (it is smaller below the class's
/// representative length): the tuned (scheme, tpi, wg) shape is shared, but the kernel built for an
/// exact length is a different program; see the note on `autotune` and the engine's extra self-test.
pub fn limb_class(bits: u32) -> u32 {
    (limbs_for_bits(bits) as u32).max(MIN_CLASS_LIMBS)
}

/// The bit length autotuning uses to exercise limb class `limbs`: the largest length the class
/// supports (`13 * limbs - 4`, so R > 16n still holds), which is at least 48 and so leaves room for
/// both probable primes and composites even when the class's smallest members are tiny.
pub fn class_bits(limbs: u32) -> u32 {
    LIMB_BITS * limbs - 4
}

/// Which limb schemes the tuner may pick from. `Auto` measures every scheme the adapter can run and
/// takes the fastest; `Only(s)` forces one (the CLI's `--limbs 13|32`). No vendor or device name
/// enters this: capabilities decide what can run, measurement decides what is faster.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SchemeChoice {
    #[default]
    Auto,
    Only(LimbScheme),
}

impl SchemeChoice {
    pub fn allows(self, s: LimbScheme) -> bool {
        match self {
            SchemeChoice::Auto => true,
            SchemeChoice::Only(only) => only == s,
        }
    }

    /// `auto`, `13` or `32`.
    pub fn parse(s: &str) -> Result<SchemeChoice, String> {
        if s == "auto" {
            return Ok(SchemeChoice::Auto);
        }
        LimbScheme::parse(s)
            .map(SchemeChoice::Only)
            .ok_or_else(|| format!("the limb scheme must be auto, 13 or 32, not {s:?}"))
    }
}

/// Unrolled configurations for `bits`-bit numbers that `choice` allows and this adapter can run: every
/// allowed scheme, every (tpi, workgroup) whose lane count fits the adapter's subgroups, and only those
/// whose capability needs (`KernelConfig::requires`) the adapter offers. Empty when nothing fits (no
/// subgroup support, or a forced scheme the adapter lacks the capabilities for).
pub fn candidates(gpu: &Gpu, bits: u32, choice: SchemeChoice) -> Vec<KernelConfig> {
    let have = gpu.caps();
    let mut out = Vec::new();
    for &scheme in LimbScheme::ALL {
        if !choice.allows(scheme) {
            continue;
        }
        for tpi in TPIS {
            if gpu.check_subgroups(tpi).is_err() {
                continue;
            }
            for wg in WGS {
                let cfg = KernelConfig::for_bits_scheme(scheme, bits, tpi, wg, Variant::Unrolled);
                if cfg.validate().is_ok() && have.satisfies(cfg.requires()) {
                    out.push(cfg);
                }
            }
        }
    }
    out
}

/// Why `candidates(gpu, bits, choice)` is empty, for error messages.
fn no_candidates_reason(gpu: &Gpu, bits: u32, choice: SchemeChoice) -> String {
    for &scheme in LimbScheme::ALL {
        if !choice.allows(scheme) {
            continue;
        }
        // The narrowest cooperative kernel is the easiest to satisfy: if it fails, so do the rest.
        if let Err(e) = gpu.check_subgroups(TPIS[0]) {
            return e;
        }
        let probe = KernelConfig::for_bits_scheme(scheme, bits, TPIS[0], WGS[0], Variant::Unrolled);
        if let Err(e) = probe.check_caps(gpu.caps()) {
            return e;
        }
    }
    "no configuration fits".into()
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
    (3..1000u32)
        .filter(|p| (2..*p).take_while(|d| d * d <= *p).all(|d| p % d != 0))
        .collect()
}

fn fermat_set(bits: u32) -> Arc<FermatSet> {
    static CACHE: Mutex<Option<HashMap<u32, Arc<FermatSet>>>> = Mutex::new(None);
    let mut guard = CACHE.lock().unwrap_or_else(|e| e.into_inner());
    let cache = guard.get_or_insert_with(HashMap::new);
    if let Some(set) = cache.get(&bits) {
        return set.clone();
    }
    let mut rng = XorShift64::new(0xFE12_3A5E ^ u64::from(bits));
    let mut ns: Vec<BigUint> = (0..14)
        .map(|_| random_odd_modulus(&mut rng, bits))
        .collect();
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
    let set = Arc::new(FermatSet {
        ns,
        expect,
        counted,
    });
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
    gpu.check_health()?;
    if bits < MIN_SELF_TEST_BITS {
        return Err(format!(
            "the self-test needs at least {MIN_SELF_TEST_BITS}-bit numbers, got {bits} (tune a limb class at tune::class_bits)"
        ));
    }
    if cfg.r_bits() < bits + 4 {
        return Err(format!(
            "{} limbs are too few for {bits}-bit numbers (need R > 16n)",
            cfg.limbs
        ));
    }
    let limbs = cfg.limbs as usize;
    let scheme = cfg.scheme;

    // (a) squaring mode: R > 4n moduli, plus the all-ones modulus.
    let bits_sq = cfg.r_bits() - 3;
    let mut values = Batch::random_values(24, bits_sq, 0x5EED_0001);
    let all_ones = (BigUint::one() << bits_sq) - 1u32;
    values.push((all_ones.clone(), &all_ones - 1u32));
    let squarings = 3;
    let batch = Batch::from_values_cfg(&values, &cfg);
    let result = pollster::block_on(run_async(gpu, cfg, &batch, squarings, 1))?;
    for (i, (n, x0)) in values.iter().enumerate() {
        let got_limbs = &result.x[i * limbs..(i + 1) * limbs];
        if !scheme.normalised(got_limbs) {
            return Err(format!("squaring instance {i}: limbs are not normalised"));
        }
        let got = scheme.unpack(got_limbs);
        if got >= n * 2u32 {
            return Err(format!("squaring instance {i}: result is not below 2n"));
        }
        if &got % n != scheme.chain(n, x0, squarings, limbs) {
            return Err(format!("squaring instance {i}: wrong value"));
        }
    }

    // (b) Fermat ladder: composites, probable primes and the all-ones number, in several chunks.
    let set = fermat_set(bits);
    let mut runner = pollster::block_on(Runner::new_async(gpu, cfg, bits, 8))?;
    let verdicts = pollster::block_on(runner.fermat_async(gpu, &set.ns))?;
    for (i, (got, want)) in verdicts.iter().zip(&set.expect).enumerate() {
        if got != want {
            return Err(format!(
                "Fermat candidate {i}: GPU says {got}, CPU says {want}"
            ));
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
    gpu.check_health()?;
    let limbs = cfg.limbs as usize;
    let on_cpu = gpu.info.device_type == wgpu::DeviceType::Cpu;
    let count = cfg.instances_per_workgroup()
        * if on_cpu {
            CPU_MEASURE_GROUPS
        } else {
            MEASURE_GROUPS
        };
    let max_work: u64 = if on_cpu { 1 << 18 } else { 1 << 26 };
    let values = Batch::random_values(count, cfg.r_bits() - 3, 0x5EED_0002);
    let batch = Batch::from_values_cfg(&values, &cfg);

    let mut squarings = 8u32;
    loop {
        let t = pollster::block_on(run_async(gpu, cfg, &batch, squarings, 1))?.best_secs;
        if t >= MEASURE_TARGET_SECS || u64::from(count) * u64::from(squarings) >= max_work {
            break;
        }
        squarings *= 2;
    }
    let mut best = f64::MAX;
    let mut last = None;
    for _ in 0..2 {
        let r = pollster::block_on(run_async(gpu, cfg, &batch, squarings, 1))?;
        best = best.min(r.best_secs);
        last = Some(r);
    }
    let last = last.expect("two timed runs");
    for i in (0..count as usize)
        .step_by((count as usize / 4).max(1))
        .take(4)
    {
        let (n, x0) = &values[i];
        let got = cfg.scheme.unpack(&last.x[i * limbs..(i + 1) * limbs]);
        if &got % n != cfg.scheme.chain(n, x0, squarings, limbs) {
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
    s.chars()
        .map(|c| {
            if matches!(c, '|' | '\n' | '\r') {
                '/'
            } else {
                c
            }
        })
        .collect()
}

/// The cache key of the limb class of `bits`-bit numbers on an adapter described by these fields:
/// `name|backend|driver driver_info|<choice>|class`, where `<choice>` is `any` for `Auto` and
/// `only-<name>` (`only-13`, `only-32`) for a forced scheme. A forced run therefore has its own entry
/// and never replaces or deletes the `Auto` one.
///
/// Caches written before the limb schemes existed use `name|backend|driver driver_info|class` (no
/// choice field). Those lines never match a key built here, so they are re-measured once: they were
/// tuned among 13-bit kernels only and would otherwise hide a faster 32-bit scheme forever. (Their
/// value text, `unrolled <tpi> <wg>|<rate>`, is unchanged and still parses; only the key moved.) No
/// key's lookup prefix (`key|`) is a prefix of another's: the fields are `|`-separated, field text
/// never contains `|` (see `clean`), the choice field is `any` or `only-..` and the class is numeric.
#[doc(hidden)]
pub fn cache_key_text(
    name: &str,
    backend: &str,
    driver: &str,
    driver_info: &str,
    bits: u32,
    choice: SchemeChoice,
) -> String {
    let which = match choice {
        SchemeChoice::Auto => "any".to_string(),
        SchemeChoice::Only(s) => format!("only-{}", s.name()),
    };
    format!(
        "{}|{backend}|{} {}|{which}|{}",
        clean(name),
        clean(driver),
        clean(driver_info),
        limb_class(bits)
    )
}

#[doc(hidden)]
pub fn cache_key(gpu: &Gpu, bits: u32, choice: SchemeChoice) -> String {
    let i = &gpu.info;
    cache_key_text(
        &i.name,
        &format!("{:?}", i.backend),
        &i.driver,
        &i.driver_info,
        bits,
        choice,
    )
}

/// `unrolled <tpi> <wg>|<rate>` for a 13-bit winner (the text older versions wrote and read) and
/// `unrolled32 <tpi> <wg>|<rate>` for a 32-bit one.
#[doc(hidden)]
pub fn cache_value(cfg: &KernelConfig, rate: f64) -> String {
    let word = match (cfg.variant, cfg.scheme) {
        (Variant::Unrolled, LimbScheme::Limb13) => "unrolled",
        (Variant::Unrolled, LimbScheme::Limb32) => "unrolled32",
        (Variant::Rolled, _) => "rolled",
    };
    format!("{word} {} {}|{}", cfg.tpi, cfg.wg, rate)
}

/// Parses `unrolled|unrolled32 <tpi> <wg>|<rate>`; any other first word is a miss. The caller looks the
/// shape up among the candidates, so corrupt cache text never reaches `KernelConfig` arithmetic.
#[doc(hidden)]
pub fn parse_value(v: &str) -> Option<(LimbScheme, u32, u32, f64)> {
    let (c, rate) = v.split_once('|')?;
    let mut f = c.split_whitespace();
    let scheme = match f.next()? {
        "unrolled" => LimbScheme::Limb13,
        "unrolled32" => LimbScheme::Limb32,
        _ => return None,
    };
    let (tpi, wg) = (f.next()?.parse().ok()?, f.next()?.parse().ok()?);
    let rate: f64 = rate.trim().parse().ok()?;
    (rate.is_finite() && rate > 0.0).then_some((scheme, tpi, wg, rate))
}

/// The cached winner for `key` among `lines` (cache file lines) that is still one of `cands`: the
/// line's scheme, tpi and workgroup must all match a candidate, so a line naming a scheme the adapter
/// cannot run (a 32-bit line on a device without int64), a scheme the choice forbids, or a shape that
/// is no longer offered is a miss. Returns the candidate (not a config rebuilt from the text) and the
/// cached rate.
#[doc(hidden)]
pub fn cache_lookup(
    lines: &[String],
    key: &str,
    cands: &[KernelConfig],
) -> Option<(KernelConfig, f64)> {
    let prefix = format!("{key}|");
    lines.iter().find_map(|l| {
        let (scheme, tpi, wg, rate) = l.strip_prefix(&prefix).and_then(parse_value)?;
        cands
            .iter()
            .find(|c| c.scheme == scheme && c.tpi == tpi && c.wg == wg)
            .map(|&cfg| (cfg, rate))
    })
}

fn read_cache(path: &Path) -> Vec<String> {
    std::fs::read_to_string(path)
        .map(|s| s.lines().map(str::to_string).collect())
        .unwrap_or_default()
}

/// Replaces (or adds) the entry for `key` in the cache file at `path`, creating missing parent
/// directories. The file is written to `<path>.tmp.<pid>` and renamed over `path`, so concurrent
/// readers never see a partial file and several processes can share one cache. (Two processes
/// writing at the same moment can each drop the other's new entry, which only costs a retune.)
/// Failures are reported on stderr and otherwise ignored: the cache is an optimisation.
pub fn write_cache(path: &Path, key: &str, value: &str) {
    let prefix = format!("{key}|");
    let mut lines: Vec<String> = read_cache(path)
        .into_iter()
        .filter(|l| !l.starts_with(&prefix))
        .collect();
    lines.push(format!("{prefix}{value}"));
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(format!(".tmp.{}", std::process::id()));
    let tmp = PathBuf::from(tmp);
    let result = (|| {
        if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir)?;
        }
        std::fs::write(&tmp, lines.join("\n") + "\n")?;
        std::fs::rename(&tmp, path)
    })();
    if let Err(e) = result {
        let _ = std::fs::remove_file(&tmp);
        eprintln!("cannot write tuning cache {}: {e}", path.display());
    }
}

/// Records `cfg` as the best configuration for the limb class of `bits`-bit numbers on this adapter
/// under `choice` (a forced choice has its own entry, see `cache_key_text`).
pub fn store(
    gpu: &Gpu,
    bits: u32,
    choice: SchemeChoice,
    cfg: &KernelConfig,
    squarings_per_sec: f64,
    path: &Path,
) {
    write_cache(
        path,
        &cache_key(gpu, bits, choice),
        &cache_value(cfg, squarings_per_sec),
    );
}

/// The environment values that decide where the tuning cache lives (injectable for tests).
#[derive(Clone, Debug, Default)]
pub struct CacheEnv {
    /// `BIGMONT_CACHE`: the cache file itself.
    pub bigmont_cache: Option<PathBuf>,
    /// `LOCALAPPDATA` (Windows).
    pub local_app_data: Option<PathBuf>,
    /// `XDG_CACHE_HOME`.
    pub xdg_cache_home: Option<PathBuf>,
    /// `HOME`.
    pub home: Option<PathBuf>,
}

impl CacheEnv {
    pub fn from_process() -> CacheEnv {
        let var = |name: &str| std::env::var_os(name).map(PathBuf::from);
        CacheEnv {
            bigmont_cache: var("BIGMONT_CACHE"),
            local_app_data: var("LOCALAPPDATA"),
            xdg_cache_home: var("XDG_CACHE_HOME"),
            home: var("HOME"),
        }
    }
}

/// The per-user default cache file, or `None` when no cache directory can be determined (the
/// engine then runs without a cache). `BIGMONT_CACHE` wins; otherwise Windows uses
/// `%LOCALAPPDATA%\bigmont\tune.cache` and other systems `$XDG_CACHE_HOME/bigmont/tune.cache` or
/// `$HOME/.cache/bigmont/tune.cache`. Empty variables count as unset.
pub fn default_cache_path(env: &CacheEnv, windows: bool) -> Option<PathBuf> {
    let set = |p: &Option<PathBuf>| p.clone().filter(|p| !p.as_os_str().is_empty());
    if let Some(p) = set(&env.bigmont_cache) {
        return Some(p);
    }
    let dir = if windows {
        set(&env.local_app_data)?
    } else {
        set(&env.xdg_cache_home).or_else(|| set(&env.home).map(|h| h.join(".cache")))?
    };
    Some(dir.join("bigmont").join("tune.cache"))
}

/// The cache file to use: none with `--no-cache`, else `--cache PATH`, else the default.
pub fn resolve_cache(
    explicit: Option<PathBuf>,
    disabled: bool,
    env: &CacheEnv,
    windows: bool,
) -> Option<PathBuf> {
    if disabled {
        None
    } else {
        explicit.or_else(|| default_cache_path(env, windows))
    }
}

/// Picks the fastest configuration that passes the self-test, reusing `cache` unless `retune`.
/// `Err` when none passes (the engine then falls back to the CPU).
///
/// Tuning is per limb class (`limb_class(bits)`), not per exact bit length: the self-test and the
/// timing run at the class's representative length `class_bits`, and the cache entry is keyed by the
/// class. The winner is a kernel shape (scheme, tpi, wg) at the representative length; apply it to an
/// exact bit length with `KernelConfig::retarget`. (A 13-bit kernel has the same limb count across the
/// class; a 32-bit one can have fewer limbs below the representative, which the self-test at the
/// representative covers.) This keeps tiny sizes testable (a 2-bit modulus has only the value 3) and
/// means two bit lengths in one class cost one tuning, not two.
///
/// `choice` limits the schemes tried: `Auto` times every scheme the adapter can run and the fastest
/// wins (the 13-bit scheme on an exact tie); a forced choice is cached under its own key.
///
/// A device-level failure (lost device, timed-out wait, uncaptured wgpu error: see
/// `Gpu::check_health`) stops tuning at once and returns that error, and nothing is written to the
/// cache: it says nothing about the configurations.
pub fn autotune(
    gpu: &Gpu,
    bits: u32,
    cache: Option<&Path>,
    retune: bool,
    choice: SchemeChoice,
) -> Result<Tuned, String> {
    gpu.check_health()?;
    let bits = class_bits(limb_class(bits));
    let cands = candidates(gpu, bits, choice);
    if cands.is_empty() {
        let why = no_candidates_reason(gpu, bits, choice);
        return Err(format!(
            "no usable kernel configuration on {} ({why})",
            gpu.describe()
        ));
    }
    let key = cache_key(gpu, bits, choice);
    if let (Some(path), false) = (cache, retune)
        && let Some((cfg, rate)) = cache_lookup(&read_cache(path), &key, &cands)
    {
        match self_test(gpu, cfg, bits) {
            Ok(()) => {
                return Ok(Tuned {
                    cfg,
                    squarings_per_sec: rate,
                    from_cache: true,
                });
            }
            Err(e) if gpu.check_health().is_err() => return Err(e),
            Err(e) => eprintln!("cached {cfg:?} failed its self-test ({e}); retuning"),
        }
    }
    let mut best: Option<Tuned> = None;
    let mut reasons = Vec::new();
    for cfg in cands {
        match self_test(gpu, cfg, bits)
            .and_then(|()| measure(gpu, cfg).map_err(|e| format!("{cfg:?}: {e}")))
        {
            Ok(rate) => {
                eprintln!("tune {cfg:?}: {rate:.0} squarings/s");
                if best.is_none_or(|b| rate > b.squarings_per_sec) {
                    best = Some(Tuned {
                        cfg,
                        squarings_per_sec: rate,
                        from_cache: false,
                    });
                }
            }
            Err(e) if gpu.check_health().is_err() => return Err(e),
            Err(e) => {
                eprintln!("skip {e}");
                reasons.push(e);
            }
        }
    }
    let best = best.ok_or_else(|| {
        format!(
            "no kernel configuration passed the self-test on {}: {}",
            gpu.describe(),
            reasons.join("; ")
        )
    })?;
    if let Some(path) = cache {
        store(gpu, bits, choice, &best.cfg, best.squarings_per_sec, path);
    }
    Ok(best)
}
