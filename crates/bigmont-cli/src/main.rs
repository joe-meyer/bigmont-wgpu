//! `bigmont`: tests candidate files and writes verdict files, a drop-in for an existing CUDA-based Fermat driver.
//!
//!   bigmont CAND VERDICT [engine flags] [--gec [--inject IDX BDRY]]
//!   bigmont CAND|LIST VERDICT --ee SIZES [--bins] [--chunk N] [--growth F] [--marker N] [engine flags]
//!   bigmont canary CAND EXPECTED [engine flags]
//!   bigmont tune [--bits N] [--limbs auto|13|32] [--adapter S] [--vk-backend S] [--device N] [--cache PATH | --no-cache]
//!
//! Early exit (`--ee SIZES`): the candidates are grouped into "k"s by a `G2S1` sizes file and tested in
//! waves; a k that yields a probable prime is retired and its remaining candidates are skipped (their
//! verdict byte is `--marker`, 3 to 255, default 3). With `--bins`, the first argument is a text file listing
//! candidate files (one path per line) that are concatenated. `--lock FILE` serialises the GPU phase
//! between processes (an exclusive lock taken after the inputs are read, released before the output is
//! written; in plain mode it surrounds the engine run).
//!
//! Error check: `--gec` runs the GPU buckets with the Gerbicz check; a candidate that fails it gets verdict
//! byte 2 instead of 0 or 1. `--inject IDX BDRY` (needs `--gec`, plain runs on the GPU only) flips a bit in
//! candidate IDX (0-based, in the candidate file) during Gerbicz block BDRY, to drill the alarm path. BDRY
//! must be at most (bits - 1) / 40, the candidate's number of full 40-iteration blocks (equal to it targets
//! the tail recomputation); a larger BDRY, or a candidate the CPU answers, injects nothing.
//!
//! Limb scheme: `--limbs auto` (the default) measures every scheme the adapter can run and uses the
//! fastest; `--limbs 13` or `--limbs 32` forces one (32 needs 64-bit integers in shaders). A forced
//! scheme keeps its own tuning-cache entries. Cache entries written before the limb schemes existed
//! (13-bit only) are not reused: the first run re-measures and writes the new entries.
//!
//! Tuning cache: `--cache PATH`, else `$BIGMONT_CACHE`, else a per-user file (`%LOCALAPPDATA%\bigmont\tune.cache`
//! on Windows, `$XDG_CACHE_HOME/bigmont/tune.cache` or `~/.cache/bigmont/tune.cache` elsewhere); `--no-cache`
//! tunes every run and writes nothing. With no usable cache location the engine runs without a cache.

use bigmont_cli::canary;
use bigmont_cli::early_exit::{run_waves, schedule};
use bigmont_cli::engine::{Backend, Engine, EngineReport, Options};
use bigmont_cli::{formats, tune};
use bigmont_core::gpu::{AdapterSelector, Gpu};
use bigmont_core::kernel::KernelConfig;
use std::fs::{File, OpenOptions, TryLockError};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

/// Candidates per engine call: large enough to fill GPU dispatches, small enough for progress output.
const CHUNK: usize = 1 << 15;

const USAGE: &str = "usage: bigmont CAND VERDICT [--backend auto|gpu|cpu] [--adapter SUBSTR] [--vk-backend SUBSTR] [--device N]
                    [--limbs auto|13|32] [--retune] [--cache PATH | --no-cache] [--verify-rate N] [--repeat N] [--lock FILE]
                    [--gec [--inject IDX BDRY]]
                    [--ee SIZES [--bins] [--chunk N] [--growth F] [--marker N]]
       bigmont canary CAND EXPECTED [same engine flags]
       bigmont tune [--bits N] [--limbs auto|13|32] [--adapter SUBSTR] [--vk-backend SUBSTR] [--device N] [--cache PATH | --no-cache]
error check: --gec runs the GPU buckets with the Gerbicz error check (off by default): a candidate whose computation
              fails the check gets verdict byte 2 (neither composite nor probable prime; retest it) and is counted
              as a GEC alarm in the summary. Buckets answered by the CPU (fewer candidates than the GPU threshold,
              or after a GPU error) are not checked. --inject IDX BDRY (with --gec, plain GPU runs only) flips a
              bit in candidate IDX (0-based) during Gerbicz block BDRY, so that candidate must come out as 2.
              BDRY is at most (bits-1)/40, the number of full 40-iteration blocks (equal to it targets the tail
              recomputation); a larger BDRY injects nothing, nor does a candidate the CPU answers.
early exit:  --ee SIZES (a G2S1 file) tests the candidates in waves and skips the rest of a k once one candidate is a probable
              prime; skipped verdict bytes are --marker (3-255, default 3). --bins reads CAND as a text file listing
              candidate files (one per line), tested as one concatenated array; it needs --ee. The first wave covers
              --chunk positions per k (default 6), each later wave --growth times more (default 1.15; 1.0 = constant;
              a --chunk of 0 or less steps one position at a time whatever the growth).
lock:        --lock FILE takes an exclusive lock (created if missing) before any GPU work and holds it until the last
              wave (early-exit mode) or the end of the engine run; other bigmont processes using FILE wait their turn.
              The gpu-phase time of the g1_fermat[EE] line includes engine setup and any first-run tuning.
              Input and usage errors exit with status 2; a failure while testing (the engine gave up) exits with 3.
limb scheme: --limbs auto (default) measures every scheme the adapter can run and uses the fastest; 13 or 32 forces one
              (tuning caches from before the limb schemes are re-measured once)
tuning cache: --cache PATH, else $BIGMONT_CACHE, else a per-user file (%LOCALAPPDATA%\\bigmont\\tune.cache on Windows,
              $XDG_CACHE_HOME/bigmont/tune.cache or ~/.cache/bigmont/tune.cache elsewhere); --no-cache disables it";

/// A schedule with more boundaries than this is refused before it is built (`--chunk 1 --growth 1.0`
/// on a k of billions of candidates would otherwise allocate gigabytes).
const MAX_BOUNDARIES: u64 = 1_000_000;

struct Args {
    positional: Vec<String>,
    opts: Options,
    bits: u32,
    /// `--cache PATH` / `--no-cache`, resolved against the environment after parsing.
    cache_flag: Option<PathBuf>,
    no_cache: bool,
    /// `--ee SIZES`: early-exit mode.
    ee: Option<PathBuf>,
    /// `--bins`: CAND is a text file listing candidate files.
    bins: bool,
    chunk: i32,
    growth: f64,
    marker: u8,
    /// `--lock FILE`.
    lock: Option<PathBuf>,
    /// `--inject IDX BDRY`: the fault drill (candidate index in the file, Gerbicz block).
    inject: Option<(usize, u32)>,
}

fn usage_err(msg: impl std::fmt::Display) -> String {
    format!("bigmont: {msg}\n{USAGE}")
}

fn parse<T: std::str::FromStr>(flag: &str, v: &str) -> Result<T, String> {
    v.parse()
        .map_err(|_| usage_err(format!("invalid value {v:?} for {flag}")))
}

fn parse_args(raw: &[String]) -> Result<Args, String> {
    let mut a = Args {
        positional: Vec::new(),
        opts: Options {
            selector: AdapterSelector::from_env(),
            ..Options::default()
        },
        bits: 2029,
        cache_flag: None,
        no_cache: false,
        ee: None,
        bins: false,
        chunk: 6,
        growth: 1.15,
        marker: 3,
        lock: None,
        inject: None,
    };
    let mut it = raw.iter();
    while let Some(arg) = it.next() {
        let (flag, inline) = match arg.split_once('=') {
            Some((f, v)) if f.starts_with("--") => (f, Some(v.to_string())),
            _ => (arg.as_str(), None),
        };
        if !flag.starts_with("--") {
            a.positional.push(arg.clone());
            continue;
        }
        if flag == "--gec" {
            if inline.is_some() {
                return Err(usage_err("--gec takes no value"));
            }
            a.opts.gec = true;
            continue;
        }
        if flag == "--inject" {
            if inline.is_some() {
                return Err(usage_err("--inject needs two values: --inject IDX BDRY"));
            }
            let (Some(idx), Some(block)) = (it.next(), it.next()) else {
                return Err(usage_err("--inject needs two values: --inject IDX BDRY"));
            };
            a.inject = Some((parse(flag, idx)?, parse(flag, block)?));
            continue;
        }
        if flag == "--retune" {
            a.opts.retune = true;
            continue;
        }
        if flag == "--no-cache" {
            a.no_cache = true;
            continue;
        }
        if flag == "--bins" {
            if inline.is_some() {
                return Err(usage_err("--bins takes no value"));
            }
            a.bins = true;
            continue;
        }
        if !matches!(
            flag,
            "--backend"
                | "--adapter"
                | "--vk-backend"
                | "--device"
                | "--cache"
                | "--verify-rate"
                | "--repeat"
                | "--lock"
                | "--bits"
                | "--limbs"
                | "--ee"
                | "--chunk"
                | "--growth"
                | "--marker"
        ) {
            return Err(usage_err(format!("unknown flag {flag}")));
        }
        let value = match inline {
            Some(v) => v,
            None => it
                .next()
                .cloned()
                .ok_or_else(|| usage_err(format!("{flag} needs a value")))?,
        };
        match flag {
            "--backend" => {
                a.opts.backend = match value.as_str() {
                    "auto" => Backend::Auto,
                    "gpu" => Backend::Gpu,
                    "cpu" => Backend::Cpu,
                    _ => {
                        return Err(usage_err(format!(
                            "--backend must be auto, gpu or cpu, not {value:?}"
                        )));
                    }
                }
            }
            "--adapter" => a.opts.selector.name = Some(value),
            "--vk-backend" => a.opts.selector.backend = Some(value),
            "--device" => a.opts.selector.index = Some(parse(flag, &value)?),
            "--cache" => a.cache_flag = Some(PathBuf::from(value)),
            "--verify-rate" => a.opts.verify_rate = parse(flag, &value)?,
            "--bits" => a.bits = parse(flag, &value)?,
            "--limbs" => {
                a.opts.scheme = tune::SchemeChoice::parse(&value)
                    .map_err(|e| usage_err(format!("--limbs: {e}")))?
            }
            "--ee" => a.ee = Some(PathBuf::from(value)),
            "--lock" => a.lock = Some(PathBuf::from(value)),
            "--chunk" => a.chunk = parse(flag, &value)?,
            "--growth" => {
                a.growth = parse(flag, &value)?;
                if !a.growth.is_finite() {
                    return Err(usage_err(format!(
                        "--growth must be a finite number, not {value:?}"
                    )));
                }
            }
            "--marker" => {
                a.marker = value.parse().map_err(|_| {
                    usage_err(format!(
                        "--marker must be a whole number from 3 to 255, not {value:?}"
                    ))
                })?;
                // 0, 1 and 2 are verdict bytes (composite, probable prime, Gerbicz alarm): a skipped
                // position carrying one would read as a real verdict. The CUDA driver refuses them too.
                if a.marker < 3 {
                    return Err(usage_err(format!(
                        "--marker must be at least 3 (0, 1 and 2 are verdict bytes), not {value:?}"
                    )));
                }
            }
            _ => eprintln!("bigmont: note: {flag} is accepted and ignored"), // --repeat
        }
    }
    if a.no_cache && a.cache_flag.is_some() {
        return Err(usage_err("--cache and --no-cache are mutually exclusive"));
    }
    if a.inject.is_some() {
        if !a.opts.gec {
            return Err(usage_err("--inject needs --gec"));
        }
        if a.ee.is_some() || a.bins {
            return Err(usage_err("--inject applies to plain runs only (not --ee)"));
        }
        if a.opts.backend == Backend::Cpu {
            return Err(usage_err(
                "--inject needs the GPU backend (not --backend cpu)",
            ));
        }
        // The drill runs a handful of candidates: below the default GPU threshold they would be
        // answered by the CPU, which is not checked and injects nothing.
        a.opts.min_gpu_batch = 1;
    }
    a.opts.cache = tune::resolve_cache(
        a.cache_flag.take(),
        a.no_cache,
        &tune::CacheEnv::from_process(),
        cfg!(windows),
    );
    Ok(a)
}

/// What a run produced: the verdicts, the engine's counters and whether a GPU was in use.
struct RunOutput {
    verdicts: Vec<u8>,
    report: EngineReport,
    gpu_active: bool,
}

/// Exit status for a failure while testing (the engine could not produce verdicts), as opposed to a
/// problem with the command line or the input files (2). The CUDA driver likewise ends a run that
/// cannot obtain a verdict with its own status (3), apart from its input errors (2).
const EXIT_RUN_FAILED: u8 = 3;

/// Why a test run did not produce verdicts.
#[derive(Debug, PartialEq, Eq)]
enum RunError {
    /// The request cannot be served as given (for example `--backend gpu` without a usable GPU).
    Config(String),
    /// The engine failed while testing (device failure it could not recover from, a malformed reply).
    Failed(String),
}

impl RunError {
    /// The process exit status for this error.
    fn status(&self) -> u8 {
        match self {
            RunError::Config(_) => 2,
            RunError::Failed(_) => EXIT_RUN_FAILED,
        }
    }

    /// Prints the message and returns the exit status.
    fn report(self) -> ExitCode {
        let status = self.status();
        match self {
            RunError::Config(m) => eprintln!("{m}"),
            RunError::Failed(m) => eprintln!("bigmont: run failed: {m}"),
        }
        ExitCode::from(status)
    }
}

/// Runs the engine over `numbers` in chunks and prints the summary.
///
/// `inject` is the `--inject IDX BDRY` drill: the engine is told to fault candidate `IDX` (an index
/// into `numbers`) in the chunk that holds it.
fn run_engine(
    opts: Options,
    numbers: &[num_bigint::BigUint],
    inject: Option<(usize, u32)>,
) -> Result<RunOutput, RunError> {
    let mut engine = Engine::open(opts);
    eprintln!("bigmont: {}", engine.describe());
    // An empty input never reaches `verdicts`, so a demanded-but-missing GPU is reported here.
    engine.require_backend().map_err(RunError::Config)?;
    let t = Instant::now();
    let mut verdicts = Vec::with_capacity(numbers.len());
    for (k, chunk) in numbers.chunks(CHUNK).enumerate() {
        let start = k * CHUNK;
        if let Some((idx, block)) = inject
            && (start..start + chunk.len()).contains(&idx)
        {
            engine.inject_fault(idx - start, block);
        }
        verdicts.extend(engine.verdicts(chunk).map_err(RunError::Failed)?);
        if numbers.len() > CHUNK {
            eprintln!("bigmont: {}/{} candidates", verdicts.len(), numbers.len());
        }
    }
    let secs = t.elapsed().as_secs_f64();
    let prps = verdicts.iter().filter(|&&v| v == 1).count() as u64;
    let report = print_engine_summary(&engine, secs, prps);
    Ok(RunOutput {
        verdicts,
        report,
        gpu_active: engine.gpu_active(),
    })
}

/// Prints the engine's one-line summary of a run that took `secs` and found `prps` probable primes.
fn print_engine_summary(engine: &Engine, secs: f64, prps: u64) -> EngineReport {
    let r = engine.report();
    let tests = r.gpu_tests + r.cpu_tests;
    let work_secs = (secs - r.tune_secs).max(1e-9);
    eprintln!(
        "bigmont: {} | tests {tests} (gpu {}, cpu {}) | {secs:.2} s ({:.2} s tuning) | {:.1} tests/s excluding tuning | PRP {prps} | disagreements: prp {}, composite {} | rechecks {} | GEC alarms {} | gpu errors {} | device losses {} (recovered {}, poll timeouts {})",
        engine.describe(),
        r.gpu_tests,
        r.cpu_tests,
        r.tune_secs,
        tests as f64 / work_secs,
        r.prp_disagreements,
        r.composite_disagreements,
        r.rechecks,
        r.gec_alarms,
        r.gpu_errors,
        r.device_losses,
        r.device_recoveries,
        r.poll_timeouts,
    );
    r
}

/// Takes the exclusive `--lock` file lock (creating the file), waiting for other holders. Returns the
/// open file (the lock lasts until it is dropped) and the seconds spent waiting; `(None, 0.0)` without
/// `--lock`.
fn take_lock(path: Option<&Path>) -> Result<(Option<File>, f64), String> {
    let Some(path) = path else {
        return Ok((None, 0.0));
    };
    let t = Instant::now();
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)
        .map_err(|e| format!("bigmont: cannot open lock file {}: {e}", path.display()))?;
    match file.try_lock() {
        Ok(()) => {}
        Err(TryLockError::WouldBlock) => {
            eprintln!("bigmont: waiting for the lock on {}", path.display());
            file.lock()
                .map_err(|e| format!("bigmont: cannot lock {}: {e}", path.display()))?;
        }
        Err(TryLockError::Error(e)) => {
            return Err(format!("bigmont: cannot lock {}: {e}", path.display()));
        }
    }
    Ok((Some(file), t.elapsed().as_secs_f64()))
}

/// Whether the schedule for `chunk`, `growth` and a largest k of `s_max` would have more than `limit`
/// boundaries (`early_exit::schedule(..).len() > limit`). Walks the same steps as that function but only
/// counts, and stops at the limit, so it never allocates.
fn schedule_too_long(chunk: i32, growth: f64, s_max: u32, limit: u64) -> bool {
    let s = i64::from(s_max);
    let (mut at, mut c, mut n) = (0i64, f64::from(chunk), 1u64); // the schedule starts with boundary 0
    if n > limit {
        return true;
    }
    while at < s {
        n += 1;
        if n > limit {
            return true;
        }
        at = at
            .saturating_add((c.round_ties_even() as i64).max(1))
            .min(s);
        if growth > 1.0 {
            c *= growth;
        }
    }
    false
}

fn cmd_verdicts(a: Args) -> Result<ExitCode, String> {
    if let Some(sizes) = a.ee.clone() {
        return cmd_early_exit(a, &sizes);
    }
    if a.bins {
        return Err(usage_err("--bins requires --ee SIZES"));
    }
    let [cand, verdict] = a.positional.as_slice() else {
        return Err(usage_err("expected CAND and VERDICT"));
    };
    let file = formats::read_candidates(cand.as_ref())?;
    if let Some((idx, _)) = a.inject
        && idx >= file.numbers.len()
    {
        return Err(usage_err(format!(
            "--inject index {idx} is out of range: {cand} has {} candidates",
            file.numbers.len()
        )));
    }
    let (lock, _) = take_lock(a.lock.as_deref())?;
    let run = match run_engine(a.opts, &file.numbers, a.inject) {
        Ok(run) => run,
        Err(e) => return Ok(e.report()),
    };
    drop(lock);
    formats::write_verdicts(verdict.as_ref(), file.bits, &run.verdicts)?;
    Ok(ExitCode::SUCCESS)
}

/// `--ee SIZES`: the candidates (one file, or the files listed by `--bins`) are tested in waves and a
/// k is retired at its first probable prime. Every input problem is reported before the lock is taken
/// or the engine opened, and no verdict file is written on error.
fn cmd_early_exit(a: Args, sizes_path: &Path) -> Result<ExitCode, String> {
    let [input, verdict] = a.positional.as_slice() else {
        return Err(usage_err("expected CAND and VERDICT"));
    };
    let inputs = if a.bins {
        formats::read_bin_list(input.as_ref())?
    } else {
        vec![PathBuf::from(input)]
    };
    let sizes = formats::read_sizes(sizes_path)?;
    // Judged from the sizes alone, before the schedule or the candidates are allocated.
    let s_max = sizes.iter().copied().max().unwrap_or(0);
    if schedule_too_long(a.chunk, a.growth, s_max, MAX_BOUNDARIES) {
        return Err(format!(
            "bigmont: the largest k has {s_max} candidates, which would take more than {MAX_BOUNDARIES} waves with --chunk {} --growth {}; raise --chunk or --growth",
            a.chunk, a.growth
        ));
    }
    let file = formats::read_candidate_files(&inputs)?;
    let sum: u64 = sizes.iter().map(|&s| u64::from(s)).sum();
    if sum != file.numbers.len() as u64 {
        return Err(format!(
            "bigmont: sizes sum {sum} != count {}",
            file.numbers.len()
        ));
    }
    let bounds = schedule(a.chunk, a.growth, s_max);

    let (lock, lock_wait) = take_lock(a.lock.as_deref())?;
    let t = Instant::now();
    let mut engine = Engine::open(a.opts);
    eprintln!("bigmont: {}", engine.describe());
    // No candidates (or all ks empty) means `run_waves` never calls the engine; still honour `--backend gpu`.
    if let Err(m) = engine.require_backend() {
        return Ok(RunError::Config(m).report());
    }
    let out = match run_waves(&mut engine, &file.numbers, &sizes, &bounds, a.marker) {
        Ok(out) => out,
        Err(m) => return Ok(RunError::Failed(m).report()),
    };
    let secs = t.elapsed().as_secs_f64();
    print_engine_summary(&engine, secs, out.prp);
    // The GPU phase is over: free the engine's GPU memory, then let the next process start its waves.
    drop(engine);
    drop(lock);
    formats::write_verdicts(verdict.as_ref(), file.bits, &out.verdicts)?;
    let count = file.numbers.len() as u64;
    let fraction = if count == 0 {
        0.0
    } else {
        out.tested as f64 / count as f64
    };
    eprintln!(
        "g1_fermat[EE]: bits={} count={count} ks={} tested={} ({fraction:.3}) waves={} | {secs:.3}s gpu-phase (lock wait {lock_wait:.2}s) | {:.1} tests/s | prp={} gec_alarm={}",
        file.bits,
        sizes.len(),
        out.tested,
        out.waves,
        out.tested as f64 / secs.max(1e-9),
        out.prp,
        out.alarms,
    );
    Ok(ExitCode::SUCCESS)
}

/// The early-exit flags and the fault drill only mean something to the verdict command.
fn reject_ee_flags(a: &Args, sub: &str) -> Result<(), String> {
    if a.ee.is_some() || a.bins {
        return Err(usage_err(format!(
            "--ee and --bins are not valid for `bigmont {sub}`"
        )));
    }
    if a.inject.is_some() {
        return Err(usage_err(format!(
            "--inject is not valid for `bigmont {sub}`"
        )));
    }
    Ok(())
}

fn cmd_canary(a: Args) -> Result<ExitCode, String> {
    reject_ee_flags(&a, "canary")?;
    let [cand, expected] = a.positional.as_slice() else {
        return Err(usage_err("expected CAND and EXPECTED"));
    };
    let file = formats::read_candidates(cand.as_ref())?;
    let (_, want) = formats::read_verdicts(expected.as_ref())?;
    if let Some(msg) = canary::length_failure(file.numbers.len(), want.len()) {
        eprintln!("bigmont: canary FAILED: {msg}");
        return Ok(ExitCode::from(1));
    }
    let (lock, _) = take_lock(a.lock.as_deref())?;
    let run = match run_engine(a.opts, &file.numbers, None) {
        Ok(run) => run,
        Err(e) => return Ok(e.report()),
    };
    drop(lock);
    let failed = canary::failures(&run.verdicts, &want, &run.report, run.gpu_active);
    if failed.is_empty() {
        eprintln!(
            "bigmont: canary ok: all {} verdicts match",
            run.verdicts.len()
        );
        Ok(ExitCode::SUCCESS)
    } else {
        for f in &failed {
            eprintln!("bigmont: canary FAILED: {f}");
        }
        Ok(ExitCode::from(1))
    }
}

/// The `--limbs` spelling of `choice`.
fn choice_name(choice: tune::SchemeChoice) -> &'static str {
    match choice {
        tune::SchemeChoice::Auto => "auto",
        tune::SchemeChoice::Only(s) => s.name(),
    }
}

fn cmd_tune(a: Args) -> Result<ExitCode, String> {
    reject_ee_flags(&a, "tune")?;
    if !a.positional.is_empty() {
        return Err(usage_err("tune takes no positional arguments"));
    }
    let (_lock, _) = take_lock(a.lock.as_deref())?; // held until the tuning run ends
    let gpu = Gpu::open(&a.opts.selector).map_err(|e| format!("bigmont: no GPU: {e}"))?;
    println!("adapter: {}", gpu.describe());
    // Tuning is per limb class: test and time at the class's representative length.
    let (class, bits) = (
        tune::limb_class(a.bits),
        tune::class_bits(tune::limb_class(a.bits)),
    );
    let choice = a.opts.scheme;
    let cands = tune::candidates(&gpu, bits, choice);
    if cands.is_empty() {
        return Err(format!(
            "bigmont: no kernel configuration fits {} for {} bits with --limbs {}",
            gpu.describe(),
            a.bits,
            choice_name(choice)
        ));
    }
    println!(
        "{} bits: limb class {class}, tuned at {bits} bits, --limbs {}, {} candidate configurations",
        a.bits,
        choice_name(choice),
        cands.len()
    );
    println!(
        "{:>6} {:>4} {:>5} {:>16}  status",
        "scheme", "tpi", "wg", "squarings/s"
    );
    let mut best: Option<(f64, KernelConfig)> = None;
    for cfg in cands {
        let scheme = cfg.scheme.name();
        match tune::self_test(&gpu, cfg, bits).and_then(|()| tune::measure(&gpu, cfg)) {
            Ok(rate) => {
                println!(
                    "{scheme:>6} {:>4} {:>5} {:>16.0}  ok",
                    cfg.tpi, cfg.wg, rate
                );
                if best.as_ref().is_none_or(|(r, _)| rate > *r) {
                    best = Some((rate, cfg));
                }
            }
            Err(e) => {
                println!(
                    "{scheme:>6} {:>4} {:>5} {:>16}  FAILED: {e}",
                    cfg.tpi, cfg.wg, "-"
                );
                if let Err(problem) = gpu.check_health() {
                    return Err(format!(
                        "bigmont: the GPU failed during tuning ({problem}); nothing cached, rerun `bigmont tune`"
                    ));
                }
            }
        }
    }
    match best {
        Some((rate, cfg)) => {
            println!(
                "best: scheme {} tpi {} wg {} at {rate:.0} squarings/s",
                cfg.scheme.name(),
                cfg.tpi,
                cfg.wg
            );
            match &a.opts.cache {
                Some(path) => {
                    tune::store(&gpu, bits, choice, &cfg, rate, path);
                    println!("cached in {}", path.display());
                }
                None => println!("not cached (--no-cache, or no cache location)"),
            }
            Ok(ExitCode::SUCCESS)
        }
        None => Err("bigmont: no configuration passed the self-test".into()),
    }
}

fn real_main() -> Result<ExitCode, String> {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let (sub, rest) = match raw.first().map(String::as_str) {
        Some(s @ ("canary" | "tune")) => (s, &raw[1..]),
        _ => ("", &raw[..]),
    };
    if rest.iter().any(|a| a == "--help" || a == "-h") {
        println!("{USAGE}");
        return Ok(ExitCode::SUCCESS);
    }
    if raw.is_empty() {
        return Err(usage_err("expected CAND and VERDICT (or a subcommand)"));
    }
    let args = parse_args(rest)?;
    match sub {
        "canary" => cmd_canary(args),
        "tune" => cmd_tune(args),
        _ => cmd_verdicts(args),
    }
}

fn main() -> ExitCode {
    match real_main() {
        Ok(code) => code,
        Err(msg) => {
            eprintln!("{msg}");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> Result<Args, String> {
        let raw: Vec<String> = list.iter().map(|s| s.to_string()).collect();
        parse_args(&raw)
    }

    #[test]
    fn gec_and_inject_parse_and_inject_takes_exactly_two_values() {
        let a = args(&["c", "v", "--gec", "--inject", "1", "3", "--no-cache"]).unwrap();
        assert!(a.opts.gec);
        assert_eq!(a.inject, Some((1, 3)));
        assert_eq!(
            a.positional,
            ["c", "v"],
            "the two values are not positionals"
        );
        assert_eq!(
            a.opts.min_gpu_batch, 1,
            "the drill must not be answered by the CPU"
        );
        let plain = args(&["c", "v", "--no-cache"]).unwrap();
        assert!(!plain.opts.gec && plain.inject.is_none());
        assert_eq!(plain.opts.min_gpu_batch, Options::default().min_gpu_batch);
        // `--gec` alone does not touch the batch threshold.
        let g = args(&["c", "v", "--gec", "--no-cache"]).unwrap();
        assert!(g.opts.gec);
        assert_eq!(g.opts.min_gpu_batch, Options::default().min_gpu_batch);
        for bad in [
            &["--gec", "--inject=1"][..],
            &["--gec", "--inject", "1"],
            &["--gec", "--inject"],
            &["--gec", "--inject", "x", "3"],
            &["--gec", "--inject", "1", "-3"],
            &["--gec=1"],
        ] {
            assert!(args(bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn inject_needs_gec_a_plain_run_and_the_gpu() {
        let err = |extra: &[&str]| {
            let mut v = vec!["c", "v", "--inject", "0", "0"];
            v.extend(extra);
            args(&v).err().unwrap_or_default()
        };
        assert!(err(&[]).contains("--inject needs --gec"));
        assert!(err(&["--gec", "--ee", "s"]).contains("plain runs only"));
        assert!(err(&["--gec", "--bins"]).contains("plain runs only"));
        assert!(err(&["--gec", "--backend", "cpu"]).contains("GPU"));
        assert!(args(&["c", "v", "--inject", "0", "0", "--gec", "--backend", "gpu"]).is_ok());
    }

    /// Input and configuration problems exit 2, a failure while testing exits 3, so a caller can tell a
    /// bad request from a device that gave up.
    #[test]
    fn run_errors_map_to_distinct_exit_statuses() {
        assert_eq!(RunError::Config("x".into()).status(), 2);
        assert_eq!(RunError::Failed("x".into()).status(), 3);
    }

    /// The length guard must agree with the real schedule, on both sides of the limit.
    #[test]
    fn schedule_too_long_agrees_with_the_schedule_it_guards() {
        let (mut over, mut within) = (0u32, 0u32);
        for chunk in [-3, 0, 1, 2, 6, 100] {
            for growth in [0.5, 1.0, 1.15, 2.0] {
                for s_max in [0u32, 1, 2, 7, 100, 5_000, 250_000] {
                    let len = schedule(chunk, growth, s_max).len() as u64;
                    for limit in (1..=60).chain([len - 1, len, len + 1, MAX_BOUNDARIES]) {
                        let want = len > limit;
                        assert_eq!(
                            schedule_too_long(chunk, growth, s_max, limit),
                            want,
                            "chunk {chunk} growth {growth} s_max {s_max} limit {limit} ({len} boundaries)"
                        );
                        if want {
                            over += 1;
                        } else {
                            within += 1;
                        }
                    }
                }
            }
        }
        assert!(
            over > 500 && within > 500,
            "both outcomes are exercised ({over} over, {within} within)"
        );
        // The real limit: 999_999 steps of 1 make exactly 1_000_000 boundaries (allowed), one more is refused.
        assert!(!schedule_too_long(1, 1.0, 999_999, MAX_BOUNDARIES));
        assert!(schedule_too_long(1, 1.0, 1_000_000, MAX_BOUNDARIES));
        assert!(schedule_too_long(1, 1.0, u32::MAX, MAX_BOUNDARIES));
        assert!(
            !schedule_too_long(1, 1.15, u32::MAX, MAX_BOUNDARIES),
            "a growing schedule over a huge k is short"
        );
    }
}
