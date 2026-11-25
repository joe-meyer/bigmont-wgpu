//! `bigmont`: a drop-in for Beacon's `g1_fermat_v3` on candidate/verdict files.
//!
//!   bigmont CAND VERDICT [engine flags]
//!   bigmont canary CAND EXPECTED [engine flags]
//!   bigmont tune [--bits N] [--adapter S] [--vk-backend S] [--device N] [--cache PATH | --no-cache]
//!
//! Tuning cache: `--cache PATH`, else `$BIGMONT_CACHE`, else a per-user file (`%LOCALAPPDATA%\bigmont\tune.cache`
//! on Windows, `$XDG_CACHE_HOME/bigmont/tune.cache` or `~/.cache/bigmont/tune.cache` elsewhere); `--no-cache`
//! tunes every run and writes nothing. With no usable cache location the engine runs without a cache.

use bigmont_cli::canary;
use bigmont_cli::engine::{Backend, Engine, EngineReport, Options};
use bigmont_cli::{formats, tune};
use bigmont_core::gpu::{AdapterSelector, Gpu};
use bigmont_core::kernel::KernelConfig;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

/// Candidates per engine call: large enough to fill GPU dispatches, small enough for progress output.
const CHUNK: usize = 1 << 15;

const USAGE: &str = "usage: bigmont CAND VERDICT [--backend auto|gpu|cpu] [--adapter SUBSTR] [--vk-backend SUBSTR] [--device N]
                    [--retune] [--cache PATH | --no-cache] [--verify-rate N] [--repeat N] [--lock PATH]
       bigmont canary CAND EXPECTED [same engine flags]
       bigmont tune [--bits N] [--adapter SUBSTR] [--vk-backend SUBSTR] [--device N] [--cache PATH | --no-cache]
tuning cache: --cache PATH, else $BIGMONT_CACHE, else a per-user file (%LOCALAPPDATA%\\bigmont\\tune.cache on Windows,
              $XDG_CACHE_HOME/bigmont/tune.cache or ~/.cache/bigmont/tune.cache elsewhere); --no-cache disables it";

/// Beacon flags this driver does not implement yet.
const UNSUPPORTED: [&str; 6] = ["--inject", "--ee", "--bins", "--chunk", "--growth", "--marker"];

struct Args {
    positional: Vec<String>,
    opts: Options,
    bits: u32,
    /// `--cache PATH` / `--no-cache`, resolved against the environment after parsing.
    cache_flag: Option<PathBuf>,
    no_cache: bool,
}

fn usage_err(msg: impl std::fmt::Display) -> String {
    format!("bigmont: {msg}\n{USAGE}")
}

fn parse<T: std::str::FromStr>(flag: &str, v: &str) -> Result<T, String> {
    v.parse().map_err(|_| usage_err(format!("invalid value {v:?} for {flag}")))
}

fn parse_args(raw: &[String]) -> Result<Args, String> {
    let mut a = Args {
        positional: Vec::new(),
        opts: Options { selector: AdapterSelector::from_env(), ..Options::default() },
        bits: 2029,
        cache_flag: None,
        no_cache: false,
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
        if UNSUPPORTED.contains(&flag) {
            return Err(format!("bigmont: {flag} is not supported yet"));
        }
        if flag == "--retune" {
            a.opts.retune = true;
            continue;
        }
        if flag == "--no-cache" {
            a.no_cache = true;
            continue;
        }
        if !matches!(flag, "--backend" | "--adapter" | "--vk-backend" | "--device" | "--cache" | "--verify-rate" | "--repeat" | "--lock" | "--bits") {
            return Err(usage_err(format!("unknown flag {flag}")));
        }
        let value = match inline {
            Some(v) => v,
            None => it.next().cloned().ok_or_else(|| usage_err(format!("{flag} needs a value")))?,
        };
        match flag {
            "--backend" => {
                a.opts.backend = match value.as_str() {
                    "auto" => Backend::Auto,
                    "gpu" => Backend::Gpu,
                    "cpu" => Backend::Cpu,
                    _ => return Err(usage_err(format!("--backend must be auto, gpu or cpu, not {value:?}"))),
                }
            }
            "--adapter" => a.opts.selector.name = Some(value),
            "--vk-backend" => a.opts.selector.backend = Some(value),
            "--device" => a.opts.selector.index = Some(parse(flag, &value)?),
            "--cache" => a.cache_flag = Some(PathBuf::from(value)),
            "--verify-rate" => a.opts.verify_rate = parse(flag, &value)?,
            "--bits" => a.bits = parse(flag, &value)?,
            _ => eprintln!("bigmont: note: {flag} is accepted and ignored"), // --repeat, --lock
        }
    }
    if a.no_cache && a.cache_flag.is_some() {
        return Err(usage_err("--cache and --no-cache are mutually exclusive"));
    }
    a.opts.cache = tune::resolve_cache(a.cache_flag.take(), a.no_cache, &tune::CacheEnv::from_process(), cfg!(windows));
    Ok(a)
}

/// What a run produced: the verdicts, the engine's counters and whether a GPU was in use.
struct RunOutput {
    verdicts: Vec<u8>,
    report: EngineReport,
    gpu_active: bool,
}

/// Runs the engine over `numbers` in chunks and prints the summary.
fn run_engine(opts: Options, numbers: &[num_bigint::BigUint]) -> Result<RunOutput, String> {
    let mut engine = Engine::open(opts);
    eprintln!("bigmont: {}", engine.describe());
    let t = Instant::now();
    let mut verdicts = Vec::with_capacity(numbers.len());
    for chunk in numbers.chunks(CHUNK) {
        verdicts.extend(engine.verdicts(chunk)?);
        if numbers.len() > CHUNK {
            eprintln!("bigmont: {}/{} candidates", verdicts.len(), numbers.len());
        }
    }
    let secs = t.elapsed().as_secs_f64();
    let r = engine.report();
    let tests = r.gpu_tests + r.cpu_tests;
    let work_secs = (secs - r.tune_secs).max(1e-9);
    let prps = verdicts.iter().filter(|&&v| v == 1).count();
    eprintln!(
        "bigmont: {} | tests {tests} (gpu {}, cpu {}) | {secs:.2} s ({:.2} s tuning) | {:.1} tests/s excluding tuning | PRP {prps} | disagreements: prp {}, composite {} | rechecks {} | gpu errors {} | device losses {} (recovered {}, poll timeouts {})",
        engine.describe(),
        r.gpu_tests,
        r.cpu_tests,
        r.tune_secs,
        tests as f64 / work_secs,
        r.prp_disagreements,
        r.composite_disagreements,
        r.rechecks,
        r.gpu_errors,
        r.device_losses,
        r.device_recoveries,
        r.poll_timeouts,
    );
    Ok(RunOutput { verdicts, report: r, gpu_active: engine.gpu_active() })
}

fn cmd_verdicts(a: Args) -> Result<ExitCode, String> {
    let [cand, verdict] = a.positional.as_slice() else { return Err(usage_err("expected CAND and VERDICT")) };
    let file = formats::read_candidates(cand.as_ref())?;
    let run = run_engine(a.opts, &file.numbers)?;
    formats::write_verdicts(verdict.as_ref(), file.bits, &run.verdicts)?;
    Ok(ExitCode::SUCCESS)
}

fn cmd_canary(a: Args) -> Result<ExitCode, String> {
    let [cand, expected] = a.positional.as_slice() else { return Err(usage_err("expected CAND and EXPECTED")) };
    let file = formats::read_candidates(cand.as_ref())?;
    let (_, want) = formats::read_verdicts(expected.as_ref())?;
    if let Some(msg) = canary::length_failure(file.numbers.len(), want.len()) {
        eprintln!("bigmont: canary FAILED: {msg}");
        return Ok(ExitCode::from(1));
    }
    let run = run_engine(a.opts, &file.numbers)?;
    let failed = canary::failures(&run.verdicts, &want, &run.report, run.gpu_active);
    if failed.is_empty() {
        eprintln!("bigmont: canary ok: all {} verdicts match", run.verdicts.len());
        Ok(ExitCode::SUCCESS)
    } else {
        for f in &failed {
            eprintln!("bigmont: canary FAILED: {f}");
        }
        Ok(ExitCode::from(1))
    }
}

fn cmd_tune(a: Args) -> Result<ExitCode, String> {
    if !a.positional.is_empty() {
        return Err(usage_err("tune takes no positional arguments"));
    }
    let gpu = Gpu::open(&a.opts.selector).map_err(|e| format!("bigmont: no GPU: {e}"))?;
    println!("adapter: {}", gpu.describe());
    // Tuning is per limb class: test and time at the class's representative length.
    let (class, bits) = (tune::limb_class(a.bits), tune::class_bits(tune::limb_class(a.bits)));
    let cands = tune::candidates(&gpu, bits);
    if cands.is_empty() {
        return Err(format!("bigmont: no kernel configuration fits {} for {} bits", gpu.describe(), a.bits));
    }
    println!("{} bits: limb class {class}, tuned at {bits} bits, {} candidate configurations", a.bits, cands.len());
    println!("{:>4} {:>5} {:>16}  {}", "tpi", "wg", "squarings/s", "status");
    let mut best: Option<(f64, KernelConfig)> = None;
    for cfg in cands {
        match tune::self_test(&gpu, cfg, bits).and_then(|()| tune::measure(&gpu, cfg)) {
            Ok(rate) => {
                println!("{:>4} {:>5} {:>16.0}  ok", cfg.tpi, cfg.wg, rate);
                if best.as_ref().is_none_or(|(r, _)| rate > *r) {
                    best = Some((rate, cfg));
                }
            }
            Err(e) => {
                println!("{:>4} {:>5} {:>16}  FAILED: {e}", cfg.tpi, cfg.wg, "-");
                if let Err(problem) = gpu.check_health() {
                    return Err(format!("bigmont: the GPU failed during tuning ({problem}); nothing cached, rerun `bigmont tune`"));
                }
            }
        }
    }
    match best {
        Some((rate, cfg)) => {
            println!("best: tpi {} wg {} at {rate:.0} squarings/s", cfg.tpi, cfg.wg);
            match &a.opts.cache {
                Some(path) => {
                    tune::store(&gpu, bits, &cfg, rate, path);
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
