//! `bigmont`: a drop-in for Beacon's `g1_fermat_v3` on candidate/verdict files.
//!
//!   bigmont CAND VERDICT [engine flags]
//!   bigmont canary CAND EXPECTED [engine flags]
//!   bigmont tune [--bits N] [--adapter S] [--vk-backend S] [--device N]

use bigmont_core::gpu::{AdapterSelector, Gpu};
use bigmont_cli::engine::{Backend, Engine, Options};
use bigmont_cli::{formats, tune};
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

/// Candidates per engine call: large enough to fill GPU dispatches, small enough for progress output.
const CHUNK: usize = 1 << 15;

const USAGE: &str = "usage: bigmont CAND VERDICT [--backend auto|gpu|cpu] [--adapter SUBSTR] [--vk-backend SUBSTR] [--device N]
                    [--retune] [--cache PATH] [--verify-rate N] [--repeat N] [--lock PATH]
       bigmont canary CAND EXPECTED [same engine flags]
       bigmont tune [--bits N] [--adapter SUBSTR] [--vk-backend SUBSTR] [--device N]";

/// Beacon flags this driver does not implement yet.
const UNSUPPORTED: [&str; 6] = ["--inject", "--ee", "--bins", "--chunk", "--growth", "--marker"];

struct Args {
    positional: Vec<String>,
    opts: Options,
    bits: u32,
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
            "--cache" => a.opts.cache = Some(PathBuf::from(value)),
            "--verify-rate" => a.opts.verify_rate = parse(flag, &value)?,
            "--bits" => a.bits = parse(flag, &value)?,
            _ => eprintln!("bigmont: note: {flag} is accepted and ignored"), // --repeat, --lock
        }
    }
    Ok(a)
}

/// Runs the engine over `numbers` in chunks and prints the summary. Returns the verdicts.
fn run_engine(opts: Options, numbers: &[num_bigint::BigUint]) -> Result<Vec<u8>, String> {
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
        "bigmont: {} | tests {tests} (gpu {}, cpu {}) | {secs:.2} s ({:.2} s tuning) | {:.1} tests/s excluding tuning | PRP {prps} | disagreements: prp {}, composite {} | rechecks {} | gpu errors {}",
        engine.describe(),
        r.gpu_tests,
        r.cpu_tests,
        r.tune_secs,
        tests as f64 / work_secs,
        r.prp_disagreements,
        r.composite_disagreements,
        r.rechecks,
        r.gpu_errors,
    );
    Ok(verdicts)
}

fn cmd_verdicts(a: Args) -> Result<ExitCode, String> {
    let [cand, verdict] = a.positional.as_slice() else { return Err(usage_err("expected CAND and VERDICT")) };
    let file = formats::read_candidates(cand.as_ref())?;
    let verdicts = run_engine(a.opts, &file.numbers)?;
    formats::write_verdicts(verdict.as_ref(), file.bits, &verdicts)?;
    Ok(ExitCode::SUCCESS)
}

fn cmd_canary(a: Args) -> Result<ExitCode, String> {
    let [cand, expected] = a.positional.as_slice() else { return Err(usage_err("expected CAND and EXPECTED")) };
    let file = formats::read_candidates(cand.as_ref())?;
    let (_, want) = formats::read_verdicts(expected.as_ref())?;
    if want.len() != file.numbers.len() {
        eprintln!("bigmont: canary FAILED: {} candidates but {} expected verdicts", file.numbers.len(), want.len());
        return Ok(ExitCode::from(1));
    }
    let got = run_engine(a.opts, &file.numbers)?;
    let bad: Vec<usize> = got.iter().zip(&want).enumerate().filter(|(_, (g, w))| g != w).map(|(i, _)| i).collect();
    if bad.is_empty() {
        eprintln!("bigmont: canary ok: all {} verdicts match", got.len());
        Ok(ExitCode::SUCCESS)
    } else {
        let first: Vec<String> = bad.iter().take(10).map(usize::to_string).collect();
        eprintln!("bigmont: canary FAILED: {} of {} verdicts differ; first mismatching indices: {}", bad.len(), got.len(), first.join(", "));
        Ok(ExitCode::from(1))
    }
}

fn cmd_tune(a: Args) -> Result<ExitCode, String> {
    if !a.positional.is_empty() {
        return Err(usage_err("tune takes no positional arguments"));
    }
    let gpu = Gpu::open(&a.opts.selector).map_err(|e| format!("bigmont: no GPU: {e}"))?;
    println!("adapter: {}", gpu.describe());
    let cands = tune::candidates(&gpu, a.bits);
    if cands.is_empty() {
        return Err(format!("bigmont: no kernel configuration fits {} for {} bits", gpu.describe(), a.bits));
    }
    println!("{} bits: {} candidate configurations", a.bits, cands.len());
    println!("{:>4} {:>5} {:>16}  {}", "tpi", "wg", "squarings/s", "status");
    let mut best: Option<(f64, String)> = None;
    for cfg in cands {
        match tune::self_test(&gpu, cfg, a.bits).and_then(|()| tune::measure(&gpu, cfg)) {
            Ok(rate) => {
                println!("{:>4} {:>5} {:>16.0}  ok", cfg.tpi, cfg.wg, rate);
                if best.as_ref().is_none_or(|(r, _)| rate > *r) {
                    best = Some((rate, format!("tpi {} wg {}", cfg.tpi, cfg.wg)));
                }
            }
            Err(e) => println!("{:>4} {:>5} {:>16}  FAILED: {e}", cfg.tpi, cfg.wg, "-"),
        }
    }
    match best {
        Some((rate, what)) => {
            println!("best: {what} at {rate:.0} squarings/s");
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
    if rest.iter().any(|a| a == "--help" || a == "-h") || (sub.is_empty() && rest.is_empty()) {
        println!("{USAGE}");
        return Ok(ExitCode::SUCCESS);
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
