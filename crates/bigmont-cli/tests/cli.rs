//! Command-line behaviour of the real `bigmont` binary: usage errors, exit codes and the tuning cache.

use bigmont_cli::formats;
use bigmont_core::gpu::Gpu;
use bigmont_oracle::{XorShift64, random_odd_modulus};
use num_bigint::BigUint;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

mod common;

fn bigmont(args: &[&str]) -> Command {
    let mut c = Command::new(env!("CARGO_BIN_EXE_bigmont"));
    c.args(args);
    c
}

fn run(mut c: Command) -> Output {
    c.output().expect("run bigmont")
}

fn code(out: &Output) -> Option<i32> {
    out.status.code()
}

#[test]
fn no_arguments_is_a_usage_error_with_exit_code_2() {
    let out = run(bigmont(&[]));
    assert_eq!(code(&out), Some(2));
    assert!(String::from_utf8_lossy(&out.stderr).contains("usage:"), "usage goes to stderr");
}

#[test]
fn help_exits_0_and_prints_usage() {
    for flag in ["--help", "-h"] {
        let out = run(bigmont(&[flag]));
        assert_eq!(code(&out), Some(0), "{flag}");
        assert!(String::from_utf8_lossy(&out.stdout).contains("usage:"), "{flag}");
    }
}

#[test]
fn unknown_flags_and_missing_positionals_exit_2() {
    assert_eq!(code(&run(bigmont(&["--bogus"]))), Some(2));
    assert_eq!(code(&run(bigmont(&["only-one-file"]))), Some(2));
    assert_eq!(code(&run(bigmont(&["canary", "only-one-file"]))), Some(2));
}

#[test]
fn cache_and_no_cache_together_are_a_usage_error() {
    assert_eq!(code(&run(bigmont(&["a", "b", "--cache", "x", "--no-cache"]))), Some(2));
}

// ---- tuning cache: default location, BIGMONT_CACHE, --no-cache ----

fn tmp_dir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("bigmont-cli-test-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// The engine's GPU backend needs an adapter with subgroups; see `common::usable_gpu`.
fn gpu_available() -> bool {
    common::usable_gpu()
}

/// 80 odd 1019-bit candidates (above the engine's 64-candidate GPU threshold), written to `dir`.
fn candidates(dir: &Path) -> PathBuf {
    let mut rng = XorShift64::new(5);
    let ns: Vec<BigUint> = (0..80).map(|_| random_odd_modulus(&mut rng, 1019)).collect();
    let p = dir.join("cand.bin");
    formats::write_candidates(&p, 1024, &ns).unwrap();
    p
}

fn files_in(dir: &Path) -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(dir).unwrap().map(|e| e.unwrap().file_name().to_string_lossy().into_owned()).collect();
    v.sort();
    v
}

#[test]
fn bigmont_cache_env_names_the_default_cache_and_no_cache_disables_it() {
    if !gpu_available() {
        return;
    }
    let dir = tmp_dir("cache");
    let cand = candidates(&dir);
    let cache_dir = dir.join("nested").join("cache-dir"); // parent does not exist yet
    let cache = cache_dir.join("tune.cache");

    let mut c = bigmont(&[cand.to_str().unwrap(), dir.join("v1.bin").to_str().unwrap(), "--backend", "gpu", "--no-cache"]);
    c.env("BIGMONT_CACHE", &cache);
    let out = run(c);
    assert_eq!(code(&out), Some(0), "{}", String::from_utf8_lossy(&out.stderr));
    assert!(!cache.exists() && !cache_dir.exists(), "--no-cache must not create the cache");

    let mut c = bigmont(&[cand.to_str().unwrap(), dir.join("v2.bin").to_str().unwrap(), "--backend", "gpu"]);
    c.env("BIGMONT_CACHE", &cache);
    let out = run(c);
    assert_eq!(code(&out), Some(0), "{}", String::from_utf8_lossy(&out.stderr));
    assert!(cache.exists(), "the default (here BIGMONT_CACHE) cache is written");
    assert_eq!(files_in(&cache_dir), ["tune.cache"], "an atomic write leaves no temporary file");
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn the_tune_subcommand_writes_the_cache() {
    if !gpu_available() {
        return;
    }
    let dir = tmp_dir("tune");
    let cache = dir.join("tune.cache");
    let out = run(bigmont(&["tune", "--bits", "1019", "--cache", cache.to_str().unwrap()]));
    assert_eq!(code(&out), Some(0), "{}", String::from_utf8_lossy(&out.stderr));
    let text = std::fs::read_to_string(&cache).expect("tune writes the cache");
    // Auto may pick either scheme: the line must parse as a winner of one of them (the value is the
    // last two `|` fields, `<unrolled|unrolled32> <tpi> <wg>|<rate>`), under the Auto key
    // (`name|backend|driver|any|class`).
    let line = text.lines().find(|l| !l.is_empty()).expect("one cache line");
    let mut fields = line.rsplitn(3, '|');
    let (rate, shape, key) = (fields.next().unwrap(), fields.next().unwrap(), fields.next().unwrap());
    assert!(bigmont_cli::tune::parse_value(&format!("{shape}|{rate}")).is_some(), "{text}");
    assert!(key.contains("|any|") && !key.contains("only-"), "tune with the default --limbs uses the Auto (`any`) key: {text}");
    assert_eq!(files_in(&dir), ["tune.cache"]);
    let _ = std::fs::remove_dir_all(dir);
}

// ---- --limbs auto|13|32 ----

#[test]
fn an_invalid_limbs_value_is_a_usage_error_naming_the_allowed_values() {
    for bad in ["7", "64", "", "thirty-two"] {
        let out = run(bigmont(&["a", "b", "--limbs", bad]));
        assert_eq!(code(&out), Some(2), "--limbs {bad:?}");
        let err = String::from_utf8_lossy(&out.stderr);
        assert!(err.contains("auto") && err.contains("13") && err.contains("32"), "--limbs {bad:?}: {err}");
    }
    let out = run(bigmont(&["tune", "--limbs=7"]));
    assert_eq!(code(&out), Some(2));
    assert!(String::from_utf8_lossy(&out.stderr).contains("--limbs"));
    let out = run(bigmont(&["--help"]));
    assert!(String::from_utf8_lossy(&out.stdout).contains("--limbs"), "the usage text documents the flag");
}

fn has_int64() -> Option<bool> {
    Gpu::new().ok().map(|g| g.has_int64)
}

/// Whether a forced-32 test can run here; skips loudly (an error unless BIGMONT_ALLOW_SKIP=1) when not.
fn int64_available() -> bool {
    match has_int64() {
        Some(true) => true,
        other => {
            assert!(std::env::var("BIGMONT_ALLOW_SKIP").is_ok_and(|v| v == "1"), "no int64 adapter (found: {other:?}); set BIGMONT_ALLOW_SKIP=1 to allow skips");
            eprintln!("SKIP no adapter with SHADER_INT64");
            false
        }
    }
}

#[test]
fn limbs_32_runs_a_candidate_file_and_writes_the_expected_verdicts() {
    if !int64_available() {
        return;
    }
    let dir = tmp_dir("limbs32");
    let cand = candidates(&dir);
    let verdict = dir.join("v.bin");
    let cache = dir.join("tune.cache");
    let out = run(bigmont(&[cand.to_str().unwrap(), verdict.to_str().unwrap(), "--backend", "gpu", "--limbs", "32", "--cache", cache.to_str().unwrap()]));
    assert_eq!(code(&out), Some(0), "{}", String::from_utf8_lossy(&out.stderr));
    let file = formats::read_candidates(&cand).unwrap();
    let (_, got) = formats::read_verdicts(&verdict).unwrap();
    let want: Vec<u8> = file.numbers.iter().map(|n| u8::from(bigmont_cpu::fermat_base2(n))).collect();
    assert_eq!(got, want);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("Limb32"), "the engine's tune lines name the scheme: {stderr}");
    assert!(stderr.contains("gpu 80"), "all 80 candidates ran on the GPU: {stderr}");
    let text = std::fs::read_to_string(&cache).unwrap();
    assert!(text.lines().all(|l| l.contains("|only-32|") && l.contains("|unrolled32 ")), "{text}");
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn the_tune_subcommand_forces_a_scheme_and_prints_a_scheme_column() {
    if !int64_available() {
        return;
    }
    let dir = tmp_dir("tune32");
    let cache = dir.join("tune.cache");
    let out = run(bigmont(&["tune", "--bits", "1019", "--limbs", "32", "--cache", cache.to_str().unwrap()]));
    assert_eq!(code(&out), Some(0), "{}", String::from_utf8_lossy(&out.stderr));
    let stdout = String::from_utf8_lossy(&out.stdout);
    let header = stdout.lines().find(|l| l.contains("squarings/s") && l.contains("tpi")).expect("table header");
    assert!(header.contains("scheme"), "{header}");
    assert!(stdout.lines().filter(|l| l.trim_end().ends_with("ok")).all(|l| l.split_whitespace().next() == Some("32")), "only 32-bit rows: {stdout}");
    assert!(stdout.contains("best: scheme 32"), "{stdout}");
    let text = std::fs::read_to_string(&cache).expect("tune writes the cache");
    assert!(text.lines().all(|l| l.contains("|only-32|") && l.contains("|unrolled32 ")), "{text}");
    let _ = std::fs::remove_dir_all(dir);
}
