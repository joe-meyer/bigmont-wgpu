//! Command-line behaviour of the real `bigmont` binary: usage errors, exit codes and the tuning cache.

use bigmont_cli::formats;
use bigmont_core::gpu::Gpu;
use bigmont_oracle::{XorShift64, random_odd_modulus};
use num_bigint::BigUint;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

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

fn gpu_available() -> bool {
    match Gpu::new() {
        Ok(_) => true,
        Err(e) => {
            assert!(std::env::var("BIGMONT_ALLOW_SKIP").is_ok_and(|v| v == "1"), "no adapter: {e} (set BIGMONT_ALLOW_SKIP=1 to allow skips)");
            eprintln!("SKIP no adapter: {e}");
            false
        }
    }
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
    assert!(text.lines().any(|l| l.contains("|unrolled ")), "{text}");
    assert_eq!(files_in(&dir), ["tune.cache"]);
    let _ = std::fs::remove_dir_all(dir);
}
