use bigmont_cli::tune;
use bigmont_core::gpu::Gpu;
use bigmont_core::kernel::{KernelConfig, Variant};
use std::path::{Path, PathBuf};

fn allow_skip() -> bool {
    std::env::var("BIGMONT_ALLOW_SKIP").is_ok_and(|v| v == "1")
}

/// Strict-test skip decision: when `ok` is false the test must be skipped, which is only allowed
/// with BIGMONT_ALLOW_SKIP=1 (otherwise the test fails). Returns true when the caller should return.
fn skip_unless(ok: bool, why: &str) -> bool {
    if ok {
        return false;
    }
    assert!(allow_skip(), "{why} (set BIGMONT_ALLOW_SKIP=1 to allow)");
    eprintln!("SKIP {why}");
    true
}

fn gpu() -> Option<Gpu> {
    match Gpu::new() {
        Ok(g) => Some(g),
        Err(e) => {
            assert!(allow_skip(), "no adapter: {e} (set BIGMONT_ALLOW_SKIP=1 to allow skips)");
            eprintln!("SKIP no adapter: {e}");
            None
        }
    }
}

fn temp_cache(tag: &str) -> (PathBuf, PathBuf) {
    let dir = std::env::temp_dir().join(format!("bigmont-tune-test-{tag}-{}", std::process::id()));
    let cache = dir.join("tune.cache");
    (dir, cache)
}

/// Runs a real autotune to create a cache file and returns the cache key (everything before the
/// last two `|` fields of its only line).
fn seed_cache(gpu: &Gpu, cache: &Path) -> String {
    let first = tune::autotune(gpu, 1019, Some(cache), false).expect("autotune");
    assert!(!first.from_cache);
    let text = std::fs::read_to_string(cache).expect("cache written");
    let line = text.lines().find(|l| !l.is_empty()).expect("one cache line");
    line.rsplitn(3, '|').nth(2).expect("key|value|rate").to_string()
}

#[test]
fn candidates_are_valid_and_fit_the_adapter() {
    let Some(gpu) = gpu() else { return };
    let cands = tune::candidates(&gpu, 2029);
    if skip_unless(!cands.is_empty(), "no subgroup candidates on this adapter") {
        return;
    }
    for cfg in cands {
        assert!(cfg.validate().is_ok(), "{cfg:?}");
        assert!(cfg.limbs >= 157, "{cfg:?}");
        assert!(gpu.check_subgroups(cfg.tpi).is_ok());
        assert_eq!(cfg.variant, Variant::Unrolled);
    }
}

#[test]
fn self_test_accepts_a_good_config_at_two_sizes() {
    let Some(gpu) = gpu() else { return };
    for bits in [1019u32, 2029] {
        let cfg = tune::candidates(&gpu, bits).into_iter().find(|c| c.tpi == 8 && c.wg == 64);
        if skip_unless(cfg.is_some(), "no tpi 8 candidate on this adapter") {
            return;
        }
        let cfg = cfg.unwrap();
        tune::self_test(&gpu, cfg, bits).unwrap_or_else(|e| panic!("{cfg:?}: {e}"));
    }
}

#[test]
fn self_test_rejects_a_config_too_small_for_the_numbers() {
    let Some(gpu) = gpu() else { return };
    let cfg = KernelConfig::new(8, 64, Variant::Unrolled).with_limbs(80); // too small for 2029 bits
    let err = tune::self_test(&gpu, cfg, 2029).expect_err("must be rejected");
    assert!(err.contains("limbs"), "error should say why and name the config: {err}");
    assert!(err.contains("tpi: 8"), "error should name the config: {err}");
}

#[test]
fn autotune_picks_a_verified_config_and_uses_its_cache() {
    let Some(gpu) = gpu() else { return };
    let (dir, cache) = temp_cache("basic");
    if tune::candidates(&gpu, 1019).is_empty() {
        assert!(tune::autotune(&gpu, 1019, Some(&cache), false).is_err());
        skip_unless(false, "no subgroup candidates; autotune returned Err as designed");
        return;
    }
    let first = tune::autotune(&gpu, 1019, Some(&cache), false).expect("autotune");
    assert!(!first.from_cache);
    assert!(first.squarings_per_sec > 0.0);
    assert!(cache.exists());
    let second = tune::autotune(&gpu, 1019, Some(&cache), false).expect("autotune from cache");
    assert!(second.from_cache);
    assert_eq!(second.cfg, first.cfg);
    let third = tune::autotune(&gpu, 1019, Some(&cache), true).expect("retune");
    assert!(!third.from_cache);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn a_corrupt_cache_is_ignored_and_rewritten() {
    let Some(gpu) = gpu() else { return };
    if skip_unless(!tune::candidates(&gpu, 1019).is_empty(), "no subgroup candidates on this adapter") {
        return;
    }
    let (dir, cache) = temp_cache("corrupt");
    let key = seed_cache(&gpu, &cache);
    let garbage = [
        format!("{key}|unrolled 0 64|1e9"), // would divide by zero in KernelConfig::for_bits
        String::new(),
        "|".to_string(),
        "no separators at all".to_string(),
        format!("{key}|unrolled four 64|1e9"),
        format!("{key}|unrolled 4 64|nan"),
        format!("{key}|unrolled 4 64|-5"),
        format!("{key}|unrolled|"),
        format!("{key}|unrolled 4294967295 4294967295|1e9"),
    ];
    std::fs::write(&cache, garbage.join("\n") + "\n").unwrap();
    let tuned = tune::autotune(&gpu, 1019, Some(&cache), false).expect("retune after corrupt cache");
    assert!(!tuned.from_cache);
    // A valid entry was written back: the next call is served from the cache.
    let again = tune::autotune(&gpu, 1019, Some(&cache), false).expect("autotune from rewritten cache");
    assert!(again.from_cache);
    assert_eq!(again.cfg, tuned.cfg);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn a_cache_entry_naming_a_non_candidate_falls_back_to_a_retune() {
    let Some(gpu) = gpu() else { return };
    if skip_unless(!tune::candidates(&gpu, 1019).is_empty(), "no subgroup candidates on this adapter") {
        return;
    }
    let (dir, cache) = temp_cache("noncand");
    let key = seed_cache(&gpu, &cache);
    // tpi 8 with wg 100 is not among the candidates (wg is 64, 128 or 256).
    std::fs::write(&cache, format!("{key}|unrolled 8 100|1e9\n")).unwrap();
    let tuned = tune::autotune(&gpu, 1019, Some(&cache), false).expect("retune");
    assert!(!tuned.from_cache);
    assert_ne!(tuned.squarings_per_sec, 1e9);
    let _ = std::fs::remove_dir_all(dir);
}

// ---- default cache location (pure, no GPU) ----

use std::path::PathBuf as P;

fn env(bigmont: Option<&str>, local: Option<&str>, xdg: Option<&str>, home: Option<&str>) -> tune::CacheEnv {
    let p = |s: Option<&str>| s.map(P::from);
    tune::CacheEnv { bigmont_cache: p(bigmont), local_app_data: p(local), xdg_cache_home: p(xdg), home: p(home) }
}

#[test]
fn default_cache_path_on_windows_is_under_localappdata() {
    let e = env(None, Some(r"C:\Users\j\AppData\Local"), Some("/xdg"), Some("/home/j"));
    assert_eq!(tune::default_cache_path(&e, true), Some(P::from(r"C:\Users\j\AppData\Local").join("bigmont").join("tune.cache")));
    assert_eq!(tune::default_cache_path(&env(None, None, Some("/xdg"), Some("/home/j")), true), None, "no LOCALAPPDATA: run without a cache");
}

#[test]
fn default_cache_path_elsewhere_prefers_xdg_then_home() {
    let e = env(None, None, Some("/xdg"), Some("/home/j"));
    assert_eq!(tune::default_cache_path(&e, false), Some(P::from("/xdg").join("bigmont").join("tune.cache")));
    let e = env(None, None, None, Some("/home/j"));
    assert_eq!(tune::default_cache_path(&e, false), Some(P::from("/home/j").join(".cache").join("bigmont").join("tune.cache")));
    let e = env(None, None, Some(""), Some("/home/j"));
    assert_eq!(tune::default_cache_path(&e, false), Some(P::from("/home/j").join(".cache").join("bigmont").join("tune.cache")), "empty XDG_CACHE_HOME counts as unset");
    assert_eq!(tune::default_cache_path(&env(None, None, None, None), false), None, "no home directory: run without a cache");
    assert_eq!(tune::default_cache_path(&env(None, None, None, Some("")), false), None);
}

#[test]
fn bigmont_cache_overrides_the_default_on_every_platform() {
    for windows in [true, false] {
        let e = env(Some("/custom/tune.cache"), Some("/local"), Some("/xdg"), Some("/home/j"));
        assert_eq!(tune::default_cache_path(&e, windows), Some(P::from("/custom/tune.cache")));
        // Even with no home at all.
        assert_eq!(tune::default_cache_path(&env(Some("/custom/tune.cache"), None, None, None), windows), Some(P::from("/custom/tune.cache")));
    }
}

#[test]
fn resolve_cache_applies_the_flags_over_the_default() {
    let e = env(Some("/from-env"), None, None, None);
    assert_eq!(tune::resolve_cache(Some(P::from("/flag")), false, &e, false), Some(P::from("/flag")), "--cache beats BIGMONT_CACHE");
    assert_eq!(tune::resolve_cache(None, false, &e, false), Some(P::from("/from-env")));
    assert_eq!(tune::resolve_cache(None, true, &e, false), None, "--no-cache disables the default");
}

#[test]
fn write_cache_is_atomic_creates_parents_and_replaces_its_own_key() {
    let dir = std::env::temp_dir().join(format!("bigmont-write-cache-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let cache = dir.join("a").join("b").join("tune.cache"); // parents do not exist
    tune::write_cache(&cache, "k1", "v1");
    tune::write_cache(&cache, "k2", "v2");
    tune::write_cache(&cache, "k1", "v3");
    let text = std::fs::read_to_string(&cache).unwrap();
    assert_eq!(text.lines().collect::<Vec<_>>(), ["k2|v2", "k1|v3"]);
    let names: Vec<_> = std::fs::read_dir(cache.parent().unwrap()).unwrap().map(|e| e.unwrap().file_name().to_string_lossy().into_owned()).collect();
    assert_eq!(names, ["tune.cache"], "no temporary file is left behind");
    let _ = std::fs::remove_dir_all(dir);
}
