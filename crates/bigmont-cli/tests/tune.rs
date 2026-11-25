use bigmont_cli::tune::{
    self,
    SchemeChoice::{self, Auto, Only},
};
use bigmont_core::gpu::Gpu;
use bigmont_core::kernel::{KernelConfig, Variant};
use bigmont_core::scheme::LimbScheme::{Limb13, Limb32};
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
            assert!(
                allow_skip(),
                "no adapter: {e} (set BIGMONT_ALLOW_SKIP=1 to allow skips)"
            );
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
    let first = tune::autotune(gpu, 1019, Some(cache), false, Auto).expect("autotune");
    assert!(!first.from_cache);
    let text = std::fs::read_to_string(cache).expect("cache written");
    let line = text
        .lines()
        .find(|l| !l.is_empty())
        .expect("one cache line");
    line.rsplitn(3, '|')
        .nth(2)
        .expect("key|value|rate")
        .to_string()
}

#[test]
fn candidates_are_valid_and_fit_the_adapter() {
    let Some(gpu) = gpu() else { return };
    let cands = tune::candidates(&gpu, 2029, Auto);
    if skip_unless(!cands.is_empty(), "no subgroup candidates on this adapter") {
        return;
    }
    for cfg in cands {
        assert!(cfg.validate().is_ok(), "{cfg:?}");
        assert!(cfg.r_bits() >= 2029 + 4, "R must exceed 16n: {cfg:?}"); // 157 13-bit limbs, or 64 32-bit ones
        assert!(gpu.check_subgroups(cfg.tpi).is_ok());
        assert_eq!(cfg.variant, Variant::Unrolled);
    }
}

#[test]
fn self_test_accepts_a_good_config_at_two_sizes() {
    let Some(gpu) = gpu() else { return };
    for bits in [1019u32, 2029] {
        let cfg = tune::candidates(&gpu, bits, Auto)
            .into_iter()
            .find(|c| c.tpi == 8 && c.wg == 64);
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
    assert!(
        err.contains("limbs"),
        "error should say why and name the config: {err}"
    );
    assert!(
        err.contains("tpi: 8"),
        "error should name the config: {err}"
    );
}

#[test]
fn autotune_picks_a_verified_config_and_uses_its_cache() {
    let Some(gpu) = gpu() else { return };
    let (dir, cache) = temp_cache("basic");
    if tune::candidates(&gpu, 1019, Auto).is_empty() {
        assert!(tune::autotune(&gpu, 1019, Some(&cache), false, Auto).is_err());
        skip_unless(
            false,
            "no subgroup candidates; autotune returned Err as designed",
        );
        return;
    }
    let first = tune::autotune(&gpu, 1019, Some(&cache), false, Auto).expect("autotune");
    assert!(!first.from_cache);
    assert!(first.squarings_per_sec > 0.0);
    assert!(cache.exists());
    let second =
        tune::autotune(&gpu, 1019, Some(&cache), false, Auto).expect("autotune from cache");
    assert!(second.from_cache);
    assert_eq!(second.cfg, first.cfg);
    let third = tune::autotune(&gpu, 1019, Some(&cache), true, Auto).expect("retune");
    assert!(!third.from_cache);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn a_corrupt_cache_is_ignored_and_rewritten() {
    let Some(gpu) = gpu() else { return };
    if skip_unless(
        !tune::candidates(&gpu, 1019, Auto).is_empty(),
        "no subgroup candidates on this adapter",
    ) {
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
    let tuned =
        tune::autotune(&gpu, 1019, Some(&cache), false, Auto).expect("retune after corrupt cache");
    assert!(!tuned.from_cache);
    // A valid entry was written back: the next call is served from the cache.
    let again = tune::autotune(&gpu, 1019, Some(&cache), false, Auto)
        .expect("autotune from rewritten cache");
    assert!(again.from_cache);
    assert_eq!(again.cfg, tuned.cfg);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn a_cache_entry_naming_a_non_candidate_falls_back_to_a_retune() {
    let Some(gpu) = gpu() else { return };
    if skip_unless(
        !tune::candidates(&gpu, 1019, Auto).is_empty(),
        "no subgroup candidates on this adapter",
    ) {
        return;
    }
    let (dir, cache) = temp_cache("noncand");
    let key = seed_cache(&gpu, &cache);
    // tpi 8 with wg 100 is not among the candidates (wg is 64, 128 or 256).
    std::fs::write(&cache, format!("{key}|unrolled 8 100|1e9\n")).unwrap();
    let tuned = tune::autotune(&gpu, 1019, Some(&cache), false, Auto).expect("retune");
    assert!(!tuned.from_cache);
    assert_ne!(tuned.squarings_per_sec, 1e9);
    let _ = std::fs::remove_dir_all(dir);
}

// ---- default cache location (pure, no GPU) ----

use std::path::PathBuf as P;

fn env(
    bigmont: Option<&str>,
    local: Option<&str>,
    xdg: Option<&str>,
    home: Option<&str>,
) -> tune::CacheEnv {
    let p = |s: Option<&str>| s.map(P::from);
    tune::CacheEnv {
        bigmont_cache: p(bigmont),
        local_app_data: p(local),
        xdg_cache_home: p(xdg),
        home: p(home),
    }
}

#[test]
fn default_cache_path_on_windows_is_under_localappdata() {
    let e = env(
        None,
        Some(r"C:\Users\j\AppData\Local"),
        Some("/xdg"),
        Some("/home/j"),
    );
    assert_eq!(
        tune::default_cache_path(&e, true),
        Some(
            P::from(r"C:\Users\j\AppData\Local")
                .join("bigmont")
                .join("tune.cache")
        )
    );
    assert_eq!(
        tune::default_cache_path(&env(None, None, Some("/xdg"), Some("/home/j")), true),
        None,
        "no LOCALAPPDATA: run without a cache"
    );
}

#[test]
fn default_cache_path_elsewhere_prefers_xdg_then_home() {
    let e = env(None, None, Some("/xdg"), Some("/home/j"));
    assert_eq!(
        tune::default_cache_path(&e, false),
        Some(P::from("/xdg").join("bigmont").join("tune.cache"))
    );
    let e = env(None, None, None, Some("/home/j"));
    assert_eq!(
        tune::default_cache_path(&e, false),
        Some(
            P::from("/home/j")
                .join(".cache")
                .join("bigmont")
                .join("tune.cache")
        )
    );
    let e = env(None, None, Some(""), Some("/home/j"));
    assert_eq!(
        tune::default_cache_path(&e, false),
        Some(
            P::from("/home/j")
                .join(".cache")
                .join("bigmont")
                .join("tune.cache")
        ),
        "empty XDG_CACHE_HOME counts as unset"
    );
    assert_eq!(
        tune::default_cache_path(&env(None, None, None, None), false),
        None,
        "no home directory: run without a cache"
    );
    assert_eq!(
        tune::default_cache_path(&env(None, None, None, Some("")), false),
        None
    );
}

#[test]
fn bigmont_cache_overrides_the_default_on_every_platform() {
    for windows in [true, false] {
        let e = env(
            Some("/custom/tune.cache"),
            Some("/local"),
            Some("/xdg"),
            Some("/home/j"),
        );
        assert_eq!(
            tune::default_cache_path(&e, windows),
            Some(P::from("/custom/tune.cache"))
        );
        // Even with no home at all.
        assert_eq!(
            tune::default_cache_path(&env(Some("/custom/tune.cache"), None, None, None), windows),
            Some(P::from("/custom/tune.cache"))
        );
    }
}

#[test]
fn resolve_cache_applies_the_flags_over_the_default() {
    let e = env(Some("/from-env"), None, None, None);
    assert_eq!(
        tune::resolve_cache(Some(P::from("/flag")), false, &e, false),
        Some(P::from("/flag")),
        "--cache beats BIGMONT_CACHE"
    );
    assert_eq!(
        tune::resolve_cache(None, false, &e, false),
        Some(P::from("/from-env"))
    );
    assert_eq!(
        tune::resolve_cache(None, true, &e, false),
        None,
        "--no-cache disables the default"
    );
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
    let names: Vec<_> = std::fs::read_dir(cache.parent().unwrap())
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    assert_eq!(names, ["tune.cache"], "no temporary file is left behind");
    let _ = std::fs::remove_dir_all(dir);
}

// ---- limb classes: tuning is shared by every bit length with the same kernel shape ----

#[test]
fn limb_class_is_the_minimum_limb_count_floored_at_the_smallest_kernel() {
    use bigmont_oracle::limbs_for_bits;
    for bits in 1..=3000u32 {
        let class = tune::limb_class(bits);
        assert!(
            class >= 4,
            "{bits} bits: class {class} is below the smallest kernel (4 limbs)"
        );
        assert_eq!(class, (limbs_for_bits(bits) as u32).max(4), "{bits} bits");
        let rep = tune::class_bits(class);
        assert!(
            rep >= bits,
            "{bits} bits: the class representative ({rep} bits) must cover the class"
        );
        assert!(
            rep >= 32,
            "{bits} bits: the representative needs room for both PRPs and composites ({rep} bits)"
        );
        assert_eq!(
            tune::limb_class(rep),
            class,
            "{bits} bits: the representative must stay in its own class"
        );
        assert_eq!(
            rep,
            13 * class - 4,
            "the representative is the largest length the class supports"
        );
    }
    // 2 to 48 bits share the smallest class; 49 bits needs a fifth limb.
    for bits in 2..=48u32 {
        assert_eq!(tune::limb_class(bits), 4, "{bits} bits");
    }
    assert_eq!(tune::limb_class(49), 5);
    assert_eq!(tune::limb_class(2029), 157);
    assert_eq!(tune::class_bits(4), 48);
}

#[test]
fn every_bit_length_of_a_class_builds_the_same_kernel_as_its_representative() {
    // The tuned (tpi, wg) is chosen at the representative length; the Runner then uses it at the exact
    // length. That is sound only if both give the same limb count for every cooperative width.
    for bits in 2..=3000u32 {
        let rep = tune::class_bits(tune::limb_class(bits));
        for tpi in [4u32, 8, 16, 32] {
            let at = KernelConfig::for_bits(bits, tpi, 64, Variant::Unrolled);
            let at_rep = KernelConfig::for_bits(rep, tpi, 64, Variant::Unrolled);
            assert_eq!(at, at_rep, "{bits} bits, tpi {tpi}");
        }
    }
}

#[test]
fn self_test_refuses_lengths_too_small_for_a_meaningful_test_instead_of_hanging() {
    let Some(gpu) = gpu() else { return };
    let cfg = KernelConfig::for_bits(48, 8, 64, Variant::Unrolled);
    if skip_unless(gpu.check_subgroups(8).is_ok(), "no tpi 8 on this adapter") {
        return;
    }
    for bits in [2u32, 9, 31] {
        let err = tune::self_test(&gpu, cfg, bits).expect_err("must be refused");
        assert!(err.contains("at least 32"), "{bits} bits: {err}");
    }
}

#[test]
fn autotune_shares_one_cache_entry_across_a_limb_class_including_two_bit_numbers() {
    let Some(gpu) = gpu() else { return };
    if skip_unless(
        !tune::candidates(&gpu, 20, Auto).is_empty(),
        "no subgroup candidates on this adapter",
    ) {
        return;
    }
    let (dir, cache) = temp_cache("class");
    // 2 bits: only the number 3 exists, so a self-test at 2 bits would be impossible; the class
    // representative (48 bits) is used.
    let first = tune::autotune(&gpu, 2, Some(&cache), false, Auto).expect("autotune at 2 bits");
    assert!(!first.from_cache);
    // Whichever scheme Auto picked, the shape at 2 bits is the same kernel (4 limbs at tpi 4, ...).
    let at = |scheme, bits| {
        KernelConfig::for_bits_scheme(scheme, bits, first.cfg.tpi, first.cfg.wg, Variant::Unrolled)
    };
    assert_eq!(first.cfg, at(first.cfg.scheme, 2));
    for bits in [20u32, 40, 48] {
        let again = tune::autotune(&gpu, bits, Some(&cache), false, Auto)
            .expect("autotune from the class entry");
        assert!(
            again.from_cache,
            "{bits} bits shares the 2-bit class and must be served from the cache"
        );
        assert_eq!(
            again.cfg, first.cfg,
            "the class entry gives the same shape and scheme"
        );
        assert_eq!(again.cfg.retarget(bits), at(first.cfg.scheme, bits));
    }
    let lines = std::fs::read_to_string(&cache)
        .unwrap()
        .lines()
        .filter(|l| !l.is_empty())
        .count();
    assert_eq!(lines, 1, "one cache line for the whole class");
    // 49 bits is the next class: not served from that entry.
    let next = tune::autotune(&gpu, 49, Some(&cache), false, Auto).expect("autotune at 49 bits");
    assert!(!next.from_cache);
    let _ = std::fs::remove_dir_all(dir);
}

// ---- the limb scheme dimension: choice, cache text and keys (pure, no GPU) ----

#[test]
fn scheme_choice_parses_auto_13_and_32_and_rejects_everything_else() {
    assert_eq!(SchemeChoice::parse("auto"), Ok(Auto));
    assert_eq!(SchemeChoice::parse("13"), Ok(Only(Limb13)));
    assert_eq!(SchemeChoice::parse("32"), Ok(Only(Limb32)));
    for bad in ["64", "", "Auto", "13 ", "0x20", "-1"] {
        let err = SchemeChoice::parse(bad).expect_err(bad);
        assert!(
            err.contains("auto") && err.contains("13") && err.contains("32"),
            "the error names the allowed values: {err}"
        );
    }
    assert_eq!(SchemeChoice::default(), Auto);
}

#[test]
fn scheme_choice_allows_what_it_names() {
    for s in [Limb13, Limb32] {
        assert!(Auto.allows(s));
        assert!(Only(s).allows(s));
    }
    assert!(!Only(Limb13).allows(Limb32));
    assert!(!Only(Limb32).allows(Limb13));
}

#[test]
fn cache_values_round_trip_for_both_schemes_and_old_lines_stay_valid() {
    for scheme in [Limb13, Limb32] {
        let cfg = KernelConfig::for_bits_scheme(scheme, 1019, 8, 64, Variant::Unrolled);
        let v = tune::cache_value(&cfg, 123.5);
        assert_eq!(tune::parse_value(&v), Some((scheme, 8, 64, 123.5)), "{v}");
    }
    // 13-bit winners keep the exact text older versions wrote and read.
    let c13 = KernelConfig::for_bits_scheme(Limb13, 1019, 16, 128, Variant::Unrolled);
    assert_eq!(tune::cache_value(&c13, 123.5), "unrolled 16 128|123.5");
    let c32 = KernelConfig::for_bits_scheme(Limb32, 1019, 16, 128, Variant::Unrolled);
    assert_eq!(tune::cache_value(&c32, 9.5), "unrolled32 16 128|9.5");

    assert_eq!(
        tune::parse_value("unrolled 8 64|123.5"),
        Some((Limb13, 8, 64, 123.5))
    );
    assert_eq!(
        tune::parse_value("unrolled32 8 64|9.5"),
        Some((Limb32, 8, 64, 9.5))
    );
    for bad in [
        "unrolled64 8 64|9.5",
        "rolled 8 64|9.5",
        "unrolled32 8|9.5",
        "unrolled32 8 64|nan",
        "unrolled32 8 64|-1",
        "unrolled32 x 64|1",
        "",
        "|",
        "unrolled 8 64",
    ] {
        assert_eq!(tune::parse_value(bad), None, "{bad:?} must be a miss");
    }
}

#[test]
fn a_forced_scheme_has_its_own_cache_key_and_never_touches_the_auto_entry() {
    use bigmont_cli::tune::cache_key_text;
    let class = tune::limb_class(1019);
    let auto = cache_key_text("Some GPU", "Vulkan", "drv", "info", 1019, Auto);
    let f13 = cache_key_text("Some GPU", "Vulkan", "drv", "info", 1019, Only(Limb13));
    let f32 = cache_key_text("Some GPU", "Vulkan", "drv", "info", 1019, Only(Limb32));
    // Auto has its own `any` choice field; a cache from before the limb schemes used no field at all
    // (`name|backend|driver driver_info|class`) and must never match.
    let legacy = format!("Some GPU|Vulkan|drv info|{class}");
    assert_eq!(auto, format!("Some GPU|Vulkan|drv info|any|{class}"));
    assert_eq!(f13, format!("Some GPU|Vulkan|drv info|only-13|{class}"));
    assert_eq!(f32, format!("Some GPU|Vulkan|drv info|only-32|{class}"));
    let keys = [&legacy, &auto, &f13, &f32];
    for (i, a) in keys.iter().enumerate() {
        for (j, b) in keys.iter().enumerate() {
            if i != j {
                assert_ne!(a, b);
                assert!(
                    !b.starts_with(&format!("{a}|")),
                    "the lookup prefix of {a:?} must not match {b:?}"
                );
            }
        }
    }
    // Same class, same adapter: the key is still shared across the class.
    assert_eq!(
        cache_key_text(
            "Some GPU",
            "Vulkan",
            "drv",
            "info",
            tune::class_bits(class),
            Only(Limb32)
        ),
        f32
    );

    let dir = std::env::temp_dir().join(format!("bigmont-key-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let cache = dir.join("tune.cache");
    tune::write_cache(&cache, &legacy, "unrolled 8 64|9");
    tune::write_cache(&cache, &auto, "unrolled 8 64|1");
    tune::write_cache(&cache, &f32, "unrolled32 8 64|2");
    tune::write_cache(&cache, &f13, "unrolled 4 64|3");
    // Rewriting each key replaces only its own line; a legacy line survives every write.
    tune::write_cache(&cache, &auto, "unrolled 16 64|4");
    tune::write_cache(&cache, &f32, "unrolled32 16 64|5");
    let mut lines: Vec<String> = std::fs::read_to_string(&cache)
        .unwrap()
        .lines()
        .map(str::to_string)
        .collect();
    lines.sort();
    let mut want = vec![
        format!("{legacy}|unrolled 8 64|9"),
        format!("{auto}|unrolled 16 64|4"),
        format!("{f32}|unrolled32 16 64|5"),
        format!("{f13}|unrolled 4 64|3"),
    ];
    want.sort();
    assert_eq!(lines, want);
    let _ = std::fs::remove_dir_all(dir);
}

// ---- the limb scheme dimension on a real adapter ----

fn skip_without_int64(gpu: &Gpu) -> bool {
    skip_unless(
        gpu.has_int64,
        "this adapter lacks SHADER_INT64, so no 32-bit kernel can run",
    )
}

#[test]
fn candidates_offer_a_scheme_only_when_the_adapter_caps_allow_it() {
    let Some(gpu) = gpu() else { return };
    if skip_unless(
        gpu.has_subgroups,
        "no subgroups on this adapter, so there are no candidates at all",
    ) {
        return;
    }
    let bits = tune::class_bits(tune::limb_class(1019));
    let auto = tune::candidates(&gpu, bits, Auto);
    let c13 = tune::candidates(&gpu, bits, Only(Limb13));
    let c32 = tune::candidates(&gpu, bits, Only(Limb32));
    assert_eq!(
        auto.iter().any(|c| c.scheme == Limb32),
        gpu.has_int64,
        "Auto offers 32-bit kernels iff the adapter has int64"
    );
    assert!(auto.iter().any(|c| c.scheme == Limb13));
    assert!(c13.iter().all(|c| c.scheme == Limb13) && !c13.is_empty());
    assert_eq!(c32.is_empty(), !gpu.has_int64);
    assert!(c32.iter().all(|c| c.scheme == Limb32));
    // Auto is exactly the union of the two forced sets.
    assert_eq!(auto.len(), c13.len() + c32.len());
    for cfg in &auto {
        assert!(gpu.caps().satisfies(cfg.requires()), "{cfg:?}");
        assert!(cfg.validate().is_ok(), "{cfg:?}");
        assert!(gpu.check_subgroups(cfg.tpi).is_ok());
        assert_eq!(
            *cfg,
            KernelConfig::for_bits_scheme(cfg.scheme, bits, cfg.tpi, cfg.wg, Variant::Unrolled)
        );
    }
}

#[test]
fn a_forced_32_bit_autotune_returns_a_verified_32_bit_kernel() {
    let Some(gpu) = gpu() else { return };
    if skip_without_int64(&gpu) || skip_unless(gpu.has_subgroups, "no subgroups") {
        return;
    }
    let (dir, cache) = temp_cache("only32");
    let t = tune::autotune(&gpu, 1019, Some(&cache), false, Only(Limb32))
        .expect("forced 32-bit autotune");
    assert_eq!(t.cfg.scheme, Limb32);
    assert!(!t.from_cache && t.squarings_per_sec > 0.0);
    let bits = tune::class_bits(tune::limb_class(1019));
    assert!(tune::candidates(&gpu, bits, Only(Limb32)).contains(&t.cfg));
    tune::self_test(&gpu, t.cfg, bits).expect("the winner passes its self-test again");
    let text = std::fs::read_to_string(&cache).unwrap();
    assert!(
        text.contains("|only-32|") && text.contains("|unrolled32 "),
        "{text}"
    );
    let again = tune::autotune(&gpu, 1019, Some(&cache), false, Only(Limb32)).expect("from cache");
    assert!(again.from_cache);
    assert_eq!(again.cfg, t.cfg);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn a_forced_13_bit_autotune_returns_a_13_bit_kernel() {
    let Some(gpu) = gpu() else { return };
    if skip_unless(gpu.has_subgroups, "no subgroups") {
        return;
    }
    let (dir, cache) = temp_cache("only13");
    let t = tune::autotune(&gpu, 1019, Some(&cache), false, Only(Limb13))
        .expect("forced 13-bit autotune");
    assert_eq!(t.cfg.scheme, Limb13);
    let text = std::fs::read_to_string(&cache).unwrap();
    assert!(
        text.contains("|only-13|") && text.contains("|unrolled "),
        "{text}"
    );
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn auto_caches_its_winner_with_the_scheme_and_a_forced_run_leaves_that_entry_alone() {
    let Some(gpu) = gpu() else { return };
    if skip_unless(gpu.has_subgroups, "no subgroups") {
        return;
    }
    let (dir, cache) = temp_cache("autoscheme");
    let bits = tune::class_bits(tune::limb_class(1019));
    let first = tune::autotune(&gpu, 1019, Some(&cache), false, Auto).expect("auto autotune");
    assert!(!first.from_cache);
    assert!(
        tune::candidates(&gpu, bits, Auto).contains(&first.cfg),
        "{:?}",
        first.cfg
    );
    eprintln!("Auto chose {:?} on {}", first.cfg, gpu.describe());
    let text = std::fs::read_to_string(&cache).unwrap();
    let line = text.lines().next().unwrap();
    let value = line
        .strip_prefix(&format!("{}|", tune::cache_key(&gpu, 1019, Auto)))
        .expect("the Auto key is unchanged");
    assert_eq!(
        tune::parse_value(value).map(|(s, t, w, _)| (s, t, w)),
        Some((first.cfg.scheme, first.cfg.tpi, first.cfg.wg))
    );
    assert!(!line.contains("only-"));

    let second = tune::autotune(&gpu, 1019, Some(&cache), false, Auto).expect("auto from cache");
    assert!(second.from_cache);
    assert_eq!(second.cfg, first.cfg);

    // A forced run adds its own line and does not replace or delete the Auto one.
    let forced = tune::autotune(&gpu, 1019, Some(&cache), false, Only(Limb13)).expect("forced 13");
    assert!(!forced.from_cache);
    let lines = std::fs::read_to_string(&cache)
        .unwrap()
        .lines()
        .filter(|l| !l.is_empty())
        .count();
    assert_eq!(lines, 2, "one Auto line and one forced line");
    let third =
        tune::autotune(&gpu, 1019, Some(&cache), false, Auto).expect("auto from cache again");
    assert!(
        third.from_cache,
        "the forced run must not have disturbed the Auto entry"
    );
    assert_eq!(third.cfg, first.cfg);
    // A forced retune replaces only the forced line.
    let _ = tune::autotune(&gpu, 1019, Some(&cache), true, Only(Limb13)).expect("forced retune");
    assert_eq!(
        std::fs::read_to_string(&cache)
            .unwrap()
            .lines()
            .filter(|l| !l.is_empty())
            .count(),
        2
    );
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn a_cache_line_from_another_scheme_or_an_unknown_word_is_a_miss() {
    let Some(gpu) = gpu() else { return };
    if skip_unless(gpu.has_subgroups, "no subgroups") {
        return;
    }
    let (dir, cache) = temp_cache("foreign");
    // Under Only(Limb13) a 32-bit line (even one planted under the forced key) is not a candidate.
    let key = tune::cache_key(&gpu, 1019, Only(Limb13));
    tune::write_cache(&cache, &key, "unrolled32 8 64|1e9");
    let t = tune::autotune(&gpu, 1019, Some(&cache), false, Only(Limb13)).expect("retune");
    assert!(!t.from_cache);
    assert_eq!(t.cfg.scheme, Limb13);
    assert_ne!(t.squarings_per_sec, 1e9);
    // An unknown first word is a miss under Auto too.
    let auto_key = tune::cache_key(&gpu, 1019, Auto);
    tune::write_cache(&cache, &auto_key, "unrolled64 8 64|1e9");
    let t = tune::autotune(&gpu, 1019, Some(&cache), false, Auto).expect("retune");
    assert!(!t.from_cache);
    assert_ne!(t.squarings_per_sec, 1e9);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn a_legacy_auto_cache_line_without_the_choice_field_is_a_miss_and_is_retuned() {
    let Some(gpu) = gpu() else { return };
    if skip_unless(gpu.has_subgroups, "no subgroups") {
        return;
    }
    let (dir, cache) = temp_cache("legacy");
    // A phase 1 line: key without the `any` field, a 13-bit shape that is a valid candidate here.
    let key = tune::cache_key(&gpu, 1019, Auto);
    let legacy = key.replacen("|any|", "|", 1);
    assert_ne!(legacy, key);
    tune::write_cache(&cache, &legacy, "unrolled 4 64|1e9");
    let t =
        tune::autotune(&gpu, 1019, Some(&cache), false, Auto).expect("retune past the legacy line");
    assert!(!t.from_cache, "a legacy line must not be served");
    assert_ne!(t.squarings_per_sec, 1e9);
    let text = std::fs::read_to_string(&cache).unwrap();
    assert!(
        text.lines().any(|l| l.starts_with(&format!("{key}|"))),
        "the new entry is written under the Auto key: {text}"
    );
    assert!(
        text.lines()
            .any(|l| l == format!("{legacy}|unrolled 4 64|1e9")),
        "the legacy line is left alone: {text}"
    );
    let again = tune::autotune(&gpu, 1019, Some(&cache), false, Auto).expect("from the new entry");
    assert!(again.from_cache);
    assert_eq!(again.cfg, t.cfg);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn a_candidate_list_without_32_bit_configs_never_matches_a_32_bit_cache_line() {
    // Pure: the lookup autotune uses on a cache file. An adapter without int64 offers no Limb32
    // candidate, so a cached `unrolled32` line is a miss however plausible its shape.
    let class_bits = tune::class_bits(tune::limb_class(1019));
    let shapes = |scheme| -> Vec<KernelConfig> {
        let mut v = Vec::new();
        for tpi in [4u32, 8, 16, 32] {
            for wg in [64u32, 128, 256] {
                v.push(KernelConfig::for_bits_scheme(
                    scheme,
                    class_bits,
                    tpi,
                    wg,
                    Variant::Unrolled,
                ));
            }
        }
        v
    };
    let key = "Some GPU|Vulkan|drv info|79";
    let line = |text: &str| vec![format!("{key}|{text}")];
    for text in [
        "unrolled32 8 64|1e9",
        "unrolled32 4 256|5",
        "unrolled32 32 64|7",
    ] {
        assert_eq!(
            tune::cache_lookup(&line(text), key, &shapes(Limb13)),
            None,
            "{text}: no Limb32 candidate"
        );
        let hit = tune::cache_lookup(&line(text), key, &shapes(Limb32))
            .unwrap_or_else(|| panic!("{text}: a Limb32 candidate matches"));
        assert_eq!(hit.0.scheme, Limb32);
    }
    // And the 13-bit line is a miss when only 32-bit candidates are offered (a forced 32 run).
    assert_eq!(
        tune::cache_lookup(&line("unrolled 8 64|1e9"), key, &shapes(Limb32)),
        None
    );
    let hit =
        tune::cache_lookup(&line("unrolled 8 64|2.5"), key, &shapes(Limb13)).expect("13-bit hit");
    assert_eq!(
        (hit.0.scheme, hit.0.tpi, hit.0.wg, hit.1),
        (Limb13, 8, 64, 2.5)
    );
    // Other keys, unknown words and non-candidate shapes are misses; the first matching line wins.
    assert_eq!(
        tune::cache_lookup(
            &["other|key|unrolled 8 64|1".to_string()],
            key,
            &shapes(Limb13)
        ),
        None
    );
    assert_eq!(
        tune::cache_lookup(&line("unrolled64 8 64|1"), key, &shapes(Limb13)),
        None
    );
    assert_eq!(
        tune::cache_lookup(&line("unrolled 8 100|1"), key, &shapes(Limb13)),
        None
    );
    let both = vec![
        format!("{key}|unrolled32 8 64|9"),
        format!("{key}|unrolled 8 64|3"),
    ];
    assert_eq!(
        tune::cache_lookup(&both, key, &shapes(Limb13)).map(|h| h.1),
        Some(3.0),
        "the 32-bit line is skipped, the 13-bit one served"
    );
}

#[test]
fn a_32_bit_cache_line_is_ignored_on_an_adapter_that_cannot_run_it() {
    let Some(gpu) = gpu() else { return };
    if gpu.has_int64 || !gpu.has_subgroups {
        // Not applicable here (not a missing environment): a 32-bit line is valid on this adapter.
        eprintln!(
            "N/A: this adapter has int64 (or no subgroups), so a 32-bit cache line is valid here"
        );
        return;
    }
    let (dir, cache) = temp_cache("noint64");
    tune::write_cache(
        &cache,
        &tune::cache_key(&gpu, 1019, Auto),
        "unrolled32 8 64|1e9",
    );
    let t = tune::autotune(&gpu, 1019, Some(&cache), false, Auto).expect("retune");
    assert!(!t.from_cache);
    assert_eq!(t.cfg.scheme, Limb13);
    let err = tune::autotune(&gpu, 1019, Some(&cache), false, Only(Limb32)).expect_err("no int64");
    assert!(err.contains("int64"), "{err}");
    let _ = std::fs::remove_dir_all(dir);
}
