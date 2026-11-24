use bigmont_cli::tune;
use bigmont_core::gpu::Gpu;
use bigmont_core::kernel::{KernelConfig, Variant};

fn gpu() -> Option<Gpu> {
    match Gpu::new() {
        Ok(g) => Some(g),
        Err(e) if std::env::var("BIGMONT_ALLOW_SKIP").is_ok_and(|v| v == "1") => {
            eprintln!("SKIP no adapter: {e}");
            None
        }
        Err(e) => panic!("no adapter: {e} (set BIGMONT_ALLOW_SKIP=1 to allow skips)"),
    }
}

#[test]
fn candidates_are_valid_and_fit_the_adapter() {
    let Some(gpu) = gpu() else { return };
    for cfg in tune::candidates(&gpu, 2029) {
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
        let Some(cfg) = tune::candidates(&gpu, bits).into_iter().find(|c| c.tpi == 8 && c.wg == 64) else {
            eprintln!("SKIP no tpi 8 candidate on this adapter");
            return;
        };
        tune::self_test(&gpu, cfg, bits).unwrap_or_else(|e| panic!("{cfg:?}: {e}"));
    }
}

#[test]
fn self_test_rejects_a_config_too_small_for_the_numbers() {
    let Some(gpu) = gpu() else { return };
    let cfg = KernelConfig::new(8, 64, Variant::Unrolled).with_limbs(80); // too small for 2029 bits
    assert!(tune::self_test(&gpu, cfg, 2029).is_err());
}

#[test]
fn autotune_picks_a_verified_config_and_uses_its_cache() {
    let Some(gpu) = gpu() else { return };
    let dir = std::env::temp_dir().join(format!("bigmont-tune-test-{}", std::process::id()));
    let cache = dir.join("tune.cache");
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
