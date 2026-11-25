mod common;
use bigmont_core::gpu::{Batch, run};
use bigmont_core::kernel::{KernelConfig, Variant};
use bigmont_oracle::*;
use num_bigint::BigUint;
use num_traits::One;

/// Runs `cfg` on the selected adapter and compares every instance with the oracle.
/// Fails when there is no adapter or the adapter cannot run this configuration, unless
/// `BIGMONT_ALLOW_SKIP=1` (then it skips loudly).
fn check(cfg: KernelConfig) {
    check_chain(cfg, 131, 3);
}

/// Like `check` with `random` random instances (plus the 2 special ones) and `squarings` chained
/// squarings per dispatch.
fn check_chain(cfg: KernelConfig, random: u32, squarings: u32) {
    let Some(gpu) = common::open_gpu(cfg) else {
        return;
    };

    // 131 random + 2 special = 133 instances: spans many workgroups (wid.x > 0) and leaves a
    // partial last workgroup for every configuration used here.
    let mut values = Batch::random_values(random, 2029, 7);
    // Carry-ripple stress: n = 2^2029 - 1 (every limb all ones) and n = 2^2028 + 1 (long zero runs).
    let all_ones = (BigUint::one() << 2029u32) - 1u32;
    values.push((all_ones.clone(), &all_ones - 1u32));
    let sparse = (BigUint::one() << 2028u32) + 1u32;
    values.push((sparse.clone(), &sparse - 1u32));

    let batch = Batch::from_values(&values);
    let result = run(&gpu, cfg, &batch, squarings, 1).expect("run failed");
    for (i, (n, x0)) in values.iter().enumerate() {
        let limbs = &result.x[i * LIMBS..(i + 1) * LIMBS];
        assert!(
            limbs.iter().all(|&l| l <= LIMB_MASK),
            "instance {i}: limbs not normalised"
        );
        let got = unpack_limbs(limbs);
        assert!(got < n * 2u32, "instance {i}: result not below 2n");
        assert_eq!(
            &got % n,
            mont_square_chain(n, x0, squarings),
            "instance {i}: wrong value"
        );
    }
}

#[test]
fn tpi1_matches_oracle() {
    check(KernelConfig::new(1, 64, Variant::Rolled));
}

#[test]
fn tpi8_matches_oracle() {
    check(KernelConfig::new(8, 64, Variant::Rolled));
}

#[test]
fn tpi16_matches_oracle() {
    check(KernelConfig::new(16, 64, Variant::Rolled));
}

#[test]
fn tpi32_matches_oracle() {
    check(KernelConfig::new(32, 64, Variant::Rolled));
}

#[test]
fn tpi32_other_workgroup_sizes_match_oracle() {
    check(KernelConfig::new(32, 128, Variant::Rolled));
    check(KernelConfig::new(32, 256, Variant::Rolled));
}

/// The rolled kernel works at 4 lanes per number (40 limbs per lane) without changes.
#[test]
fn tpi4_matches_oracle() {
    check(KernelConfig::new(4, 64, Variant::Rolled));
}

#[test]
fn tpi4_unrolled_matches_oracle() {
    check(KernelConfig::new(4, 64, Variant::Unrolled));
    check(KernelConfig::new(4, 128, Variant::Unrolled));
    check(KernelConfig::new(4, 256, Variant::Unrolled));
}

#[test]
fn tpi8_unrolled_matches_oracle() {
    check(KernelConfig::new(8, 64, Variant::Unrolled));
}

#[test]
fn tpi16_unrolled_matches_oracle() {
    check(KernelConfig::new(16, 64, Variant::Unrolled));
}

#[test]
fn tpi32_unrolled_matches_oracle() {
    check(KernelConfig::new(32, 64, Variant::Unrolled));
}

#[test]
fn tpi32_unrolled_other_workgroup_sizes_match_oracle() {
    check(KernelConfig::new(32, 128, Variant::Unrolled));
    check(KernelConfig::new(32, 256, Variant::Unrolled));
}

/// Configuration validation needs no GPU.
#[test]
fn unrolled_validation_rejects_tpi1_and_accepts_cooperative_widths() {
    assert!(
        KernelConfig::new(1, 64, Variant::Unrolled)
            .validate()
            .is_err(),
        "unrolled with tpi 1 must be rejected"
    );
    assert!(KernelConfig::new(1, 64, Variant::Rolled).validate().is_ok());
    for tpi in [4, 8, 16, 32] {
        for variant in [Variant::Rolled, Variant::Unrolled] {
            let cfg = KernelConfig::new(tpi, 64, variant);
            assert!(
                cfg.validate().is_ok(),
                "{cfg:?} should validate: {:?}",
                cfg.validate()
            );
        }
    }
}

/// The speed sweep uses long dispatches (up to 256 squarings); check a long chain against the oracle too
/// (many more carry-ripple opportunities than the 3-squaring tests).
#[test]
fn unrolled_long_chain_matches_oracle() {
    for tpi in [4, 8, 16, 32] {
        check_chain(KernelConfig::new(tpi, 64, Variant::Unrolled), 70, 128);
    }
}
