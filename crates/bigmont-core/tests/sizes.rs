mod common;
use bigmont_core::gpu::{Batch, Gpu, run};
use bigmont_core::kernel::{KernelConfig, Variant};
use bigmont_core::scheme::LimbScheme;
use bigmont_oracle::*;
use num_bigint::BigUint;
use num_traits::One;

/// Squaring-mode check at an arbitrary limb count: moduli of `13 * limbs - 3` bits (R > 4n), plus
/// all-ones and sparse moduli, compared with the oracle at the same limb count.
fn check_size(cfg: KernelConfig, squarings: u32) {
    let Some(gpu) = common::open_gpu(cfg) else { return };
    let limbs = cfg.limbs as usize;
    let bits = limbs as u32 * 13 - 3;
    let mut values = Batch::random_values(70, bits, 9);
    let all_ones = (BigUint::one() << bits) - 1u32;
    values.push((all_ones.clone(), &all_ones - 1u32));
    let sparse = (BigUint::one() << (bits - 1)) + 1u32;
    values.push((sparse.clone(), &sparse - 1u32));
    let batch = Batch::from_values_n(&values, limbs);
    let result = run(&gpu, cfg, &batch, squarings, 1).expect("run failed");
    for (i, (n, x0)) in values.iter().enumerate() {
        let got_limbs = &result.x[i * limbs..(i + 1) * limbs];
        assert!(got_limbs.iter().all(|&l| l <= LIMB_MASK), "{cfg:?} instance {i}: limbs not normalised");
        let got = unpack_limbs(got_limbs);
        assert!(got < n * 2u32, "{cfg:?} instance {i}: result not below 2n");
        assert_eq!(&got % n, mont_square_chain_n(n, x0, squarings, limbs), "{cfg:?} instance {i}: wrong value");
    }
}

/// The 32-bit scheme's variant of [`check_size`]: moduli of `32 * limbs - 3` bits (R > 4n), plus
/// all-ones and sparse moduli, compared with the oracle chain at the same limb count.
fn check_size32(gpu: &Gpu, tpi: u32, limbs: u32, squarings: u32) {
    let mut cfg = KernelConfig::new(tpi, 64, Variant::Unrolled);
    cfg.scheme = LimbScheme::Limb32;
    cfg.limbs = limbs;
    if let Err(e) = gpu.check_subgroups(tpi) {
        common::skip(format!("{cfg:?}: {e}"));
        return;
    }
    let bits = limbs * 32 - 3;
    let mut values = Batch::random_values(70, bits, 9);
    let all_ones = (BigUint::one() << bits) - 1u32;
    values.push((all_ones.clone(), &all_ones - 1u32));
    let sparse = (BigUint::one() << (bits - 1)) + 1u32;
    values.push((sparse.clone(), &sparse - 1u32));
    let batch = Batch::from_values_cfg(&values, &cfg);
    let result = run(gpu, cfg, &batch, squarings, 1).expect("run failed");
    let l = limbs as usize;
    for (i, (n, x0)) in values.iter().enumerate() {
        let got = cfg.scheme.unpack(&result.x[i * l..(i + 1) * l]);
        assert!(got < n * 2u32, "{cfg:?} instance {i}: result not below 2n");
        assert_eq!(&got % n, cfg.scheme.chain(n, x0, squarings, l), "{cfg:?} instance {i}: wrong value");
    }
}

/// One device for a whole 32-bit test; `None` after a loud skip when the adapter lacks int64.
fn open32() -> Option<Gpu> {
    let mut cfg = KernelConfig::new(4, 64, Variant::Unrolled);
    cfg.scheme = LimbScheme::Limb32;
    let gpu = common::open_gpu(cfg)?;
    if !gpu.caps().satisfies(cfg.requires()) {
        common::skip(format!("{cfg:?}: the adapter lacks {}", gpu.caps().missing(cfg.requires()).join(" and ")));
        return None;
    }
    Some(gpu)
}

/// (tpi, limbs) for 32-bit limbs: 1, 2, 3, 4, 5, 9 and 16 limbs per lane at TPI 4, and 1, 2, 3, 8 at
/// TPI 8 (every count a multiple of its TPI).
const SIZES32: &[(u32, u32)] = &[(4, 4), (4, 8), (4, 12), (4, 16), (4, 20), (4, 36), (4, 64), (8, 8), (8, 16), (8, 24), (8, 64)];

#[test]
fn limb32_size_classes_match_oracle() {
    let Some(gpu) = open32() else { return };
    for &(tpi, limbs) in SIZES32 {
        check_size32(&gpu, tpi, limbs, 3);
    }
}

#[test]
fn limb32_size_classes_long_chain_match_oracle() {
    let Some(gpu) = open32() else { return };
    for &(tpi, limbs) in &[(4, 36), (8, 24), (4, 64)] {
        check_size32(&gpu, tpi, limbs, 128);
    }
}

/// (tpi, limbs): limbs per lane 20, 10, 5, 3, 30, 15, 7, 5, 6, 25, 50, 6, covering the in-body and
/// owner-modulo carry-pass layouts, remainders after the last in-body pass, and K not dividing TPI.
const SIZES: &[(u32, u32)] =
    &[(4, 80), (8, 80), (16, 80), (32, 96), (4, 120), (8, 120), (16, 112), (8, 40), (4, 24), (8, 200), (4, 200), (32, 192)];

#[test]
fn unrolled_size_classes_match_oracle() {
    for &(tpi, limbs) in SIZES {
        check_size(KernelConfig::new(tpi, 64, Variant::Unrolled).with_limbs(limbs), 3);
    }
}

#[test]
fn unrolled_size_classes_long_chain_match_oracle() {
    for &(tpi, limbs) in &[(4, 120), (8, 200), (16, 112)] {
        check_size(KernelConfig::new(tpi, 64, Variant::Unrolled).with_limbs(limbs), 128);
    }
}

#[test]
fn rolled_size_classes_match_oracle() {
    for &(tpi, limbs) in &[(4, 80), (8, 80), (8, 120), (16, 112)] {
        check_size(KernelConfig::new(tpi, 64, Variant::Rolled).with_limbs(limbs), 3);
    }
}

#[test]
fn for_bits_rounds_up_to_a_multiple_of_tpi() {
    let c = KernelConfig::for_bits(2029, 8, 64, Variant::Unrolled);
    assert_eq!(c.limbs, 160); // limbs_for_bits(2029) = 157, next multiple of 8
    let c = KernelConfig::for_bits(1024, 32, 64, Variant::Unrolled);
    assert_eq!(c.limbs, 96); // 80 is not enough for... 80 is a multiple of 8 but not of 32 -> 96
    assert!(KernelConfig::new(8, 64, Variant::Unrolled).with_limbs(81).validate().is_err());
    assert!(KernelConfig::new(8, 64, Variant::Unrolled).with_limbs(80).validate().is_ok());
}

#[test]
fn batch_limb_mismatch_is_an_error_not_a_panic() {
    let cfg = KernelConfig::new(8, 64, Variant::Unrolled);
    let Some(gpu) = common::open_gpu(cfg) else { return };
    let values = Batch::random_values(4, 1019, 1);
    let batch = Batch::from_values_n(&values, 80);
    assert!(run(&gpu, cfg, &batch, 1, 1).is_err());
}

/// One and two limbs per lane (the top limb of a lane is also its limb 0).
#[test]
fn unrolled_tiny_limbs_per_lane_match_oracle() {
    for &(tpi, limbs) in &[(32, 32), (16, 32), (4, 8), (8, 8)] {
        check_size(KernelConfig::new(tpi, 64, Variant::Unrolled).with_limbs(limbs), 3);
    }
}
