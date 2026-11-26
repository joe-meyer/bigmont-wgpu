//! The general Montgomery multiply kernel against the oracle, both limb schemes.
mod common;
use bigmont_core::gpu::run_mul;
use bigmont_core::kernel::{KernelConfig, Variant};
use bigmont_core::scheme::LimbScheme;
use bigmont_oracle::*;
use num_bigint::BigUint;
use num_traits::One;

/// n values with awkward limb patterns plus random ones; operands up to 4n - 1 (what the ladder
/// feeds the kernel after a lazy doubling).
fn cases(bits: u32, seed: u64) -> Vec<(BigUint, BigUint, BigUint)> {
    let mut rng = XorShift64::new(seed);
    let mut ns: Vec<BigUint> = (0..40)
        .map(|_| random_odd_modulus(&mut rng, bits))
        .collect();
    ns.push((BigUint::one() << bits) - 1u32); // every limb all ones
    ns.push((BigUint::one() << (bits - 1)) + 1u32); // long zero runs
    let mut out = Vec::new();
    for n in ns {
        let four_n = &n * 4u32;
        let top = &four_n - 1u32;
        out.push((
            n.clone(),
            random_below(&mut rng, &four_n),
            random_below(&mut rng, &four_n),
        ));
        out.push((n.clone(), top.clone(), top.clone()));
        out.push((n.clone(), BigUint::from(0u32), top.clone()));
        out.push((n.clone(), BigUint::one(), BigUint::one()));
    }
    out
}

fn check(scheme: LimbScheme, tpi: u32, wg: u32, bits: u32) {
    let cfg = KernelConfig::for_bits_scheme(scheme, bits, tpi, wg, Variant::Unrolled);
    let Some(gpu) = common::open_gpu(cfg) else {
        return;
    };
    if !gpu.caps().satisfies(cfg.requires()) {
        common::skip(format!(
            "{cfg:?}: the adapter lacks {}",
            gpu.caps().missing(cfg.requires()).join(" and ")
        ));
        return;
    }
    let values = cases(bits, 5);
    let limbs = cfg.limbs as usize;
    let r_bits = cfg.r_bits();
    let out = run_mul(&gpu, cfg, &values, 1).expect("run_mul");
    for (i, (n, a, b)) in values.iter().enumerate() {
        let got_limbs = &out[i * limbs..(i + 1) * limbs];
        assert!(
            scheme.normalised(got_limbs),
            "{cfg:?} case {i}: limbs not normalised"
        );
        let got = scheme.unpack(got_limbs);
        assert!(got < n * 2u32, "{cfg:?} case {i}: result not below 2n");
        assert_eq!(
            &got % n,
            mont_mul_bits(n, a, b, r_bits),
            "{cfg:?} case {i}: wrong value (n = {n})"
        );
    }
}

#[test]
fn limb13_multiply_matches_the_oracle() {
    for tpi in [4, 8, 16, 32] {
        check(LimbScheme::Limb13, tpi, 64, 2029);
    }
    check(LimbScheme::Limb13, 8, 64, 100);
}

#[test]
fn limb32_multiply_matches_the_oracle() {
    for tpi in [4, 8, 16, 32] {
        check(LimbScheme::Limb32, tpi, 64, 2029);
    }
    check(LimbScheme::Limb32, 8, 64, 100);
}

#[test]
fn repeated_multiplication_chains_like_the_oracle() {
    let cfg = KernelConfig::for_bits_scheme(LimbScheme::Limb13, 300, 8, 64, Variant::Unrolled);
    let Some(gpu) = common::open_gpu(cfg) else {
        return;
    };
    let values = cases(300, 6);
    let out = run_mul(&gpu, cfg, &values, 3).expect("run_mul");
    let limbs = cfg.limbs as usize;
    for (i, (n, a, b)) in values.iter().enumerate() {
        let mut x = a.clone();
        for _ in 0..3 {
            x = mont_mul_bits(n, &x, b, cfg.r_bits());
        }
        assert_eq!(
            cfg.scheme.unpack(&out[i * limbs..(i + 1) * limbs]) % n,
            x % n,
            "case {i}"
        );
    }
}

#[test]
fn the_multiply_source_is_generated_for_the_right_targets() {
    use bigmont_core::kernel::Target;
    let c13 = KernelConfig::for_bits_scheme(LimbScheme::Limb13, 2029, 8, 64, Variant::Unrolled);
    assert!(c13.try_mul_wgsl_for(Target::Native).is_ok());
    assert!(c13.try_mul_wgsl_for(Target::Browser).is_ok());
    let c32 = KernelConfig::for_bits_scheme(LimbScheme::Limb32, 2029, 8, 64, Variant::Unrolled);
    assert!(c32.try_mul_wgsl_for(Target::Native).is_ok());
    assert!(c32.try_mul_wgsl_for(Target::Browser).is_err());
    let rolled = KernelConfig::new(8, 64, Variant::Rolled);
    assert!(rolled.try_mul_wgsl_for(Target::Native).is_err());
}
