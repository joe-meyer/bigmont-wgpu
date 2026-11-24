use bigmont_oracle::*;
use num_bigint::BigUint;
use num_traits::One;

fn modpow_u128(mut b: u128, mut e: u128, m: u128) -> u128 {
    let mut r = 1 % m;
    b %= m;
    while e > 0 {
        if e & 1 == 1 {
            r = r * b % m;
        }
        b = b * b % m;
        e >>= 1;
    }
    r
}

/// Inverse of a modulo m (m small), by the extended Euclidean algorithm.
fn inv_u128(a: u128, m: u128) -> u128 {
    let (mut old_r, mut r) = (a as i128, m as i128);
    let (mut old_s, mut s) = (1i128, 0i128);
    while r != 0 {
        let q = old_r / r;
        (old_r, r) = (r, old_r - q * r);
        (old_s, s) = (s, old_s - q * s);
    }
    assert_eq!(old_r, 1, "not invertible");
    old_s.rem_euclid(m as i128) as u128
}

#[test]
fn pack_unpack_roundtrip() {
    let mut rng = XorShift64::new(1);
    for bits in [1, 13, 14, 31, 32, 33, 1000, 2029, 2080] {
        let x = random_bits(&mut rng, bits);
        let limbs = pack_limbs(&x);
        assert_eq!(limbs.len(), LIMBS);
        assert!(limbs.iter().all(|&l| l <= LIMB_MASK));
        assert_eq!(unpack_limbs(&limbs), x, "bits={bits}");
    }
}

#[test]
fn unpack_accepts_unnormalized_limbs() {
    // 2 * 2^13 + 5 written as limbs [5 + 2^13 * 2, 0] must equal limbs [5, 2].
    assert_eq!(unpack_limbs(&[5 + (2 << 13), 0]), unpack_limbs(&[5, 2]));
    assert_eq!(unpack_limbs(&[0xffff_ffff, 0xffff_ffff]), (BigUint::from(0xffff_ffffu32) * ((BigUint::one() << 13) + 1u32)));
}

#[test]
fn random_modulus_has_exact_size_and_is_odd() {
    let mut rng = XorShift64::new(2);
    for _ in 0..50 {
        let n = random_odd_modulus(&mut rng, 2029);
        assert_eq!(n.bits(), 2029);
        assert_eq!(n.to_u32_digits()[0] & 1, 1);
    }
}

#[test]
fn neg_inv_limb_property() {
    let mut rng = XorShift64::new(3);
    for _ in 0..50 {
        let n = random_odd_modulus(&mut rng, 2029);
        let n0 = n.to_u32_digits()[0] & LIMB_MASK;
        let ninv = neg_inv_limb(&n);
        assert!(ninv <= LIMB_MASK);
        assert_eq!((n0 * ninv + 1) & LIMB_MASK, 0, "n * ninv must be -1 mod 2^13");
    }
}

#[test]
fn inv_mod_pow2_property() {
    let mut rng = XorShift64::new(4);
    for bits in [1, 13, 100, 2080] {
        let n = random_odd_modulus(&mut rng, 2029);
        let inv = inv_mod_pow2(&n, bits);
        let mask = (BigUint::one() << bits) - 1u32;
        assert_eq!((&n * &inv) & &mask, BigUint::one() & &mask, "bits={bits}");
    }
}

#[test]
fn chain_matches_independent_small_case() {
    // n and x0 small enough for u128 arithmetic; R = 2^2080.
    let (n, x0) = (1_000_003u128, 123_456u128);
    let r_mod_n = modpow_u128(2, u128::from(R_BITS), n);
    let rinv = inv_u128(r_mod_n, n);
    let one_step = x0 * x0 % n * rinv % n;
    let two_steps = one_step * one_step % n * rinv % n;
    let (nb, xb) = (BigUint::from(n), BigUint::from(x0));
    assert_eq!(mont_square_chain(&nb, &xb, 1), BigUint::from(one_step));
    assert_eq!(mont_square_chain(&nb, &xb, 2), BigUint::from(two_steps));
    assert_eq!(mont_square_chain(&nb, &xb, 0), xb);
}

#[test]
fn chain_composes_at_full_size() {
    let mut rng = XorShift64::new(5);
    let n = random_odd_modulus(&mut rng, 2029);
    let x0 = random_below(&mut rng, &n);
    let three = mont_square_chain(&n, &x0, 3);
    let one_then_two = mont_square_chain(&n, &mont_square_chain(&n, &x0, 1), 2);
    assert_eq!(three, one_then_two);
    assert!(three < n);
}
