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
    assert_eq!(
        unpack_limbs(&[0xffff_ffff, 0xffff_ffff]),
        (BigUint::from(0xffff_ffffu32) * ((BigUint::one() << 13) + 1u32))
    );
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
        assert_eq!(
            (n0 * ninv + 1) & LIMB_MASK,
            0,
            "n * ninv must be -1 mod 2^13"
        );
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

#[test]
fn single_squaring_matches_definition_at_full_size() {
    // Independent of the oracle's m-based reduction: y = x0^2 * R^-1 mod n  <=>  y * R = x0^2 mod n.
    let mut rng = XorShift64::new(6);
    for _ in 0..5 {
        let n = random_odd_modulus(&mut rng, 2029);
        let x0 = random_below(&mut rng, &n);
        let y = mont_square_chain(&n, &x0, 1);
        assert_eq!((&y << R_BITS) % &n, (&x0 * &x0) % &n);
    }
}

#[test]
fn limbs_for_bits_covers_the_ladder_invariant() {
    assert_eq!(limbs_for_bits(2029), 157);
    assert_eq!(limbs_for_bits(1024), 80);
    for bits in [64u32, 127, 521, 1019, 2029, 3000] {
        let r = BigUint::one() << (13 * limbs_for_bits(bits) as u32);
        let n_max = (BigUint::one() << bits) - 1u32;
        assert!(r > &n_max * 16u32, "R must exceed 16n at {bits} bits");
    }
}

#[test]
fn pack_unpack_round_trip_at_other_limb_counts() {
    let mut rng = XorShift64::new(11);
    for limbs in [8usize, 40, 80, 96, 157] {
        let x = random_bits(&mut rng, (limbs as u32) * 13 - 1);
        let packed = pack_limbs_n(&x, limbs);
        assert_eq!(packed.len(), limbs);
        assert!(packed.iter().all(|&l| l <= LIMB_MASK));
        assert_eq!(unpack_limbs(&packed), x);
    }
}

#[test]
fn square_chain_n_satisfies_the_montgomery_relation() {
    let mut rng = XorShift64::new(5);
    for limbs in [40usize, 80, 120] {
        let bits = (limbs as u32) * 13 - 3;
        let n = random_odd_modulus(&mut rng, bits);
        let x = random_below(&mut rng, &n);
        let r = BigUint::one() << (13 * limbs as u32);
        let y = mont_square_chain_n(&n, &x, 1, limbs);
        assert_eq!((&y * &r) % &n, (&x * &x) % &n);
    }
}

#[test]
fn chain_bits_matches_chain_n_at_13_bit_radices() {
    let mut rng = XorShift64::new(21);
    for limbs in [6usize, 20, 80] {
        let bits = (limbs as u32) * 13 - 4;
        let n = random_odd_modulus(&mut rng, bits);
        let x = random_below(&mut rng, &n);
        for sq in [0u32, 1, 5] {
            assert_eq!(
                mont_square_chain_bits(&n, &x, sq, 13 * limbs as u32),
                mont_square_chain_n(&n, &x, sq, limbs)
            );
        }
    }
}

#[test]
fn chain_bits_satisfies_the_montgomery_relation_at_32_bit_radices() {
    // Independent of the m-based reduction: y = x^2 R^-1 mod n  <=>  y R = x^2 mod n.
    let mut rng = XorShift64::new(22);
    for (bits, words) in [(64u32, 3u32), (256, 9), (1024, 33)] {
        let n = random_odd_modulus(&mut rng, bits);
        let x = random_below(&mut rng, &n);
        let r_bits = 32 * words;
        let y = mont_square_chain_bits(&n, &x, 1, r_bits);
        assert!(y < n);
        assert_eq!((&y << r_bits) % &n, (&x * &x) % &n, "bits={bits}");
        // composition
        let three = mont_square_chain_bits(&n, &x, 3, r_bits);
        assert_eq!(
            three,
            mont_square_chain_bits(&n, &mont_square_chain_bits(&n, &x, 1, r_bits), 2, r_bits)
        );
    }
    // u128 cross-check at a small modulus, R = 2^96.
    let (n, x0) = (1_000_003u128, 777_777u128);
    let rinv = inv_u128(modpow_u128(2, 96, n), n);
    let one = x0 * x0 % n * rinv % n;
    assert_eq!(
        mont_square_chain_bits(&BigUint::from(n), &BigUint::from(x0), 1, 96),
        BigUint::from(one)
    );
}

#[test]
fn pack_unpack_u32_limbs_round_trip() {
    let mut rng = XorShift64::new(23);
    for (bits, limbs) in [
        (1u32, 1usize),
        (32, 1),
        (33, 2),
        (64, 3),
        (258, 9),
        (1056, 33),
    ] {
        let x = random_bits(&mut rng, bits);
        let packed = pack_u32_limbs(&x, limbs);
        assert_eq!(packed.len(), limbs);
        assert_eq!(unpack_u32_limbs(&packed), x, "bits={bits}");
    }
    assert_eq!(
        pack_u32_limbs(&BigUint::from(0x1_0000_0002u64), 3),
        vec![2, 1, 0]
    );
    assert_eq!(
        unpack_u32_limbs(&[0xffff_ffff, 0xffff_ffff]),
        BigUint::from(u64::MAX)
    );
    assert_eq!(unpack_u32_limbs(&[]), BigUint::from(0u32));
}

#[test]
#[should_panic]
fn pack_u32_limbs_rejects_values_that_do_not_fit() {
    pack_u32_limbs(&(BigUint::one() << 64u32), 2);
}

#[test]
fn neg_inv_u32_property() {
    let mut rng = XorShift64::new(24);
    for _ in 0..50 {
        let n = random_odd_modulus(&mut rng, 300);
        let n0 = n.to_u32_digits()[0];
        assert_eq!(
            n0.wrapping_mul(neg_inv_u32(&n)),
            u32::MAX,
            "n * ninv must be -1 mod 2^32"
        );
    }
}

#[test]
fn fermat_base2_known_values() {
    let mersenne = |p: u32| (BigUint::one() << p) - 1u32;
    assert!(fermat_base2(&mersenne(127)));
    assert!(fermat_base2(&mersenne(521)));
    assert!(!fermat_base2(&BigUint::from(91u32)));
    assert!(fermat_base2(&BigUint::from(341u32))); // smallest base-2 pseudoprime: the test is Fermat, not primality
    assert!(!fermat_base2(&BigUint::from(100u32)));
    assert!(fermat_base2(&BigUint::from(2u32)));
    assert!(!fermat_base2(&BigUint::one()));
}

#[test]
fn random_prp_passes_the_fermat_test() {
    let mut rng = XorShift64::new(3);
    let p = random_prp(&mut rng, 160);
    assert_eq!(p.bits(), 160);
    assert!(fermat_base2(&p));
}
