//! CPU reference ("oracle") for the 13-bit-limb Montgomery arithmetic used by the GPU kernels.
//! Everything here is plain big-integer arithmetic, deliberately unrelated to the GPU code's
//! tricks, so that the two can be compared.

use num_bigint::BigUint;
use num_traits::{One, Zero};

pub const LIMB_BITS: u32 = 13;
pub const LIMB_MASK: u32 = (1 << LIMB_BITS) - 1;
/// Limbs per number: 160 * 13 = 2080 bits.
pub const LIMBS: usize = 160;
/// Montgomery radix is R = 2^R_BITS.
pub const R_BITS: u32 = LIMB_BITS * LIMBS as u32;

/// Small deterministic RNG (xorshift64); test data only.
#[derive(Clone)]
pub struct XorShift64(u64);

impl XorShift64 {
    pub fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

/// Uniform random number below 2^bits.
pub fn random_bits(rng: &mut XorShift64, bits: u32) -> BigUint {
    let nbytes = bits.div_ceil(8) as usize;
    let mut bytes = Vec::with_capacity(nbytes + 8);
    while bytes.len() < nbytes {
        bytes.extend_from_slice(&rng.next_u64().to_le_bytes());
    }
    bytes.truncate(nbytes);
    BigUint::from_bytes_le(&bytes) & ((BigUint::one() << bits) - 1u32)
}

/// Odd number of exactly `bits` bits.
pub fn random_odd_modulus(rng: &mut XorShift64, bits: u32) -> BigUint {
    random_bits(rng, bits) | (BigUint::one() << (bits - 1)) | BigUint::one()
}

/// Random number in [0, n).
pub fn random_below(rng: &mut XorShift64, n: &BigUint) -> BigUint {
    random_bits(rng, n.bits() as u32 + 64) % n
}

/// Limbs needed so that R = 2^(13*limbs) exceeds 16n for every n below 2^bits. The Fermat ladder
/// doubles lazily (no reduction), so values reach 4n before the next squaring, which needs R > 16n.
pub fn limbs_for_bits(bits: u32) -> usize {
    (bits + 4).div_ceil(LIMB_BITS) as usize
}

/// Split x into `limbs` limbs of 13 bits (little-endian), one per u32.
pub fn pack_limbs_n(x: &BigUint, limbs: usize) -> Vec<u32> {
    assert!(x.bits() <= u64::from(LIMB_BITS) * limbs as u64, "value does not fit in {limbs} limbs");
    let d = x.to_u32_digits();
    (0..limbs)
        .map(|k| {
            let o = LIMB_BITS as usize * k;
            let (w, s) = (o / 32, o % 32);
            let lo = u64::from(d.get(w).copied().unwrap_or(0));
            let hi = u64::from(d.get(w + 1).copied().unwrap_or(0));
            (((lo | (hi << 32)) >> s) as u32) & LIMB_MASK
        })
        .collect()
}

/// Split x into LIMBS limbs of 13 bits (little-endian), one per u32.
pub fn pack_limbs(x: &BigUint) -> Vec<u32> {
    pack_limbs_n(x, LIMBS)
}

/// Inverse of `pack_limbs`; limbs may exceed 13 bits (carries are summed in).
pub fn unpack_limbs(limbs: &[u32]) -> BigUint {
    let mut digits = vec![0u32; (limbs.len() * LIMB_BITS as usize).div_ceil(32) + 3];
    for (k, &v) in limbs.iter().enumerate() {
        let o = LIMB_BITS as usize * k;
        let (w, s) = (o / 32, o % 32);
        let mut carry = u64::from(v) << s;
        let mut i = w;
        while carry != 0 {
            let sum = u64::from(digits[i]) + (carry & 0xffff_ffff);
            digits[i] = sum as u32;
            carry = (carry >> 32) + (sum >> 32);
            i += 1;
        }
    }
    BigUint::new(digits)
}

/// -n^-1 mod 2^13 for odd n (Newton iteration on the low word).
pub fn neg_inv_limb(n: &BigUint) -> u32 {
    let n0 = n.to_u32_digits().first().copied().unwrap_or(0);
    assert!(n0 & 1 == 1, "modulus must be odd");
    let mut inv = n0; // correct to 3 bits
    for _ in 0..4 {
        // 3 -> 6 -> 12 -> 24 -> 48 correct bits
        inv = inv.wrapping_mul(2u32.wrapping_sub(n0.wrapping_mul(inv)));
    }
    inv.wrapping_neg() & LIMB_MASK
}

/// n^-1 mod 2^bits for odd n (Newton / Hensel lifting).
pub fn inv_mod_pow2(n: &BigUint, bits: u32) -> BigUint {
    assert!(n.bit(0), "modulus must be odd");
    let mut inv = BigUint::one(); // n * 1 = 1 mod 2 for odd n
    let mut p = 1u32;
    while p < bits {
        p = (p * 2).min(bits);
        let modulus = BigUint::one() << p;
        let t = (n * &inv) % &modulus; // n * inv mod 2^p
        let two_minus_t = (BigUint::from(2u32) + &modulus - t) % &modulus;
        inv = (inv * two_minus_t) % &modulus;
    }
    inv
}

/// -n^-1 mod 2^32 for odd n: the Montgomery factor for 32-bit limbs.
pub fn neg_inv_u32(n: &BigUint) -> u32 {
    let n0 = n.to_u32_digits().first().copied().unwrap_or(0);
    assert!(n0 & 1 == 1, "modulus must be odd");
    let mut inv = n0; // correct to 3 bits
    for _ in 0..4 {
        inv = inv.wrapping_mul(2u32.wrapping_sub(n0.wrapping_mul(inv)));
    }
    inv.wrapping_neg()
}

/// Split x into `limbs` 32-bit limbs (little-endian).
pub fn pack_u32_limbs(x: &BigUint, limbs: usize) -> Vec<u32> {
    assert!(x.bits() <= 32 * limbs as u64, "value does not fit in {limbs} 32-bit limbs");
    let mut d = x.to_u32_digits();
    d.resize(limbs, 0);
    d
}

/// Inverse of `pack_u32_limbs`.
pub fn unpack_u32_limbs(limbs: &[u32]) -> BigUint {
    BigUint::new(limbs.to_vec())
}

/// x <- x^2 * R^-1 mod n, `squarings` times, with R = 2^r_bits. `n` must be odd. The result is in
/// [0, n); with `squarings == 0` it is `x0 % n`.
pub fn mont_square_chain_bits(n: &BigUint, x0: &BigUint, squarings: u32, r_bits: u32) -> BigUint {
    assert!(n.bit(0), "modulus must be odd");
    let mask = (BigUint::one() << r_bits) - 1u32;
    let ninv = inv_mod_pow2(n, r_bits);
    let mut x = x0 % n;
    for _ in 0..squarings {
        let x2 = &x * &x;
        // m = -x2 * n^-1 mod R, so that x2 + m*n is divisible by R
        let m = ((&mask + 1u32) - ((&x2 * &ninv) & &mask)) & &mask;
        x = ((x2 + m * n) >> r_bits) % n;
    }
    x
}

/// x <- x^2 * R^-1 mod n, `squarings` times, with R = 2^(13*limbs). `n` must be odd. The result is in
/// [0, n); with `squarings == 0` it is `x0 % n`.
pub fn mont_square_chain_n(n: &BigUint, x0: &BigUint, squarings: u32, limbs: usize) -> BigUint {
    assert!(n.bit(0), "modulus must be odd");
    let r_bits = LIMB_BITS * limbs as u32;
    let mask = (BigUint::one() << r_bits) - 1u32;
    let ninv = inv_mod_pow2(n, r_bits);
    let mut x = x0 % n;
    for _ in 0..squarings {
        let x2 = &x * &x;
        // m = -x2 * n^-1 mod R, so that x2 + m*n is divisible by R
        let m = ((&mask + 1u32) - ((&x2 * &ninv) & &mask)) & &mask;
        x = ((x2 + m * n) >> r_bits) % n;
    }
    x
}

pub fn mont_square_chain(n: &BigUint, x0: &BigUint, squarings: u32) -> BigUint {
    mont_square_chain_n(n, x0, squarings, LIMBS)
}

/// Base-2 Fermat test: 2^(n-1) = 1 (mod n). True for primes and base-2 pseudoprimes.
pub fn fermat_base2(n: &BigUint) -> bool {
    if n < &BigUint::from(2u32) {
        return false;
    }
    if n == &BigUint::from(2u32) {
        return true;
    }
    if !n.bit(0) {
        return false;
    }
    BigUint::from(2u32).modpow(&(n - 1u32), n).is_one()
}

/// Odd number of exactly `bits` bits that passes `fermat_base2` (small-prime trial filter first).
pub fn random_prp(rng: &mut XorShift64, bits: u32) -> BigUint {
    let small: Vec<u32> = (3u32..1000).filter(|&p| (2..p).take_while(|d| d * d <= p).all(|d| p % d != 0)).collect();
    loop {
        let c = random_odd_modulus(rng, bits);
        if small.iter().any(|&p| (&c % p).is_zero() && c != BigUint::from(p)) {
            continue;
        }
        if fermat_base2(&c) {
            return c;
        }
    }
}
