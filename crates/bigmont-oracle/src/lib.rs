//! CPU reference ("oracle") for the 13-bit-limb Montgomery arithmetic used by the GPU kernels.
//! Everything here is plain big-integer arithmetic, deliberately unrelated to the GPU code's
//! tricks, so that the two can be compared.

use num_bigint::BigUint;
use num_traits::One;

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

/// Split x into LIMBS limbs of 13 bits (little-endian), one per u32.
pub fn pack_limbs(x: &BigUint) -> Vec<u32> {
    assert!(x.bits() <= u64::from(R_BITS), "value does not fit in {LIMBS} limbs");
    let d = x.to_u32_digits();
    (0..LIMBS)
        .map(|k| {
            let o = LIMB_BITS as usize * k;
            let (w, s) = (o / 32, o % 32);
            let lo = u64::from(d.get(w).copied().unwrap_or(0));
            let hi = u64::from(d.get(w + 1).copied().unwrap_or(0));
            (((lo | (hi << 32)) >> s) as u32) & LIMB_MASK
        })
        .collect()
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

/// x <- x^2 * R^-1 mod n, `squarings` times, with R = 2^R_BITS. Result in [0, n).
pub fn mont_square_chain(n: &BigUint, x0: &BigUint, squarings: u32) -> BigUint {
    let mask = (BigUint::one() << R_BITS) - 1u32;
    let ninv = inv_mod_pow2(n, R_BITS);
    let mut x = x0.clone();
    for _ in 0..squarings {
        let x2 = &x * &x;
        // m = -x2 * n^-1 mod R, so that x2 + m*n is divisible by R
        let m = ((&mask + 1u32) - ((&x2 * &ninv) & &mask)) & &mask;
        x = ((x2 + m * n) >> R_BITS) % n;
    }
    x
}
