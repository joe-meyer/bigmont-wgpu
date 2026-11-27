//! Oracle checks for the experimental one-thread-per-number kernels (`bigmont_core::onethread`):
//! scheme A (13-bit limbs, 24-bit multiply-add, deferred carries) and scheme B (32-bit limbs,
//! u64 products, needs SHADER_INT64). Strict like the other GPU tests: a missing adapter, or a
//! missing SHADER_INT64 for scheme B, fails unless BIGMONT_ALLOW_SKIP=1.
mod common;
use bigmont_core::gpu::Gpu;
use bigmont_core::onethread::{OneThread, Scheme, run};
use bigmont_oracle::*;
use num_bigint::BigUint;
use num_traits::One;

const SIZES: &[u32] = &[64, 130, 256, 384, 512, 1024];

/// The selected adapter (or None after a loud skip); for `scheme` B also requires SHADER_INT64.
fn open(scheme: Scheme) -> Option<Gpu> {
    let gpu = match Gpu::new() {
        Ok(g) => g,
        Err(e) => {
            common::skip(format!("onethread {scheme:?}: no adapter: {e}"));
            return None;
        }
    };
    if scheme.needs_int64() && !gpu.has_int64 {
        common::skip(format!("onethread {scheme:?}: adapter {} lacks SHADER_INT64", gpu.info.name));
        return None;
    }
    eprintln!("adapter: {} ({:?}), int64 {}, scheme {scheme:?}", gpu.info.name, gpu.info.backend, gpu.has_int64);
    Some(gpu)
}

/// `count` instances: random moduli of exactly `bits` bits, plus the all-ones modulus 2^bits - 1
/// and the sparse modulus 2^(bits-1) + 1 (each with x = n - 1 and with a random x).
fn values(bits: u32, count: usize, seed: u64) -> Vec<(BigUint, BigUint)> {
    let mut rng = XorShift64::new(seed);
    let all_ones = (BigUint::one() << bits) - 1u32;
    let sparse = (BigUint::one() << (bits - 1)) + 1u32;
    let mut v = vec![
        (all_ones.clone(), &all_ones - 1u32),
        (sparse.clone(), &sparse - 1u32),
        (all_ones.clone(), random_below(&mut rng, &all_ones)),
        (sparse.clone(), random_below(&mut rng, &sparse)),
    ];
    while v.len() < count {
        let n = random_odd_modulus(&mut rng, bits);
        let x = random_below(&mut rng, &n);
        v.push((n, x));
    }
    v.truncate(count);
    v
}

/// Runs `squarings` per dispatch x `repeats` chained dispatches and compares every instance with the
/// oracle. Scheme A: limbs below 2^13 and value below 2n; scheme B: value below n. Both: value
/// mod n equals the oracle chain at the scheme's radix.
fn check(gpu: &Gpu, scheme: Scheme, bits: u32, wg: u32, count: usize, squarings: u32, repeats: u32) {
    let vals = values(bits, count, 0x5eed ^ u64::from(bits) ^ (u64::from(wg) << 20));
    check_values(gpu, scheme, bits, wg, vals, squarings, repeats);
}

fn check_values(gpu: &Gpu, scheme: Scheme, bits: u32, wg: u32, vals: Vec<(BigUint, BigUint)>, squarings: u32, repeats: u32) {
    let count = vals.len();
    let k = OneThread::new(scheme, bits, wg);
    let batch = k.batch(&vals);
    let out = run(gpu, &k, &batch, squarings, repeats).unwrap_or_else(|e| panic!("{k:?}: run failed: {e}"));
    let l = k.limbs();
    assert_eq!(out.x.len(), count * l);
    let total = squarings * repeats;
    let mut bad = Vec::new();
    for (i, (n, x0)) in vals.iter().enumerate() {
        let limbs = &out.x[i * l..(i + 1) * l];
        let got = k.decode(limbs);
        let why = if scheme.is_limb13() && limbs.iter().any(|&v| v > LIMB_MASK) {
            Some("limbs not normalised")
        } else if scheme.is_limb13() && got >= n * 2u32 {
            Some("value not below 2n")
        } else if !scheme.is_limb13() && &got >= n {
            Some("value not below n")
        } else if &got % n != mont_square_chain_bits(n, x0, total, k.r_bits()) {
            Some("wrong value")
        } else {
            None
        };
        if let Some(why) = why {
            bad.push(format!("{i} ({why})"));
        }
    }
    assert!(bad.is_empty(), "{k:?}, {count} instances, {total} squarings: {} failed: {}", bad.len(), bad.iter().take(20).cloned().collect::<Vec<_>>().join(", "));
}

fn all_sizes(scheme: Scheme) {
    let Some(gpu) = open(scheme) else { return };
    for &bits in SIZES {
        // 64 squarings in one dispatch; instance counts that are not multiples of the workgroup.
        check(&gpu, scheme, bits, 64, 200, 64, 1);
        // Short chains, two chained dispatches, other workgroup sizes.
        check(&gpu, scheme, bits, 128, 130, 1, 1);
        check(&gpu, scheme, bits, 256, 300, 3, 2);
    }
}

#[test]
fn limb13_matches_oracle_at_all_sizes() {
    all_sizes(Scheme::Limb13);
}

#[test]
fn wide32_matches_oracle_at_all_sizes() {
    all_sizes(Scheme::Wide32);
}

#[test]
fn wide32_split_matches_oracle_at_all_sizes() {
    all_sizes(Scheme::Wide32Split);
}

#[test]
fn limb13_sqr_matches_oracle_at_all_sizes() {
    all_sizes(Scheme::Limb13Sqr);
}

#[test]
fn wide32_sqr_matches_oracle_at_all_sizes() {
    all_sizes(Scheme::Wide32Sqr);
}

/// Worst-case inputs for the carry-headroom bounds: scheme A accepts x up to 2n - 1 (limbs all near
/// 2^13 for the all-ones modulus), and the symmetric kernel adds up to 20 / 40 products of the
/// maximum size per column between carry passes. The sizes put the limb count just below and above
/// multiples of the pass periods and go past the largest benchmark size, so the multi-pass paths run.
#[test]
fn limb13_symmetric_squaring_at_the_headroom_limits() {
    let Some(gpu) = open(Scheme::Limb13Sqr) else { return };
    for &bits in &[254u32, 256, 508, 510, 512, 520, 780, 1022, 1034, 1500] {
        let all_ones = (BigUint::one() << bits) - 1u32;
        let sparse = (BigUint::one() << (bits - 1)) + 1u32;
        let mut rng = XorShift64::new(0xface ^ u64::from(bits));
        let mut vals = Vec::new();
        for n in [all_ones.clone(), sparse.clone(), random_odd_modulus(&mut rng, bits), random_odd_modulus(&mut rng, bits)] {
            let two_n_minus_1 = &n * 2u32 - 1u32;
            vals.push((n.clone(), two_n_minus_1)); // largest input scheme A allows
            vals.push((n.clone(), &n - 1u32));
            vals.push((n.clone(), random_below(&mut rng, &(&n * 2u32))));
        }
        check_values(&gpu, Scheme::Limb13Sqr, bits, 64, vals.clone(), 1, 1);
        check_values(&gpu, Scheme::Limb13Sqr, bits, 64, vals, 33, 2);
    }
}

#[test]
fn wide32_symmetric_squaring_at_the_edges() {
    let Some(gpu) = open(Scheme::Wide32Sqr) else { return };
    for &bits in &[32u32, 33, 62, 64, 254, 382, 510, 1022, 1500] {
        let all_ones = (BigUint::one() << bits) - 1u32;
        let sparse = (BigUint::one() << (bits - 1)) + 1u32;
        let mut rng = XorShift64::new(0xbeef ^ u64::from(bits));
        let mut vals = Vec::new();
        for n in [all_ones.clone(), sparse.clone(), random_odd_modulus(&mut rng, bits), random_odd_modulus(&mut rng, bits)] {
            vals.push((n.clone(), &n - 1u32)); // largest input scheme B allows
            vals.push((n.clone(), BigUint::one()));
            vals.push((n.clone(), random_below(&mut rng, &n)));
        }
        check_values(&gpu, Scheme::Wide32Sqr, bits, 64, vals.clone(), 1, 1);
        check_values(&gpu, Scheme::Wide32Sqr, bits, 64, vals, 33, 2);
    }
}

#[test]
fn zero_squarings_return_the_input() {
    for scheme in [Scheme::Limb13, Scheme::Wide32, Scheme::Limb13Sqr, Scheme::Wide32Sqr] {
        let Some(gpu) = open(scheme) else { continue };
        check(&gpu, scheme, 256, 64, 65, 0, 1);
    }
}

#[test]
fn single_instance_and_long_chain() {
    for scheme in [Scheme::Limb13, Scheme::Wide32, Scheme::Wide32Split, Scheme::Limb13Sqr, Scheme::Wide32Sqr] {
        let Some(gpu) = open(scheme) else { continue };
        check(&gpu, scheme, 512, 64, 1, 257, 1);
        check(&gpu, scheme, 384, 64, 77, 500, 1);
    }
}

#[test]
fn wide32_without_int64_is_an_error_not_a_panic() {
    let Some(gpu) = open(Scheme::Limb13) else { return };
    if gpu.has_int64 {
        return; // nothing to check on this adapter
    }
    let k = OneThread::new(Scheme::Wide32, 256, 64);
    let batch = k.batch(&values(256, 4, 1));
    assert!(run(&gpu, &k, &batch, 1, 1).is_err());
}
