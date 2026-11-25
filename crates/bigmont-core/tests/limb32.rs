//! Oracle checks for the 32-bit-limb (u64 products) cooperative squaring kernel. Strict like the
//! other GPU tests: a missing adapter, SHADER_INT64 or subgroup width fails unless
//! BIGMONT_ALLOW_SKIP=1.
mod common;
use bigmont_core::gpu::{Batch, Gpu, run};
use bigmont_core::kernel::{KernelConfig, Variant};
use bigmont_core::scheme::LimbScheme;
use bigmont_oracle::*;
use num_bigint::BigUint;
use num_traits::One;

/// One device per test (not per check): opening many devices in one process is what exhausts the
/// Linux NVIDIA driver's static TLS (see the device-loss investigation).
fn open() -> Option<Gpu> {
    match Gpu::new() {
        Ok(g) => {
            eprintln!("adapter: {} ({:?})", g.info.name, g.info.backend);
            Some(g)
        }
        Err(e) => {
            common::skip(format!("limb32: no adapter: {e}"));
            None
        }
    }
}

/// `count` instances: random moduli of exactly `bits` bits, plus the all-ones modulus 2^bits - 1 and
/// the sparse modulus 2^(bits-1) + 1 (each with x = n - 1, x = 0, x = 1 and with a random x).
fn values(bits: u32, count: usize, seed: u64) -> Vec<(BigUint, BigUint)> {
    let mut rng = XorShift64::new(seed);
    let all_ones = (BigUint::one() << bits) - 1u32;
    let sparse = (BigUint::one() << (bits - 1)) + 1u32;
    let mut v = vec![
        (all_ones.clone(), &all_ones - 1u32),
        (sparse.clone(), &sparse - 1u32),
        (all_ones.clone(), BigUint::from(0u32)),
        (all_ones.clone(), BigUint::one()),
        (sparse.clone(), BigUint::from(0u32)),
        (sparse.clone(), BigUint::one()),
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

fn check(gpu: &Gpu, bits: u32, tpi: u32, wg: u32, count: usize, squarings: u32, repeats: u32) {
    let cfg = KernelConfig::for_bits_scheme(LimbScheme::Limb32, bits, tpi, wg, Variant::Unrolled);
    if !gpu.caps().satisfies(cfg.requires()) {
        common::skip(format!("{cfg:?}: the adapter lacks {}", gpu.caps().missing(cfg.requires()).join(" and ")));
        return;
    }
    if let Err(e) = gpu.check_subgroups(tpi) {
        common::skip(format!("{cfg:?}: {e}"));
        return;
    }
    let vals = values(bits, count, 0xc0de ^ u64::from(bits) ^ (u64::from(tpi) << 24) ^ (u64::from(wg) << 32));
    let batch = Batch::from_values_cfg(&vals, &cfg);
    let out = run(gpu, cfg, &batch, squarings, repeats).unwrap_or_else(|e| panic!("{cfg:?}: run failed: {e}"));
    let l = cfg.limbs as usize;
    assert_eq!(out.x.len(), count * l);
    let total = squarings * repeats;
    let mut bad = Vec::new();
    for (i, (n, x0)) in vals.iter().enumerate() {
        let got = cfg.scheme.unpack(&out.x[i * l..(i + 1) * l]);
        let why = if got >= n * 2u32 {
            Some("value not below 2n")
        } else if &got % n != cfg.scheme.chain(n, x0, total, l) {
            Some("wrong value")
        } else {
            None
        };
        if let Some(w) = why {
            bad.push(format!("{i} ({w})"));
        }
    }
    assert!(bad.is_empty(), "{cfg:?}, {squarings} x {repeats} squarings: {} of {count} instances wrong: {}", bad.len(), bad.join(", "));
}

#[test]
fn every_width_matches_the_oracle_at_small_sizes() {
    let Some(gpu) = open() else { return };
    for bits in [64, 130, 254] {
        for tpi in [4, 8, 16, 32] {
            check(&gpu, bits, tpi, 64, 130, 1, 1);
            check(&gpu, bits, tpi, 64, 130, 64, 2);
        }
    }
}

#[test]
fn the_2029_bit_class_matches_the_oracle() {
    let Some(gpu) = open() else { return };
    for tpi in [4, 8, 16, 32] {
        check(&gpu, 2029, tpi, 64, 70, 1, 1);
        check(&gpu, 2029, tpi, 128, 200, 257, 1);
    }
}

#[test]
fn other_sizes_and_workgroups_match_the_oracle() {
    let Some(gpu) = open() else { return };
    check(&gpu, 510, 8, 64, 300, 64, 2);
    check(&gpu, 1022, 4, 256, 300, 31, 3);
    check(&gpu, 1022, 16, 32, 100, 31, 1);
}
