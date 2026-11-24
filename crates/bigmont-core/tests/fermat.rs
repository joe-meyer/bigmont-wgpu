mod common;
use bigmont_core::kernel::{KernelConfig, Variant};
use bigmont_core::runner::Runner;
use bigmont_oracle::*;
use num_bigint::BigUint;
use num_traits::One;

fn mersenne(p: u32) -> BigUint {
    (BigUint::one() << p) - 1u32
}

/// Runs the Fermat ladder on `ns` (all of `bits` bits) and compares each verdict with the oracle.
fn verify(bits: u32, tpi: u32, wg: u32, ns: &[BigUint], segment: Option<u32>, capacity: u32) {
    let cfg = KernelConfig::for_bits(bits, tpi, wg, Variant::Unrolled);
    let Some(gpu) = common::open_gpu(cfg) else { return };
    let mut runner = Runner::new(&gpu, cfg, bits, capacity).expect("runner");
    runner.segment_override = segment;
    let got = runner.fermat(&gpu, ns).expect("fermat failed");
    assert_eq!(got.len(), ns.len());
    for (i, n) in ns.iter().enumerate() {
        assert_eq!(got[i], fermat_base2(n), "{cfg:?} candidate {i} ({} bits)", n.bits());
    }
}

fn mixed(bits: u32, random: u32, prps: u32, seed: u64) -> Vec<BigUint> {
    let mut rng = XorShift64::new(seed);
    let mut ns: Vec<BigUint> = (0..random).map(|_| random_odd_modulus(&mut rng, bits)).collect();
    ns.extend((0..prps).map(|_| random_prp(&mut rng, bits)));
    ns
}

#[test]
fn mersenne_primes_and_composites_across_sizes() {
    // 2^127-1, 2^521-1, 2^1279-1, 2^2203-1 are prime. 2^2029-1 is composite but, like every 2^p-1
    // with p prime (2029 is prime), a base-2 Fermat pseudoprime: p | 2^p - 2 = n - 1, so
    // 2^(n-1) = 1 mod n. 2^1024-1 and 2^2030-1 (composite exponents) fail the test.
    for (p, expect) in [(127u32, true), (521, true), (1279, true), (2203, true), (2029, true), (2030, false), (1024, false)] {
        let n = mersenne(p);
        assert_eq!(fermat_base2(&n), expect);
        verify(p, 8, 64, &[n], None, 64);
    }
}

#[test]
fn mixed_batch_at_the_beacon_size_class_spans_chunks() {
    // 70 random + 3 probable primes, capacity 64 forces two chunks; segment 97 forces several dispatches.
    verify(2029, 8, 64, &mixed(2029, 70, 3, 21), Some(97), 64);
}

#[test]
fn every_cooperative_width_agrees_with_the_oracle() {
    let ns = mixed(1019, 20, 2, 5);
    for tpi in [4u32, 8, 16, 32] {
        verify(1019, tpi, 64, &ns, Some(200), 64);
    }
}

#[test]
fn other_bit_lengths_and_workgroup_sizes() {
    for (bits, wg) in [(64u32, 64u32), (127, 64), (1500, 128), (2029, 256), (3000, 64)] {
        verify(bits, 8, wg, &mixed(bits, 12, 1, u64::from(bits)), None, 32);
    }
}

#[test]
fn rejects_candidates_of_the_wrong_length_or_even() {
    let cfg = KernelConfig::for_bits(2029, 8, 64, Variant::Unrolled);
    let Some(gpu) = common::open_gpu(cfg) else { return };
    let mut runner = Runner::new(&gpu, cfg, 2029, 16).unwrap();
    assert!(runner.fermat(&gpu, &[mersenne(521)]).is_err());
    let even = BigUint::one() << 2028u32;
    assert!(runner.fermat(&gpu, &[even]).is_err());
}

#[test]
fn runner_is_reusable_across_calls() {
    let cfg = KernelConfig::for_bits(1019, 8, 64, Variant::Unrolled);
    let Some(gpu) = common::open_gpu(cfg) else { return };
    let mut runner = Runner::new(&gpu, cfg, 1019, 64).unwrap();
    for seed in [1u64, 2, 3] {
        let ns = mixed(1019, 10, 1, seed);
        let got = runner.fermat(&gpu, &ns).unwrap();
        for (n, g) in ns.iter().zip(got) {
            assert_eq!(g, fermat_base2(n));
        }
    }
}
