mod common;
use bigmont_core::gpu::Gpu;
use bigmont_core::kernel::{KernelConfig, Variant};
use bigmont_core::runner::Runner;
use bigmont_oracle::*;
use num_bigint::BigUint;
use num_traits::One;
use std::collections::BTreeMap;

fn mersenne(p: u32) -> BigUint {
    (BigUint::one() << p) - 1u32
}

/// Runs the Fermat ladder on `ns` (all of `bits` bits) and compares each verdict with the oracle.
fn verify(bits: u32, tpi: u32, wg: u32, ns: &[BigUint], segment: Option<u32>, capacity: u32) {
    let cfg = KernelConfig::for_bits(bits, tpi, wg, Variant::Unrolled);
    let Some(gpu) = common::open_gpu(cfg) else { return };
    verify_on(&gpu, bits, tpi, wg, ns, segment, capacity);
}

/// Like `verify` on an already opened device; returns the number of probable primes among `ns`.
fn verify_on(gpu: &Gpu, bits: u32, tpi: u32, wg: u32, ns: &[BigUint], segment: Option<u32>, capacity: u32) -> usize {
    let cfg = KernelConfig::for_bits(bits, tpi, wg, Variant::Unrolled);
    let mut runner = Runner::new(gpu, cfg, bits, capacity).expect("runner");
    runner.segment_override = segment;
    let got = runner.fermat(gpu, ns).expect("fermat failed");
    assert_eq!(got.len(), ns.len());
    for (i, n) in ns.iter().enumerate() {
        assert_eq!(got[i], fermat_base2(n), "{cfg:?} candidate {i} (n = {n}, {} bits)", n.bits());
    }
    got.iter().filter(|&&v| v).count()
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

// ---- small bit lengths: the GPU is used down to 2-bit numbers (n = 3) ----

/// Runs `groups` (exact bit length -> candidates) through a `Runner` per length with `tpi`
/// (workgroup 64) and returns how many probable primes the GPU found.
fn verify_groups(gpu: &Gpu, tpi: u32, groups: &BTreeMap<u32, Vec<BigUint>>) -> usize {
    groups.iter().map(|(&bits, ns)| verify_on(gpu, bits, tpi, 64, ns, None, 256)).sum()
}

#[test]
fn every_odd_number_from_3_to_4095_matches_the_oracle() {
    // Bits 2..=12 exhaustively: 3 alone is the 2-bit group, then 5 and 7 (3 bits), 9..15 (4 bits), ...
    let mut groups: BTreeMap<u32, Vec<BigUint>> = BTreeMap::new();
    for v in (3u32..4096).step_by(2) {
        let n = BigUint::from(v);
        groups.entry(n.bits() as u32).or_default().push(n);
    }
    assert_eq!(groups.keys().copied().collect::<Vec<_>>(), (2..=12).collect::<Vec<_>>());
    assert_eq!(groups[&2], [BigUint::from(3u32)], "the 2-bit group is the single number 3");
    assert_eq!(groups.values().map(Vec::len).sum::<usize>(), 2047);
    let oracle_prps = groups.values().flatten().filter(|n| fermat_base2(n)).count();
    assert!(oracle_prps >= 563, "the 563 odd primes below 4096 all pass base 2: {oracle_prps}");
    let cfg = KernelConfig::for_bits(2, 8, 64, Variant::Unrolled);
    let Some(gpu) = common::open_gpu(cfg) else { return };
    for tpi in [8u32, 4] {
        assert_eq!(verify_groups(&gpu, tpi, &groups), oracle_prps, "tpi {tpi}: GPU probable-prime count");
    }
}

#[test]
fn the_smallest_candidate_n_equals_3_is_a_probable_prime_on_the_gpu() {
    let cfg = KernelConfig::for_bits(2, 8, 64, Variant::Unrolled);
    let Some(gpu) = common::open_gpu(cfg) else { return };
    for tpi in [8u32, 4, 16, 32] {
        let cfg = KernelConfig::for_bits(2, tpi, 64, Variant::Unrolled);
        if gpu.check_subgroups(tpi).is_err() {
            continue;
        }
        let mut runner = Runner::new(&gpu, cfg, 2, 16).expect("a 2-bit runner");
        assert_eq!(runner.fermat(&gpu, &[BigUint::from(3u32)]).unwrap(), [true], "tpi {tpi}");
        // Several copies, and a sub-wave count, still give one verdict each.
        assert_eq!(runner.fermat(&gpu, &vec![BigUint::from(3u32); 5]).unwrap(), [true; 5], "tpi {tpi}");
    }
}

#[test]
fn edge_lengths_around_the_word_and_limb_boundaries() {
    // 2^k-1 and 2^(k-1)+1 have exactly k bits; 2^k+1 has k+1 bits. Lengths straddle 13-bit limbs
    // (26, 39, 52, 65), 32-bit exponent words (32, 64) and the 64-bit CPU limb.
    let ks = [20u32, 31, 32, 33, 47, 48, 49, 63, 64, 65];
    let mut rng = XorShift64::new(0xB175);
    let mut groups: BTreeMap<u32, Vec<BigUint>> = BTreeMap::new();
    let mut add = |n: BigUint| groups.entry(n.bits() as u32).or_default().push(n);
    for &k in &ks {
        add(mersenne(k));
        add((BigUint::one() << (k - 1)) + 1u32);
        add((BigUint::one() << k) + 1u32);
        for _ in 0..8 {
            add(random_odd_modulus(&mut rng, k));
        }
        for _ in 0..3 {
            add(random_prp(&mut rng, k));
        }
    }
    // Known primes at the boundary lengths: 2^31-1, 2^61-1, 2^64-59, and the Fermat prime 65537.
    add(mersenne(31));
    add(mersenne(61));
    add((BigUint::one() << 64u32) - 59u32);
    add(BigUint::from(65_537u32));
    let oracle_prps = groups.values().flatten().filter(|n| fermat_base2(n)).count();
    assert!(oracle_prps >= 3 * ks.len(), "every length has probable primes: {oracle_prps}");
    let cfg = KernelConfig::for_bits(20, 8, 64, Variant::Unrolled);
    let Some(gpu) = common::open_gpu(cfg) else { return };
    for tpi in [8u32, 4] {
        assert_eq!(verify_groups(&gpu, tpi, &groups), oracle_prps, "tpi {tpi}: GPU probable-prime count");
    }
}

#[test]
fn runner_rejects_fewer_than_two_bits() {
    let cfg = KernelConfig::for_bits(2, 8, 64, Variant::Unrolled);
    let Some(gpu) = common::open_gpu(cfg) else { return };
    for bits in [0u32, 1] {
        let err = Runner::new(&gpu, cfg, bits, 16).err().unwrap_or_else(|| panic!("{bits} bits must be rejected"));
        assert!(err.contains("2-bit minimum") && err.contains(&format!("{bits} bits")), "unclear error: {err}");
    }
    assert!(Runner::new(&gpu, cfg, 2, 16).is_ok(), "2 bits is the smallest supported length");
}
