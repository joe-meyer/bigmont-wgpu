//! Base-2 Fermat verdicts through the `Runner`, for every limb scheme. Each test opens one device
//! and runs its checks for each `LimbScheme::ALL` entry the device supports (strict: an
//! unsupported scheme fails unless BIGMONT_ALLOW_SKIP=1).
mod common;
use bigmont_core::gpu::Gpu;
use bigmont_core::kernel::{KernelConfig, Variant};
use bigmont_core::runner::Runner;
use bigmont_core::scheme::LimbScheme;
use bigmont_oracle::*;
use num_bigint::BigUint;
use num_traits::One;
use std::collections::BTreeMap;

fn mersenne(p: u32) -> BigUint {
    (BigUint::one() << p) - 1u32
}

/// One device per test (not per check): opening many devices in one process exhausts the Linux
/// NVIDIA driver's static TLS. `None` after a loud skip.
fn open() -> Option<Gpu> {
    match Gpu::new() {
        Ok(g) => {
            eprintln!("adapter: {} ({:?})", g.info.name, g.info.backend);
            Some(g)
        }
        Err(e) => {
            common::skip(format!("fermat: no adapter: {e}"));
            None
        }
    }
}

/// Whether the device can run `scheme` at `tpi` lanes (capabilities and subgroup width); a loud
/// skip (a failure unless BIGMONT_ALLOW_SKIP=1) when it cannot.
fn supports(gpu: &Gpu, scheme: LimbScheme, tpi: u32) -> bool {
    let cfg = KernelConfig::for_bits_scheme(scheme, 2, tpi, 64, Variant::Unrolled);
    if !gpu.caps().satisfies(cfg.requires()) {
        common::skip(format!("{cfg:?}: the adapter lacks {}", gpu.caps().missing(cfg.requires()).join(" and ")));
        return false;
    }
    if let Err(e) = gpu.check_subgroups(tpi) {
        common::skip(format!("{cfg:?}: {e}"));
        return false;
    }
    true
}

/// Runs the Fermat ladder on `ns` (all of `bits` bits) with `scheme` and compares each verdict with
/// the oracle; returns the number of probable primes among `ns`.
#[allow(clippy::too_many_arguments)]
fn verify_on(gpu: &Gpu, scheme: LimbScheme, bits: u32, tpi: u32, wg: u32, ns: &[BigUint], segment: Option<u32>, capacity: u32) -> usize {
    let cfg = KernelConfig::for_bits_scheme(scheme, bits, tpi, wg, Variant::Unrolled);
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
    let Some(gpu) = open() else { return };
    for &scheme in LimbScheme::ALL {
        if !supports(&gpu, scheme, 8) {
            continue;
        }
        for (p, expect) in [(127u32, true), (521, true), (1279, true), (2203, true), (2029, true), (2030, false), (1024, false)] {
            let n = mersenne(p);
            assert_eq!(fermat_base2(&n), expect);
            verify_on(&gpu, scheme, p, 8, 64, &[n], None, 64);
        }
    }
}

#[test]
fn mixed_batch_at_the_beacon_size_class_spans_chunks() {
    // 70 random + 3 probable primes, capacity 64 forces two chunks; segment 97 forces several dispatches.
    let Some(gpu) = open() else { return };
    for &scheme in LimbScheme::ALL {
        if supports(&gpu, scheme, 8) {
            verify_on(&gpu, scheme, 2029, 8, 64, &mixed(2029, 70, 3, 21), Some(97), 64);
        }
    }
}

#[test]
fn every_cooperative_width_agrees_with_the_oracle() {
    let ns = mixed(1019, 20, 2, 5);
    let Some(gpu) = open() else { return };
    for &scheme in LimbScheme::ALL {
        for tpi in [4u32, 8, 16, 32] {
            if supports(&gpu, scheme, tpi) {
                verify_on(&gpu, scheme, 1019, tpi, 64, &ns, Some(200), 64);
            }
        }
    }
}

#[test]
fn other_bit_lengths_and_workgroup_sizes() {
    let Some(gpu) = open() else { return };
    for &scheme in LimbScheme::ALL {
        if !supports(&gpu, scheme, 8) {
            continue;
        }
        for (bits, wg) in [(64u32, 64u32), (127, 64), (1500, 128), (2029, 256), (3000, 64)] {
            verify_on(&gpu, scheme, bits, 8, wg, &mixed(bits, 12, 1, u64::from(bits)), None, 32);
        }
    }
}

#[test]
fn rejects_candidates_of_the_wrong_length_or_even() {
    let Some(gpu) = open() else { return };
    for &scheme in LimbScheme::ALL {
        if !supports(&gpu, scheme, 8) {
            continue;
        }
        let cfg = KernelConfig::for_bits_scheme(scheme, 2029, 8, 64, Variant::Unrolled);
        let mut runner = Runner::new(&gpu, cfg, 2029, 16).unwrap();
        assert!(runner.fermat(&gpu, &[mersenne(521)]).is_err());
        let even = BigUint::one() << 2028u32;
        assert!(runner.fermat(&gpu, &[even]).is_err());
    }
}

#[test]
fn runner_is_reusable_across_calls() {
    let Some(gpu) = open() else { return };
    for &scheme in LimbScheme::ALL {
        if !supports(&gpu, scheme, 8) {
            continue;
        }
        let cfg = KernelConfig::for_bits_scheme(scheme, 1019, 8, 64, Variant::Unrolled);
        let mut runner = Runner::new(&gpu, cfg, 1019, 64).unwrap();
        for seed in [1u64, 2, 3] {
            let ns = mixed(1019, 10, 1, seed);
            let got = runner.fermat(&gpu, &ns).unwrap();
            for (n, g) in ns.iter().zip(got) {
                assert_eq!(g, fermat_base2(n), "{cfg:?} seed {seed}");
            }
        }
    }
}

// ---- small bit lengths: the GPU is used down to 2-bit numbers (n = 3) ----

/// Runs `groups` (exact bit length -> candidates) through a `Runner` per length with `scheme` and
/// `tpi` (workgroup 64) and returns how many probable primes the GPU found.
fn verify_groups(gpu: &Gpu, scheme: LimbScheme, tpi: u32, groups: &BTreeMap<u32, Vec<BigUint>>) -> usize {
    groups.iter().map(|(&bits, ns)| verify_on(gpu, scheme, bits, tpi, 64, ns, None, 256)).sum()
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
    let Some(gpu) = open() else { return };
    for &scheme in LimbScheme::ALL {
        for tpi in [8u32, 4] {
            if supports(&gpu, scheme, tpi) {
                assert_eq!(verify_groups(&gpu, scheme, tpi, &groups), oracle_prps, "{scheme:?} tpi {tpi}: GPU probable-prime count");
            }
        }
    }
}

#[test]
fn the_smallest_candidate_n_equals_3_is_a_probable_prime_on_the_gpu() {
    let Some(gpu) = open() else { return };
    for &scheme in LimbScheme::ALL {
        for tpi in [8u32, 4, 16, 32] {
            if !supports(&gpu, scheme, tpi) {
                continue;
            }
            let cfg = KernelConfig::for_bits_scheme(scheme, 2, tpi, 64, Variant::Unrolled);
            let mut runner = Runner::new(&gpu, cfg, 2, 16).expect("a 2-bit runner");
            assert_eq!(runner.fermat(&gpu, &[BigUint::from(3u32)]).unwrap(), [true], "{scheme:?} tpi {tpi}");
            // Several copies, and a sub-wave count, still give one verdict each.
            assert_eq!(runner.fermat(&gpu, &vec![BigUint::from(3u32); 5]).unwrap(), [true; 5], "{scheme:?} tpi {tpi}");
        }
    }
}

/// For each length `k`: 2^k-1, 2^(k-1)+1 (exactly k bits), 2^k+1 (k+1 bits), random odd moduli and
/// probable primes, grouped by exact bit length.
fn boundary_groups(ks: &[u32], seed: u64) -> BTreeMap<u32, Vec<BigUint>> {
    let mut rng = XorShift64::new(seed);
    let mut groups: BTreeMap<u32, Vec<BigUint>> = BTreeMap::new();
    let mut add = |n: BigUint| groups.entry(n.bits() as u32).or_default().push(n);
    for &k in ks {
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
    groups
}

#[test]
fn edge_lengths_around_the_word_and_limb_boundaries() {
    // 2^k-1 and 2^(k-1)+1 have exactly k bits; 2^k+1 has k+1 bits. Lengths straddle 13-bit limbs
    // (26, 39, 52, 65), 32-bit limbs and exponent words (31, 32, 33, 62..65) and the 64-bit CPU limb.
    let ks = [3u32, 20, 26, 31, 32, 33, 39, 47, 48, 49, 52, 62, 63, 64, 65];
    let mut groups = boundary_groups(&ks, 0xB175);
    let mut add = |n: BigUint| groups.entry(n.bits() as u32).or_default().push(n);
    // Known primes at the boundary lengths: 2^31-1, 2^61-1, 2^64-59, and the Fermat prime 65537.
    add(mersenne(31));
    add(mersenne(61));
    add((BigUint::one() << 64u32) - 59u32);
    add(BigUint::from(65_537u32));
    let oracle_prps = groups.values().flatten().filter(|n| fermat_base2(n)).count();
    assert!(oracle_prps >= 3 * ks.len(), "every length has probable primes: {oracle_prps}");
    let Some(gpu) = open() else { return };
    for &scheme in LimbScheme::ALL {
        for tpi in [8u32, 4] {
            if supports(&gpu, scheme, tpi) {
                assert_eq!(verify_groups(&gpu, scheme, tpi, &groups), oracle_prps, "{scheme:?} tpi {tpi}: GPU probable-prime count");
            }
        }
    }
}

#[test]
fn lengths_at_the_32_bit_limb_class_boundaries() {
    // limbs_for_bits(bits) = ceil((bits + 4) / 32) steps up at 32k - 3, so 32k - 4 is the longest
    // number of a class and 32k - 3 the shortest of the next: R is barely above 16n at the first.
    let ks: Vec<u32> = [2u32, 3, 8, 64].iter().flat_map(|k| [32 * k - 4, 32 * k - 3]).collect();
    assert_eq!(ks, [60, 61, 92, 93, 252, 253, 2044, 2045]);
    let groups = boundary_groups(&ks, 0x32B1);
    let oracle_prps = groups.values().flatten().filter(|n| fermat_base2(n)).count();
    assert!(oracle_prps >= 3 * ks.len(), "every length has probable primes: {oracle_prps}");
    let Some(gpu) = open() else { return };
    for &scheme in LimbScheme::ALL {
        for tpi in [8u32, 4] {
            if supports(&gpu, scheme, tpi) {
                assert_eq!(verify_groups(&gpu, scheme, tpi, &groups), oracle_prps, "{scheme:?} tpi {tpi}: GPU probable-prime count");
            }
        }
    }
}

#[test]
fn runner_rejects_fewer_than_two_bits() {
    let Some(gpu) = open() else { return };
    for &scheme in LimbScheme::ALL {
        if !supports(&gpu, scheme, 8) {
            continue;
        }
        let cfg = KernelConfig::for_bits_scheme(scheme, 2, 8, 64, Variant::Unrolled);
        for bits in [0u32, 1] {
            let err = Runner::new(&gpu, cfg, bits, 16).err().unwrap_or_else(|| panic!("{bits} bits must be rejected"));
            assert!(err.contains("2-bit minimum") && err.contains(&format!("{bits} bits")), "unclear error: {err}");
        }
        assert!(Runner::new(&gpu, cfg, 2, 16).is_ok(), "2 bits is the smallest supported length");
    }
}
