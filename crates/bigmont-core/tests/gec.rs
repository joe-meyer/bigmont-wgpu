//! The Gerbicz check in `Runner::fermat_checked`: clean runs agree with the oracle and never alarm,
//! injected faults alarm exactly the injected candidate.
mod common;
use bigmont_core::gpu::Gpu;
use bigmont_core::kernel::{KernelConfig, Variant};
use bigmont_core::runner::{Fault, Runner, Verdict};
use bigmont_core::scheme::LimbScheme;
use bigmont_oracle::*;
use num_bigint::BigUint;

fn open() -> Option<Gpu> {
    match Gpu::new() {
        Ok(g) => Some(g),
        Err(e) => {
            common::skip(format!("gec: no adapter: {e}"));
            None
        }
    }
}

fn runner(gpu: &Gpu, scheme: LimbScheme, tpi: u32, bits: u32, cap: u32) -> Option<Runner> {
    let cfg = KernelConfig::for_bits_scheme(scheme, bits, tpi, 64, Variant::Unrolled);
    if !gpu.caps().satisfies(cfg.requires()) {
        common::skip(format!(
            "{cfg:?}: the adapter lacks {}",
            gpu.caps().missing(cfg.requires()).join(" and ")
        ));
        return None;
    }
    if let Err(e) = gpu.check_subgroups(tpi) {
        common::skip(format!("{cfg:?}: {e}"));
        return None;
    }
    let mut r = Runner::new(gpu, cfg, bits, cap).expect("runner");
    r.enable_gec(gpu).expect("enable_gec");
    Some(r)
}

/// Random odd numbers plus a few probable primes, so both verdicts occur.
fn mixed(bits: u32, random: u32, prps: u32, seed: u64) -> Vec<BigUint> {
    let mut rng = XorShift64::new(seed);
    let mut v: Vec<BigUint> = (0..random)
        .map(|_| random_odd_modulus(&mut rng, bits))
        .collect();
    for _ in 0..prps {
        v.push(random_prp(&mut rng, bits));
    }
    v
}

fn truth(ns: &[BigUint]) -> Vec<Verdict> {
    ns.iter()
        .map(|n| {
            if fermat_base2(n) {
                Verdict::Prp
            } else {
                Verdict::Composite
            }
        })
        .collect()
}

#[test]
fn a_clean_checked_run_equals_the_oracle_and_never_alarms() {
    let Some(gpu) = open() else { return };
    for scheme in LimbScheme::ALL.iter().copied() {
        // 3 bits: no full block (the tail check covers it); 41/42: one block with a 0/1 iteration tail; then
        // several blocks; 2029 is the production size.
        for bits in [3u32, 41, 42, 81, 130, 300, 2029] {
            let Some(mut r) = runner(&gpu, scheme, 8, bits, 64) else {
                break;
            };
            let ns = mixed(bits, 12, 2, 40 + u64::from(bits));
            let got = r.fermat_checked(&gpu, &ns).expect("fermat_checked");
            assert_eq!(got, truth(&ns), "{scheme:?} {bits} bits");
        }
    }
}

#[test]
fn checked_and_unchecked_runs_agree_and_chunking_keeps_the_order() {
    let Some(gpu) = open() else { return };
    let Some(mut r) = runner(&gpu, LimbScheme::Limb13, 8, 300, 5) else {
        return;
    };
    let ns = mixed(300, 17, 3, 9); // 20 candidates over chunks of 5
    let checked = r.fermat_checked(&gpu, &ns).expect("checked");
    let plain = r.fermat(&gpu, &ns).expect("plain");
    let from_plain: Vec<Verdict> = plain
        .iter()
        .map(|&p| if p { Verdict::Prp } else { Verdict::Composite })
        .collect();
    assert_eq!(checked, from_plain);
}

#[test]
fn an_injected_fault_alarms_only_its_candidate() {
    let Some(gpu) = open() else { return };
    for scheme in LimbScheme::ALL.iter().copied() {
        let bits = 300u32; // 299 iterations: blocks 0..=6 and a 19-iteration tail
        let Some(mut r) = runner(&gpu, scheme, 8, bits, 5) else {
            break;
        };
        let ns = mixed(bits, 9, 3, 77); // 12 candidates, chunks of 5: index 7 is in the second chunk
        let want = truth(&ns);
        for (index, block) in [(0usize, 0u32), (7, 3), (11, 6), (3, 1)] {
            r.fault = Some(Fault { index, block });
            let got = r.fermat_checked(&gpu, &ns).expect("fermat_checked");
            for (i, v) in got.iter().enumerate() {
                if i == index {
                    assert_eq!(*v, Verdict::Alarm, "{scheme:?} index {index} block {block}");
                } else {
                    assert_eq!(
                        *v, want[i],
                        "{scheme:?} candidate {i} unaffected by a fault in {index}"
                    );
                }
            }
        }
        // The fault was consumed: a following clean run is clean.
        assert_eq!(r.fermat_checked(&gpu, &ns).expect("clean"), want);
        // A fault in the tail (block number nb = 7, a 19-iteration tail) is caught by the recomputation.
        for index in [0usize, 6, 11] {
            r.fault = Some(Fault { index, block: 7 });
            let got = r.fermat_checked(&gpu, &ns).expect("tail fault");
            for (i, v) in got.iter().enumerate() {
                let expect = if i == index { Verdict::Alarm } else { want[i] };
                assert_eq!(
                    *v, expect,
                    "{scheme:?} tail fault in {index}, candidate {i}"
                );
            }
        }
        // A block that does not exist flips nothing.
        r.fault = Some(Fault { index: 2, block: 8 });
        assert_eq!(r.fermat_checked(&gpu, &ns).expect("no such block"), want);
    }
}

#[test]
fn a_fault_in_a_probable_prime_alarms_instead_of_passing_it() {
    let Some(gpu) = open() else { return };
    let Some(mut r) = runner(&gpu, LimbScheme::Limb13, 8, 300, 16) else {
        return;
    };
    let ns = mixed(300, 2, 2, 5); // the last two are probable primes
    assert_eq!(truth(&ns)[3], Verdict::Prp);
    r.fault = Some(Fault { index: 3, block: 2 });
    assert_eq!(r.fermat_checked(&gpu, &ns).expect("run")[3], Verdict::Alarm);
}

#[test]
fn checking_needs_enable_gec() {
    let Some(gpu) = open() else { return };
    let cfg = KernelConfig::for_bits_scheme(LimbScheme::Limb13, 300, 8, 64, Variant::Unrolled);
    let mut r = Runner::new(&gpu, cfg, 300, 8).expect("runner");
    let err = r.fermat_checked(&gpu, &mixed(300, 2, 0, 1)).unwrap_err();
    assert!(err.contains("enable_gec"), "{err}");
}
