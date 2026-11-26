//! The Gerbicz check in `Runner::fermat_checked`: clean runs agree with the oracle and never alarm,
//! and simulated device faults alarm exactly the hit candidates.
//!
//! Faults are simulated in the device, not in the runner: the runner is built from a mutated kernel
//! source (`common::faults`) through `Runner::new_with_source` / `Runner::enable_gec_with_source`,
//! and the real runner, kernels and host comparison run on top of it.
mod common;
use bigmont_core::gpu::Gpu;
use bigmont_core::kernel::{KernelConfig, Target, Variant};
use bigmont_core::runner::{Runner, Verdict};
use bigmont_core::scheme::LimbScheme;
use bigmont_oracle::*;
use common::faults::*;
use num_bigint::BigUint;

/// 300-bit candidates: 299 ladder iterations, blocks 0..=6 (`NB = 7`) and a 19-iteration tail.
const BITS: u32 = 300;
const NB: u32 = 7;
const TAIL_START: u32 = NB * GEC_BLOCK;

fn open() -> Option<Gpu> {
    match Gpu::new() {
        Ok(g) => Some(g),
        Err(e) => {
            common::skip(format!("gec: no adapter: {e}"));
            None
        }
    }
}

/// The configuration for `scheme` at `bits`, or `None` after a loud skip when the adapter cannot
/// run it.
fn config(gpu: &Gpu, scheme: LimbScheme, bits: u32) -> Option<KernelConfig> {
    let cfg = KernelConfig::for_bits_scheme(scheme, bits, 8, 64, Variant::Unrolled);
    if !gpu.caps().satisfies(cfg.requires()) {
        common::skip(format!(
            "{cfg:?}: the adapter lacks {}",
            gpu.caps().missing(cfg.requires()).join(" and ")
        ));
        return None;
    }
    if let Err(e) = gpu.check_subgroups(cfg.tpi) {
        common::skip(format!("{cfg:?}: {e}"));
        return None;
    }
    Some(cfg)
}

/// A checked runner built the production way (`Runner::new`, `enable_gec`).
fn runner(gpu: &Gpu, scheme: LimbScheme, bits: u32, cap: u32) -> Option<Runner> {
    let cfg = config(gpu, scheme, bits)?;
    let mut r = Runner::new(gpu, cfg, bits, cap).expect("runner");
    r.enable_gec(gpu).expect("enable_gec");
    Some(r)
}

/// A checked runner on a simulated faulty device: the generated Fermat and multiply sources go
/// through `fermat` and `mul` first.
fn faulty(
    gpu: &Gpu,
    scheme: LimbScheme,
    bits: u32,
    cap: u32,
    fermat: impl Fn(&str) -> String,
    mul: impl Fn(&str) -> String,
) -> Option<Runner> {
    let cfg = config(gpu, scheme, bits)?;
    let f = fermat(
        &cfg.try_fermat_wgsl_for(Target::host())
            .expect("fermat source"),
    );
    let m = mul(&cfg.try_mul_wgsl_for(Target::host()).expect("mul source"));
    let mut r = Runner::new_with_source(gpu, cfg, bits, cap, f).expect("runner from source");
    r.enable_gec_with_source(gpu, m)
        .expect("enable_gec from source");
    Some(r)
}

fn same(src: &str) -> String {
    src.to_string()
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

/// Twelve 300-bit candidates, the last three probable primes.
fn twelve() -> Vec<BigUint> {
    mixed(BITS, 9, 3, 77)
}

/// Runs `ns` on the faulty runner and asserts that exactly the candidates `hit` alarm and every
/// other one equals the oracle. Returns false when the adapter cannot run `scheme` (after a skip).
#[allow(clippy::too_many_arguments)]
fn expect_alarms(
    gpu: &Gpu,
    scheme: LimbScheme,
    bits: u32,
    cap: u32,
    ns: &[BigUint],
    fermat: impl Fn(&str) -> String,
    mul: impl Fn(&str) -> String,
    hit: impl Fn(usize) -> bool,
    what: &str,
) -> bool {
    let Some(mut r) = faulty(gpu, scheme, bits, cap, fermat, mul) else {
        return false;
    };
    let got = r.fermat_checked(gpu, ns).expect("fermat_checked");
    let want: Vec<Verdict> = truth(ns)
        .into_iter()
        .enumerate()
        .map(|(i, v)| if hit(i) { Verdict::Alarm } else { v })
        .collect();
    assert_eq!(got, want, "{scheme:?} {bits} bits, cap {cap}: {what}");
    true
}

#[test]
fn a_clean_checked_run_equals_the_oracle_and_never_alarms() {
    let Some(gpu) = open() else { return };
    for scheme in LimbScheme::ALL.iter().copied() {
        // 3 bits: no full block (the tail check covers it); 41/42: one block with a 0/1 iteration tail; then
        // several blocks; 2029 is the production size.
        for bits in [3u32, 41, 42, 81, 130, 300, 2029] {
            let Some(mut r) = runner(&gpu, scheme, bits, 64) else {
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
    let Some(mut r) = runner(&gpu, LimbScheme::Limb13, 300, 5) else {
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
fn a_register_fault_mid_block_alarms_only_its_candidate() {
    let Some(gpu) = open() else { return };
    let ns = twelve();
    assert_eq!(truth(&ns)[10], Verdict::Prp);
    for scheme in LimbScheme::ALL.iter().copied() {
        // One chunk: block 0, a middle block, the last block (NB - 1), and a probable prime (10),
        // which must alarm rather than pass.
        for (inst, block) in [(0u32, 0u32), (3, 3), (11, 6), (4, 1), (10, 2)] {
            let iteration = block * GEC_BLOCK + 19;
            let ran = expect_alarms(
                &gpu,
                scheme,
                BITS,
                16,
                &ns,
                |s| register_flip(s, scheme, inst, iteration),
                same,
                |i| i == inst as usize,
                &format!("register flip in slot {inst}, block {block}"),
            );
            if !ran {
                break;
            }
        }
        // Chunks of 5: slot 2 is candidate 2 and candidate 7 (the second chunk); the third chunk
        // has only slots 0 and 1, so nothing else is hit.
        expect_alarms(
            &gpu,
            scheme,
            BITS,
            5,
            &ns,
            |s| register_flip(s, scheme, 2, 3 * GEC_BLOCK + 19),
            same,
            |i| i == 2 || i == 7,
            "register flip in slot 2 of every chunk",
        );
    }
}

#[test]
fn a_register_fault_in_the_tail_is_caught_by_the_recomputation() {
    let Some(gpu) = open() else { return };
    let ns = twelve();
    for scheme in LimbScheme::ALL.iter().copied() {
        for inst in [0u32, 6, 11] {
            if !expect_alarms(
                &gpu,
                scheme,
                BITS,
                16,
                &ns,
                |s| register_flip(s, scheme, inst, TAIL_START + 9),
                same,
                |i| i == inst as usize,
                &format!("register flip in the tail, slot {inst}"),
            ) {
                break;
            }
        }
        // 20 bits: 19 iterations, no full block; the recomputation covers the whole run.
        let small = mixed(20, 6, 2, 3);
        expect_alarms(
            &gpu,
            scheme,
            20,
            16,
            &small,
            |s| register_flip(s, scheme, 1, 5),
            same,
            |i| i == 1,
            "register flip in a run with no full block",
        );
        assert_eq!(
            truth(&small)[7],
            Verdict::Prp,
            "slot 7 must be a probable prime"
        );
        expect_alarms(
            &gpu,
            scheme,
            20,
            16,
            &small,
            |s| register_flip(s, scheme, 7, 5),
            same,
            |i| i == 7,
            "register flip in a probable prime with no full block",
        );
    }
}

#[test]
fn a_memory_read_fault_at_a_dispatch_start_alarms() {
    let Some(gpu) = open() else { return };
    let ns = twelve();
    for scheme in LimbScheme::ALL.iter().copied() {
        for (inst, first) in [
            (1u32, 0u32),
            (5, 3 * GEC_BLOCK),
            (9, 6 * GEC_BLOCK),
            (2, TAIL_START),
        ] {
            if !expect_alarms(
                &gpu,
                scheme,
                BITS,
                16,
                &ns,
                |s| read_flip(s, scheme, inst, first),
                same,
                |i| i == inst as usize,
                &format!("read flip in slot {inst} at iteration {first}"),
            ) {
                break;
            }
        }
    }
}

#[test]
fn a_faulty_multiply_unit_alarms() {
    let Some(gpu) = open() else { return };
    let ns = twelve();
    for scheme in LimbScheme::ALL.iter().copied() {
        for inst in [0u32, 4, 11] {
            if !expect_alarms(
                &gpu,
                scheme,
                BITS,
                16,
                &ns,
                same,
                |s| mul_path_flip(s, scheme, inst),
                |i| i == inst as usize,
                &format!("multiply fault in slot {inst}"),
            ) {
                break;
            }
        }
    }
}

#[test]
fn a_corrupted_modulus_in_memory_alarms() {
    let Some(gpu) = open() else { return };
    let ns = twelve();
    for scheme in LimbScheme::ALL.iter().copied() {
        // A probable prime (11) and a composite (3); the flip in the first dispatch leaves only
        // block 0's ladder on the true modulus.
        for (inst, first) in [(11u32, 3 * GEC_BLOCK), (3, 3 * GEC_BLOCK), (11, 0)] {
            if !expect_alarms(
                &gpu,
                scheme,
                BITS,
                16,
                &ns,
                |s| cell_flip_n(s, inst, first),
                same,
                |i| i == inst as usize,
                &format!("modulus cell flip in slot {inst} at iteration {first}"),
            ) {
                break;
            }
        }
    }
}

/// A modulus that is wrong on every load (buffer intact), in every kernel: they all agree on the
/// wrong modulus. The unchecked run turns the probable prime into a silent false composite; the
/// identity still fails, because `A`, `B` and the `2^(sum V)` pass start from the host's
/// `1~ = R mod n`, which is not `1~` modulo the wrong modulus.
#[test]
fn a_modulus_wrong_on_every_load_alarms() {
    let Some(gpu) = open() else { return };
    let ns = twelve();
    assert_eq!(truth(&ns)[11], Verdict::Prp);
    for scheme in LimbScheme::ALL.iter().copied() {
        let Some(cfg) = config(&gpu, scheme, BITS) else {
            break;
        };
        let f = bad_modulus_path(&cfg.try_fermat_wgsl_for(Target::host()).unwrap(), 11);
        let mut r = Runner::new_with_source(&gpu, cfg, BITS, 16, f).expect("runner");
        let plain = r.fermat(&gpu, &ns).expect("plain");
        assert!(!plain[11], "{scheme:?}: the unchecked run must be fooled");
        for inst in [11u32, 3] {
            expect_alarms(
                &gpu,
                scheme,
                BITS,
                16,
                &ns,
                |s| bad_modulus_path(s, inst),
                |s| bad_modulus_path(s, inst),
                |i| i == inst as usize,
                &format!("modulus wrong on every load in slot {inst}"),
            );
        }
    }
}

#[test]
fn a_corrupted_montgomery_factor_in_memory_alarms() {
    let Some(gpu) = open() else { return };
    let ns = twelve();
    for scheme in LimbScheme::ALL.iter().copied() {
        for inst in [11u32, 3] {
            if !expect_alarms(
                &gpu,
                scheme,
                BITS,
                16,
                &ns,
                |s| cell_flip_ninv(s, inst, 3 * GEC_BLOCK),
                same,
                |i| i == inst as usize,
                &format!("ninv cell flip in slot {inst}"),
            ) {
                break;
            }
        }
    }
}

#[test]
fn a_corrupted_exponent_in_memory_alarms() {
    let Some(gpu) = open() else { return };
    let ns = twelve();
    for scheme in LimbScheme::ALL.iter().copied() {
        // Exponent bit 32 is read at iteration BITS - 2 - 32 = 266, in block 6, after the flip.
        for inst in [11u32, 3] {
            if !expect_alarms(
                &gpu,
                scheme,
                BITS,
                16,
                &ns,
                |s| cell_flip_e(s, inst, 3 * GEC_BLOCK),
                same,
                |i| i == inst as usize,
                &format!("exponent cell flip in slot {inst}"),
            ) {
                break;
            }
        }
    }
}

#[test]
fn a_fault_that_never_fires_leaves_the_run_clean() {
    let Some(gpu) = open() else { return };
    let ns = twelve();
    let never = 10_000u32; // beyond the 299 iterations
    for scheme in LimbScheme::ALL.iter().copied() {
        let ran = expect_alarms(
            &gpu,
            scheme,
            BITS,
            5,
            &ns,
            |s| {
                cell_flip_n(
                    &read_flip(&register_flip(s, scheme, 2, never), scheme, 2, never),
                    2,
                    never,
                )
            },
            |s| mul_path_flip(s, scheme, 1_000), // no such slot
            |_| false,
            "faults that never fire",
        );
        if !ran {
            break;
        }
    }
}

/// Ladder submissions of one checked 300-bit chunk when every submission runs at most `seg`
/// iterations: each block, the tail (both passes in each submission), the `A^(2^40)` pass and the
/// `2^(sum V)` pass over `40 + bitlen(NB)` bits, each split into `ceil(len / seg)` submissions.
fn checked_dispatches(seg: u32) -> u64 {
    let tail = BITS - 1 - TAIL_START;
    let l = GEC_BLOCK + (32 - NB.leading_zeros());
    u64::from(
        NB * GEC_BLOCK.div_ceil(seg)
            + tail.div_ceil(seg)
            + GEC_BLOCK.div_ceil(seg)
            + l.div_ceil(seg),
    )
}

#[test]
fn checked_submissions_are_split_like_the_plain_path() {
    let Some(gpu) = open() else { return };
    let ns = twelve();
    for scheme in LimbScheme::ALL.iter().copied() {
        let mut counts = Vec::new();
        for seg in [3u32, 1000] {
            let Some(mut r) = runner(&gpu, scheme, BITS, 16) else {
                return;
            };
            r.segment_override = Some(seg);
            let got = r.fermat_checked(&gpu, &ns).expect("fermat_checked");
            assert_eq!(got, truth(&ns), "{scheme:?}, segments of {seg}");
            assert_eq!(
                r.stats.dispatches,
                checked_dispatches(seg),
                "{scheme:?}, segments of {seg}"
            );
            counts.push(r.stats.dispatches);
        }
        assert!(counts[0] >= u64::from(NB * 14), "{scheme:?}: {counts:?}");
        assert!(counts[0] > counts[1], "{scheme:?}: {counts:?}");
        // A whole block in one submission is one submission per block, the tail and each check pass.
        assert_eq!(counts[1], u64::from(NB + 3), "{scheme:?}");
    }
}

/// Without an override the segment length adapts and is kept across blocks, passes and chunks: with
/// a target no submission reaches, it doubles from 1 (31 iterations in five submissions, then the
/// rest of block 0) and a later chunk runs every block, the tail and each check pass in one.
#[test]
fn the_adaptive_segment_length_carries_across_blocks_and_chunks() {
    let Some(gpu) = open() else { return };
    let ns = twelve();
    for scheme in LimbScheme::ALL.iter().copied() {
        let Some(mut r) = runner(&gpu, scheme, BITS, 6) else {
            return;
        };
        r.target_secs = 1e9;
        let got = r.fermat_checked(&gpu, &ns[..6]).expect("first chunk");
        assert_eq!(got, truth(&ns[..6]), "{scheme:?}");
        // Block 0: 1 + 2 + 4 + 8 + 16, then the remaining 9; block 1 onwards: one each.
        assert_eq!(r.stats.dispatches, u64::from(6 + NB - 1 + 3), "{scheme:?}");
        let before = r.stats.dispatches;
        let got = r.fermat_checked(&gpu, &ns[6..]).expect("second chunk");
        assert_eq!(got, truth(&ns[6..]), "{scheme:?}");
        assert_eq!(r.stats.dispatches - before, u64::from(NB + 3), "{scheme:?}");
    }
}

/// Faults keyed to a block start, mid-block and to the tail still alarm only their candidate when
/// every block is split into segments of 7.
#[test]
fn faults_still_alarm_with_split_blocks() {
    let Some(gpu) = open() else { return };
    let ns = twelve();
    for scheme in LimbScheme::ALL.iter().copied() {
        // (slot, iteration, a read flip at a dispatch start rather than a register flip, what)
        let cases = [
            (3u32, 3 * GEC_BLOCK + 19, false, "register flip mid-block"),
            (5, 3 * GEC_BLOCK, true, "read flip at a block start"),
            (6, TAIL_START + 9, false, "register flip in the tail"),
        ];
        for (inst, iteration, read, what) in cases {
            let fermat = |s: &str| match read {
                true => read_flip(s, scheme, inst, iteration),
                false => register_flip(s, scheme, inst, iteration),
            };
            let Some(mut r) = faulty(&gpu, scheme, BITS, 16, fermat, same) else {
                return;
            };
            r.segment_override = Some(7);
            let got = r.fermat_checked(&gpu, &ns).expect("fermat_checked");
            let want: Vec<Verdict> = truth(&ns)
                .into_iter()
                .enumerate()
                .map(|(i, v)| {
                    if i == inst as usize {
                        Verdict::Alarm
                    } else {
                        v
                    }
                })
                .collect();
            assert_eq!(got, want, "{scheme:?}, segments of 7: {what}");
        }
    }
}

#[test]
fn checking_needs_enable_gec() {
    let Some(gpu) = open() else { return };
    let cfg = KernelConfig::for_bits_scheme(LimbScheme::Limb13, 300, 8, 64, Variant::Unrolled);
    let mut r = Runner::new(&gpu, cfg, 300, 8).expect("runner");
    let err = r.fermat_checked(&gpu, &mixed(300, 2, 0, 1)).unwrap_err();
    assert!(err.contains("enable_gec"), "{err}");
}

#[test]
fn the_fault_helpers_find_their_anchors_in_every_generated_source() {
    for scheme in LimbScheme::ALL.iter().copied() {
        for bits in [20u32, 300, 2029] {
            let cfg = KernelConfig::for_bits_scheme(scheme, bits, 8, 64, Variant::Unrolled);
            let f = cfg.try_fermat_wgsl_for(Target::Native).expect("fermat");
            let m = cfg.try_mul_wgsl_for(Target::Native).expect("mul");
            // Bit 17: 13-bit limb 1 bit 4 (lane 1's x0 when a lane holds one limb), 32-bit limb 0.
            let flip = match (scheme, bits) {
                (LimbScheme::Limb13, 20) => "lane == 1u && fp.w == 0u) { x0 = x0 ^ 16u; }",
                (LimbScheme::Limb13, _) => "lane == 0u && fp.w == 0u) { x1 = x1 ^ 16u; }",
                (LimbScheme::Limb32, _) => "lane == 0u && fp.w == 0u) { x0 = x0 ^ 131072u; }",
            };
            assert!(
                register_flip(&f, scheme, 0, 0).contains(flip),
                "{scheme:?} {bits}"
            );
            assert!(
                read_flip(&f, scheme, 0, 0).contains(flip),
                "{scheme:?} {bits}"
            );
            let mul_flip = flip.replace(" && fp.w == 0u", "");
            assert!(mul_path_flip(&m, scheme, 0).contains(&mul_flip));
            for (mutated, decl) in [
                (cell_flip_n(&f, 0, 0), "read_write> n_limbs"),
                (cell_flip_ninv(&f, 0, 0), "read_write> ninv_buf"),
                (cell_flip_e(&f, 0, 0), "read_write> e_words"),
            ] {
                assert!(mutated.contains(decl), "{scheme:?} {bits}: {decl}");
                assert_eq!(mutated.lines().count(), f.lines().count() + 1);
            }
            for src in [&f, &m] {
                assert_ne!(bad_modulus_path(src, 0), *src, "{scheme:?} {bits}");
            }
        }
    }
}
