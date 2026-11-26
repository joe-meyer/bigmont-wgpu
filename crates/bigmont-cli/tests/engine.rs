use bigmont_cli::engine::{Backend, Engine, Options};
use bigmont_oracle::*;
use num_bigint::BigUint;
use num_traits::One;

mod common;

fn opts(backend: Backend) -> Options {
    Options {
        backend,
        min_gpu_batch: 4,
        capacity: 32,
        verify_rate: 0,
        ..Options::default()
    }
}

fn batch() -> (Vec<BigUint>, Vec<u8>) {
    let mut rng = XorShift64::new(9);
    let mut ns: Vec<BigUint> = (0..10)
        .map(|_| random_odd_modulus(&mut rng, 1019))
        .collect();
    ns.extend((0..2).map(|_| random_prp(&mut rng, 1019)));
    ns.extend((0..9).map(|_| random_odd_modulus(&mut rng, 777))); // second bit length
    ns.extend([
        BigUint::from(2u32),
        BigUint::from(3u32),
        BigUint::from(4u32),
        BigUint::one(),
        BigUint::from(0u32),
        (BigUint::one() << 1019u32) + 2u32,
    ]);
    let want = ns.iter().map(|n| u8::from(fermat_base2(n))).collect();
    (ns, want)
}

/// Odd candidates of at least 3 (the ones that need a backend): the engine's verdict rules resolve
/// 0, 1, 2 and the even numbers without one.
fn needs_backend(n: &BigUint) -> bool {
    n.bit(0) && n >= &BigUint::from(3u32)
}

/// (PRPs, composites) among the candidates that take the GPU path with `opts(Backend::Gpu)`: odd,
/// at least 3, in a bit-length bucket of at least `min_gpu_batch` (4) entries. The bucket size is the
/// only condition: the bit length does not matter, down to the 2-bit number 3.
fn gpu_path_counts(ns: &[BigUint], want: &[u8]) -> (u64, u64) {
    let on_gpu = |n: &BigUint| {
        needs_backend(n)
            && ns
                .iter()
                .filter(|m| m.bits() == n.bits() && needs_backend(m))
                .count()
                >= 4
    };
    let (mut prps, mut composites) = (0, 0);
    for (n, &w) in ns.iter().zip(want) {
        if on_gpu(n) {
            if w == 1 { prps += 1 } else { composites += 1 }
        }
    }
    (prps, composites)
}

/// Removes a file when dropped, also when the test fails.
struct RemoveOnDrop(std::path::PathBuf);
impl Drop for RemoveOnDrop {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

#[test]
fn cpu_backend_matches_the_oracle_including_degenerate_inputs() {
    let (ns, want) = batch();
    let mut e = Engine::open(opts(Backend::Cpu));
    assert_eq!(e.verdicts(&ns).unwrap(), want);
}

#[test]
fn gpu_backend_matches_the_oracle_and_buckets_by_bit_length() {
    if !common::usable_gpu() {
        return;
    }
    let (ns, want) = batch();
    let cache =
        std::env::temp_dir().join(format!("bigmont-engine-test-{}.cache", std::process::id()));
    let _cleanup = RemoveOnDrop(cache.clone());
    let mut e = Engine::open(Options {
        cache: Some(cache),
        ..opts(Backend::Gpu)
    });
    eprintln!("{}", e.describe());
    assert_eq!(e.verdicts(&ns).unwrap(), want);
    let r = e.report();
    assert_eq!(r.prp_disagreements, 0);
    assert!(
        r.gpu_tests > 0,
        "the 1019-bit bucket has 12 entries, above min_gpu_batch 4: it must run on the GPU"
    );
    // Both bit-length buckets (1019 and 777 bits) are above min_gpu_batch. The 2-bit `3` is alone in
    // its bucket, below min_gpu_batch (4), so it is the only candidate the CPU counts as a test; 0, 1,
    // 2, 4 and the even number follow the verdict rules.
    assert_eq!(
        r.cpu_tests, 1,
        "only the 2-bit candidate 3 (a bucket of one) is tested on the CPU"
    );
    let (gpu_prps, _) = gpu_path_counts(&ns, &want);
    assert!(
        gpu_prps >= 2,
        "the batch holds at least the two planted probable primes"
    );
    assert_eq!(r.gpu_errors, 0);
    assert!(
        r.rechecks >= gpu_prps,
        "every GPU PRP must be re-run on the CPU: {} rechecks, {gpu_prps} PRPs",
        r.rechecks
    );
}

#[test]
fn auto_falls_back_to_cpu_when_no_gpu_works() {
    // BIGMONT_ADAPTER selects an adapter that does not exist, so GPU open fails; Auto must still answer.
    let (ns, want) = batch();
    let selector = bigmont_core::gpu::AdapterSelector {
        name: Some("no-such-adapter-xyz".into()),
        ..Default::default()
    };
    let mut e = Engine::open(Options {
        selector,
        ..opts(Backend::Auto)
    });
    assert!(e.describe().to_lowercase().contains("cpu"));
    assert_eq!(e.verdicts(&ns).unwrap(), want);
}

#[test]
fn forced_gpu_backend_errors_when_the_adapter_is_missing() {
    let selector = bigmont_core::gpu::AdapterSelector {
        name: Some("no-such-adapter-xyz".into()),
        ..Default::default()
    };
    let mut e = Engine::open(Options {
        selector,
        ..opts(Backend::Gpu)
    });
    let (ns, _) = batch();
    assert!(e.verdicts(&ns).is_err());
}

#[test]
fn verify_rate_rechecks_composites_without_changing_verdicts() {
    if !common::usable_gpu() {
        return;
    }
    let (ns, want) = batch();
    let mut e = Engine::open(Options {
        verify_rate: 2,
        ..opts(Backend::Gpu)
    });
    assert_eq!(e.verdicts(&ns).unwrap(), want);
    let r = e.report();
    assert_eq!(r.composite_disagreements, 0);
    assert_eq!(r.prp_disagreements, 0);
    assert_eq!(r.gpu_errors, 0, "the GPU path must not have fallen back");
    let (prps, composites) = gpu_path_counts(&ns, &want);
    assert_eq!(
        r.gpu_tests,
        prps + composites,
        "every odd candidate in a bucket of at least min_gpu_batch runs on the GPU, whatever its bit length"
    );
    // Every PRP, plus every 2nd GPU composite (counted across the call).
    assert_eq!(r.rechecks, prps + composites / 2);
    assert!(r.rechecks >= 7);
}

// ---- small bit lengths run on the GPU; a bucket skips the GPU only for size (min_gpu_batch), besides a failed autotune or Runner Err ----

/// The limb-class field of every line of a tuning cache file, sorted (the line format is
/// `name|backend|driver|any|class|unrolled tpi wg|rate`; a forced scheme has `only-13`/`only-32` in
/// place of `any`, and a 32-bit winner `unrolled32`). Read from the file, independent of the engine's
/// own counters.
fn cache_classes(path: &std::path::Path) -> Vec<String> {
    let text = std::fs::read_to_string(path).unwrap_or_default();
    let mut classes: Vec<String> = text
        .lines()
        .filter(|l| !l.is_empty())
        .map(|l| {
            l.rsplitn(3, '|')
                .nth(2)
                .and_then(|key| key.rsplit('|').next())
                .unwrap_or_else(|| panic!("malformed cache line {l:?}"))
                .to_string()
        })
        .collect();
    classes.sort();
    classes
}

fn temp_cache(tag: &str) -> (std::path::PathBuf, RemoveOnDrop) {
    let path = std::env::temp_dir().join(format!(
        "bigmont-engine-test-{tag}-{}.cache",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&path);
    (path.clone(), RemoveOnDrop(path))
}

/// `random` random odd numbers and `prps` probable primes of exactly `bits` bits.
fn bucket(bits: u32, random: usize, prps: usize, seed: u64) -> Vec<BigUint> {
    let mut rng = XorShift64::new(seed);
    let mut ns: Vec<BigUint> = (0..random)
        .map(|_| random_odd_modulus(&mut rng, bits))
        .collect();
    ns.extend((0..prps).map(|_| random_prp(&mut rng, bits)));
    ns
}

#[test]
fn buckets_under_64_bits_run_on_the_gpu_at_the_default_batch_size() {
    if !common::usable_gpu() {
        return;
    }
    // The default min_gpu_batch (64) with buckets of 70 at 20, 40, 63 and 64 bits: all on the GPU.
    // A 63-entry bucket at 30 bits is one short of min_gpu_batch: that, and only that, sends it to the CPU.
    let defaults = Options::default();
    assert_eq!(
        defaults.min_gpu_batch, 64,
        "this test is about the default batch size"
    );
    let mut ns = Vec::new();
    for bits in [20u32, 40, 63, 64] {
        ns.extend(bucket(bits, 67, 3, u64::from(bits)));
    }
    ns.extend(bucket(30, 60, 3, 30));
    let want: Vec<u8> = ns.iter().map(|n| u8::from(fermat_base2(n))).collect();
    // Expected counts derived from the inputs, not from engine counters.
    let big: Vec<&BigUint> = ns.iter().filter(|n| n.bits() != 30).collect();
    assert_eq!(big.len(), 4 * 70);
    let gpu_prps = big.iter().filter(|n| fermat_base2(n)).count() as u64;
    assert!(
        gpu_prps >= 12,
        "at least the 3 planted PRPs per bucket: {gpu_prps}"
    );

    let (cache, _cleanup) = temp_cache("buckets");
    let mut e = Engine::open(Options {
        backend: Backend::Gpu,
        capacity: 64,
        verify_rate: 0,
        cache: Some(cache.clone()),
        ..defaults
    });
    eprintln!("{}", e.describe());
    assert_eq!(e.verdicts(&ns).unwrap(), want);
    let r = e.report();
    assert_eq!(
        r.gpu_tests,
        4 * 70,
        "the 20, 40, 63 and 64-bit buckets all run on the GPU"
    );
    assert_eq!(
        r.cpu_tests, 63,
        "only the 63-entry bucket (below min_gpu_batch) runs on the CPU"
    );
    assert_eq!(r.gpu_errors, 0);
    assert_eq!((r.prp_disagreements, r.composite_disagreements), (0, 0));
    assert_eq!(
        r.rechecks, gpu_prps,
        "every GPU PRP is re-run on the CPU (verify_rate 0: no composite samples)"
    );
    // 20 and 40 bits share limb class 4 and 63 and 64 bits share class 6 (see tune::limb_class).
    assert_eq!(
        r.tuned_classes, 2,
        "four bit lengths in two limb classes tune twice"
    );
    // Independent of the counter: the cache file holds exactly one line per class (4 and 6), not one
    // per bit length. (The 30-bit bucket ran on the CPU and tuned nothing.)
    assert_eq!(
        cache_classes(&cache),
        ["4", "6"],
        "one cache line per limb class"
    );
}

#[test]
fn every_bit_length_down_to_two_bits_runs_on_the_gpu_and_tunes_one_limb_class_once() {
    if !common::usable_gpu() {
        return;
    }
    // Every odd number from 3 to 2047, four copies each, so each bit length from 2 (the number 3) to
    // 11 has a bucket of at least min_gpu_batch (4): ten distinct bit lengths in one limb class.
    let ns: Vec<BigUint> = (3u32..2048)
        .step_by(2)
        .flat_map(|v| std::iter::repeat_n(BigUint::from(v), 4))
        .collect();
    let want: Vec<u8> = ns.iter().map(|n| u8::from(fermat_base2(n))).collect();
    let bit_lengths: std::collections::BTreeSet<u64> = ns.iter().map(BigUint::bits).collect();
    assert_eq!(bit_lengths.len(), 10);
    let prps = want.iter().filter(|&&w| w == 1).count() as u64;

    let (cache, _cleanup) = temp_cache("tiny");
    let mut e = Engine::open(Options {
        backend: Backend::Gpu,
        min_gpu_batch: 4,
        capacity: 256,
        verify_rate: 0,
        cache: Some(cache.clone()),
        ..Options::default()
    });
    assert_eq!(e.verdicts(&ns).unwrap(), want);
    let r = e.report();
    assert_eq!(
        r.gpu_tests,
        ns.len() as u64,
        "all 2- to 11-bit candidates run on the GPU"
    );
    assert_eq!(r.cpu_tests, 0);
    assert_eq!(r.gpu_errors, 0);
    assert_eq!(r.rechecks, prps);
    assert_eq!(
        r.tuned_classes, 1,
        "ten bit lengths, one limb class: autotuned once, not ten times"
    );
    assert_eq!(
        cache_classes(&cache),
        ["4"],
        "exactly one cache line for the ten bit lengths of class 4"
    );
}

// ---- the limb scheme option: every choice gives the oracle's verdicts across the limb-count boundaries ----

/// Mixed bit lengths around the 13-bit and 32-bit limb-count boundaries (40, 100, 252, 254, 255, 300 and
/// the 2,029-bit Phase 0 size; 252, 254 and 255 share one 13-bit tuning class, 20 limbs, but 252 needs
/// 8 32-bit limbs and 254 and 255 need 9), each in a bucket large enough for the GPU: random odd numbers, plus planted
/// probable primes where they are cheap to find.
fn scheme_batch() -> (Vec<BigUint>, Vec<u8>) {
    let mut ns = Vec::new();
    for (i, bits) in [40u32, 100, 252, 254, 255, 300, 2029]
        .into_iter()
        .enumerate()
    {
        let prps = if bits <= 300 { 2 } else { 0 };
        ns.extend(bucket(bits, 6, prps, 100 + i as u64));
    }
    let want = ns.iter().map(|n| u8::from(fermat_base2(n))).collect();
    (ns, want)
}

fn run_with_scheme(
    scheme: bigmont_cli::tune::SchemeChoice,
    tag: &str,
) -> Option<(std::path::PathBuf, RemoveOnDrop)> {
    use bigmont_cli::tune::SchemeChoice;
    if !common::usable_gpu() {
        return None;
    }
    let gpu = match bigmont_core::gpu::Gpu::new() {
        Ok(g) => g,
        Err(e) => {
            assert!(
                std::env::var("BIGMONT_ALLOW_SKIP").is_ok_and(|v| v == "1"),
                "no adapter: {e} (set BIGMONT_ALLOW_SKIP=1 to allow skips)"
            );
            eprintln!("SKIP no adapter: {e}");
            return None;
        }
    };
    if scheme == SchemeChoice::Only(bigmont_core::scheme::LimbScheme::Limb32) && !gpu.has_int64 {
        assert!(
            std::env::var("BIGMONT_ALLOW_SKIP").is_ok_and(|v| v == "1"),
            "the adapter lacks SHADER_INT64 (set BIGMONT_ALLOW_SKIP=1 to allow skips)"
        );
        eprintln!("SKIP the adapter lacks SHADER_INT64");
        return None;
    }
    drop(gpu); // one device at a time: the engine opens its own
    let (ns, want) = scheme_batch();
    let (cache, guard) = temp_cache(tag);
    let mut e = Engine::open(Options {
        scheme,
        cache: Some(cache.clone()),
        ..opts(Backend::Gpu)
    });
    eprintln!("{} with {scheme:?}", e.describe());
    assert_eq!(e.verdicts(&ns).unwrap(), want, "{scheme:?}");
    let r = e.report();
    assert!(r.gpu_tests > 0, "{scheme:?}: nothing ran on the GPU");
    assert_eq!(
        r.gpu_tests,
        ns.len() as u64,
        "{scheme:?}: every bucket has at least min_gpu_batch entries"
    );
    assert_eq!(r.gpu_errors, 0, "{scheme:?}");
    assert_eq!(
        (r.prp_disagreements, r.composite_disagreements),
        (0, 0),
        "{scheme:?}"
    );

    // The extra self-test of retargeted kernels: derive from the cache file which (class, limb count)
    // pairs build a different program from the tuned shape's at the class representative, and require
    // the engine to have self-tested exactly those, once each. 13-bit shapes never differ.
    use bigmont_cli::tune;
    use bigmont_core::kernel::{KernelConfig, Variant};
    let text = std::fs::read_to_string(&cache).unwrap();
    let mut expected = std::collections::BTreeSet::new();
    let bit_lengths: std::collections::BTreeSet<u32> = ns.iter().map(|n| n.bits() as u32).collect();
    for bits in bit_lengths {
        let class = tune::limb_class(bits);
        let line = text
            .lines()
            .find(|l| {
                l.rsplitn(3, '|')
                    .nth(2)
                    .and_then(|key| key.rsplit('|').next())
                    == Some(class.to_string().as_str())
            })
            .unwrap_or_else(|| panic!("no cache line for class {class}: {text}"));
        let mut f = line.rsplitn(3, '|');
        let (rate, shape) = (f.next().unwrap(), f.next().unwrap());
        let (s, tpi, wg, _) =
            tune::parse_value(&format!("{shape}|{rate}")).expect("a parsable cache value");
        let tuned =
            KernelConfig::for_bits_scheme(s, tune::class_bits(class), tpi, wg, Variant::Unrolled);
        let exact = tuned.retarget(bits);
        if exact.limbs != tuned.limbs {
            expected.insert((class, exact.limbs));
        }
    }
    eprintln!(
        "{scheme:?}: {} retargeted kernels need their own self-test: {expected:?}",
        expected.len()
    );
    assert_eq!(
        r.retarget_self_tests,
        expected.len() as u64,
        "{scheme:?}: retargeted self-tests (class, limbs) {expected:?}"
    );
    if matches!(
        scheme,
        SchemeChoice::Only(bigmont_core::scheme::LimbScheme::Limb13)
    ) {
        assert_eq!(
            r.retarget_self_tests, 0,
            "13-bit retargets build the tuned program"
        );
    }
    Some((cache, guard))
}

#[test]
fn the_engine_matches_the_oracle_with_the_13_bit_scheme_forced() {
    let Some((cache, _cleanup)) = run_with_scheme(
        bigmont_cli::tune::SchemeChoice::Only(bigmont_core::scheme::LimbScheme::Limb13),
        "scheme13",
    ) else {
        return;
    };
    let text = std::fs::read_to_string(&cache).unwrap();
    assert!(
        text.lines()
            .all(|l| l.contains("|only-13|") && l.contains("|unrolled ")),
        "{text}"
    );
}

#[test]
fn the_engine_matches_the_oracle_with_the_32_bit_scheme_forced() {
    let Some((cache, _cleanup)) = run_with_scheme(
        bigmont_cli::tune::SchemeChoice::Only(bigmont_core::scheme::LimbScheme::Limb32),
        "scheme32",
    ) else {
        return;
    };
    let text = std::fs::read_to_string(&cache).unwrap();
    assert!(
        text.lines()
            .all(|l| l.contains("|only-32|") && l.contains("|unrolled32 ")),
        "{text}"
    );
}

#[test]
fn the_engine_matches_the_oracle_with_the_scheme_chosen_by_measurement() {
    let Some((cache, _cleanup)) =
        run_with_scheme(bigmont_cli::tune::SchemeChoice::Auto, "schemeauto")
    else {
        return;
    };
    let text = std::fs::read_to_string(&cache).unwrap();
    assert!(
        text.lines()
            .all(|l| l.contains("|any|") && !l.contains("only-")),
        "Auto uses the `any` key, not a forced or legacy one: {text}"
    );
}

// ---- a 32-bit kernel's limb count can shrink inside one 13-bit tuning class ----

/// The class representative (what the tuner self-tests) and the exact length the engine retargets to.
const CLASS_REP: u32 = 256; // tune::class_bits(20) = 13 * 20 - 4
const SHORTER: u32 = 252; // same class (20 13-bit limbs), 8 32-bit limbs where 256 bits needs 9

#[test]
fn bit_length_252_shares_a_tuning_class_with_256_but_needs_fewer_32_bit_limbs() {
    use bigmont_cli::tune;
    use bigmont_core::kernel::{KernelConfig, Variant};
    use bigmont_core::scheme::LimbScheme::{Limb13, Limb32};
    assert_eq!(tune::limb_class(SHORTER), 20);
    assert_eq!(tune::limb_class(CLASS_REP), 20);
    assert_eq!(tune::class_bits(20), CLASS_REP);
    // 13-bit: the same limb count across the class, whatever the tpi.
    // 32-bit: tpi 4 and 8 build a smaller kernel at 252 bits than the one self-tested at 256; tpi 16
    // and 32 round both up to the same size.
    let mut shrinks = Vec::new();
    for tpi in [4u32, 8, 16, 32] {
        let rep13 = KernelConfig::for_bits_scheme(Limb13, CLASS_REP, tpi, 64, Variant::Unrolled);
        assert_eq!(rep13.retarget(SHORTER), rep13, "13-bit tpi {tpi}");
        let rep32 = KernelConfig::for_bits_scheme(Limb32, CLASS_REP, tpi, 64, Variant::Unrolled);
        let at = rep32.retarget(SHORTER);
        eprintln!(
            "32-bit tpi {tpi}: {} limbs at {CLASS_REP} bits (self-tested), {} limbs at {SHORTER} bits (engine)",
            rep32.limbs, at.limbs
        );
        assert!(at.limbs <= rep32.limbs);
        assert!(
            at.r_bits() >= SHORTER + 4,
            "R > 16n at {SHORTER} bits: {at:?}"
        );
        if at.limbs < rep32.limbs {
            shrinks.push((tpi, rep32.limbs, at.limbs));
        }
    }
    assert_eq!(shrinks, [(4, 12, 8), (8, 16, 8)]);
}

#[test]
fn a_retargeted_32_bit_kernel_that_is_smaller_than_the_self_tested_one_gives_oracle_verdicts() {
    // What the engine does after tuning, with the shape fixed to the two tpis whose limb count shrinks
    // (so it does not depend on which 32-bit shape wins the timing): self-test at the class
    // representative, then run the retargeted kernel on 252-bit numbers.
    use bigmont_cli::tune;
    use bigmont_core::gpu::Gpu;
    use bigmont_core::kernel::{KernelConfig, Variant};
    use bigmont_core::runner::Runner;
    use bigmont_core::scheme::LimbScheme::Limb32;
    let gpu = match Gpu::new() {
        Ok(g) => g,
        Err(e) => {
            assert!(
                std::env::var("BIGMONT_ALLOW_SKIP").is_ok_and(|v| v == "1"),
                "no adapter: {e} (set BIGMONT_ALLOW_SKIP=1 to allow skips)"
            );
            eprintln!("SKIP no adapter: {e}");
            return;
        }
    };
    if !gpu.has_int64 || !gpu.has_subgroups {
        assert!(
            std::env::var("BIGMONT_ALLOW_SKIP").is_ok_and(|v| v == "1"),
            "the adapter lacks SHADER_INT64 or subgroups (set BIGMONT_ALLOW_SKIP=1 to allow skips)"
        );
        eprintln!("SKIP the adapter lacks SHADER_INT64 or subgroups");
        return;
    }
    let ns = bucket(SHORTER, 20, 4, 4242);
    let want: Vec<bool> = ns.iter().map(fermat_base2).collect();
    assert!(want.iter().any(|&w| w) && want.iter().any(|&w| !w));
    for tpi in [4u32, 8] {
        if gpu.check_subgroups(tpi).is_err() {
            eprintln!("SKIP tpi {tpi}: subgroup size does not fit");
            continue;
        }
        let tuned = KernelConfig::for_bits_scheme(Limb32, CLASS_REP, tpi, 64, Variant::Unrolled);
        tune::self_test(&gpu, tuned, CLASS_REP).unwrap_or_else(|e| panic!("{e}"));
        let exact = tuned.retarget(SHORTER);
        assert!(exact.limbs < tuned.limbs, "{tuned:?} -> {exact:?}");
        let mut runner = Runner::new(&gpu, exact, SHORTER, 32).expect("runner at the exact length");
        let got = runner.fermat(&gpu, &ns).expect("fermat");
        assert_eq!(
            got, want,
            "tpi {tpi}: {} limbs at {SHORTER} bits (self-tested with {})",
            exact.limbs, tuned.limbs
        );
    }
}

fn gec_opts() -> Options {
    Options {
        backend: Backend::Gpu,
        gec: true,
        min_gpu_batch: 1,
        capacity: 32,
        verify_rate: 0,
        ..Options::default()
    }
}

#[test]
fn gec_verdicts_equal_plain_verdicts_on_clean_input_and_never_alarm() {
    if !common::usable_gpu() {
        return;
    }
    let mut rng = XorShift64::new(21);
    let mut ns: Vec<BigUint> = (0..40)
        .map(|_| random_odd_modulus(&mut rng, 1019))
        .collect();
    ns.extend((0..2).map(|_| random_prp(&mut rng, 1019)));
    let want: Vec<u8> = ns.iter().map(|n| u8::from(fermat_base2(n))).collect();
    assert!(want.contains(&1), "the batch must contain a probable prime");
    let mut checked = Engine::open(gec_opts());
    assert_eq!(checked.verdicts(&ns).unwrap(), want);
    assert_eq!(checked.report().gec_alarms, 0);
    let mut plain = Engine::open(Options {
        gec: false,
        ..gec_opts()
    });
    assert_eq!(plain.verdicts(&ns).unwrap(), want);
}

#[test]
fn an_injected_fault_gives_verdict_2_for_that_candidate_only() {
    if !common::usable_gpu() {
        return;
    }
    // The Beacon harness drill: 3 candidates of 2029 bits, fault in candidate 1 at block 3.
    let mut rng = XorShift64::new(22);
    let ns: Vec<BigUint> = (0..3).map(|_| random_odd_modulus(&mut rng, 2029)).collect();
    let want: Vec<u8> = ns.iter().map(|n| u8::from(fermat_base2(n))).collect();
    let mut e = Engine::open(gec_opts());
    e.inject_fault(1, 3);
    assert_eq!(e.verdicts(&ns).unwrap(), vec![want[0], 2, want[2]]);
    assert_eq!(e.report().gec_alarms, 1);
    // The hook is one-shot: the next call is clean.
    assert_eq!(e.verdicts(&ns).unwrap(), want);
    assert_eq!(e.report().gec_alarms, 1);
}
