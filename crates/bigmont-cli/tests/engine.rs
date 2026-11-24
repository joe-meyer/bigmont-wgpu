use bigmont_cli::engine::{Backend, Engine, Options};
use bigmont_oracle::*;
use num_bigint::BigUint;
use num_traits::One;

fn opts(backend: Backend) -> Options {
    Options { backend, min_gpu_batch: 4, capacity: 32, verify_rate: 0, ..Options::default() }
}

fn batch() -> (Vec<BigUint>, Vec<u8>) {
    let mut rng = XorShift64::new(9);
    let mut ns: Vec<BigUint> = (0..10).map(|_| random_odd_modulus(&mut rng, 1019)).collect();
    ns.extend((0..2).map(|_| random_prp(&mut rng, 1019)));
    ns.extend((0..9).map(|_| random_odd_modulus(&mut rng, 777))); // second bit length
    ns.extend([BigUint::from(2u32), BigUint::from(3u32), BigUint::from(4u32), BigUint::one(), BigUint::from(0u32), (BigUint::one() << 1019u32) + 2u32]);
    let want = ns.iter().map(|n| u8::from(fermat_base2(n))).collect();
    (ns, want)
}

/// (PRPs, composites) among the candidates that take the GPU path with `opts(Backend::Gpu)`: odd,
/// at least 64 bits, in a bit-length bucket of at least `min_gpu_batch` (4) entries.
fn gpu_path_counts(ns: &[BigUint], want: &[u8]) -> (u64, u64) {
    let on_gpu = |n: &BigUint| n.bit(0) && n.bits() >= 64 && ns.iter().filter(|m| m.bits() == n.bits() && m.bit(0)).count() >= 4;
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
    let (ns, want) = batch();
    let cache = std::env::temp_dir().join(format!("bigmont-engine-test-{}.cache", std::process::id()));
    let _cleanup = RemoveOnDrop(cache.clone());
    let mut e = Engine::open(Options { cache: Some(cache), ..opts(Backend::Gpu) });
    eprintln!("{}", e.describe());
    assert_eq!(e.verdicts(&ns).unwrap(), want);
    let r = e.report();
    assert_eq!(r.prp_disagreements, 0);
    assert!(r.gpu_tests > 0, "the 1019-bit bucket has 12 entries, above min_gpu_batch 4: it must run on the GPU");
    // Both bit-length buckets (1019 and 777 bits) are above min_gpu_batch, so the only candidate the
    // CPU counts as a test is the 2-bit `3`; 0, 1, 2, 4 and the even number follow the verdict rules.
    assert_eq!(r.cpu_tests, 1, "only the 2-bit candidate 3 is tested on the CPU");
    let (gpu_prps, _) = gpu_path_counts(&ns, &want);
    assert!(gpu_prps >= 2, "the batch holds at least the two planted probable primes");
    assert_eq!(r.gpu_errors, 0);
    assert!(r.rechecks >= gpu_prps, "every GPU PRP must be re-run on the CPU: {} rechecks, {gpu_prps} PRPs", r.rechecks);
}

#[test]
fn auto_falls_back_to_cpu_when_no_gpu_works() {
    // BIGMONT_ADAPTER selects an adapter that does not exist, so GPU open fails; Auto must still answer.
    let (ns, want) = batch();
    let selector = bigmont_core::gpu::AdapterSelector { name: Some("no-such-adapter-xyz".into()), ..Default::default() };
    let mut e = Engine::open(Options { selector, ..opts(Backend::Auto) });
    assert!(e.describe().to_lowercase().contains("cpu"));
    assert_eq!(e.verdicts(&ns).unwrap(), want);
}

#[test]
fn forced_gpu_backend_errors_when_the_adapter_is_missing() {
    let selector = bigmont_core::gpu::AdapterSelector { name: Some("no-such-adapter-xyz".into()), ..Default::default() };
    let mut e = Engine::open(Options { selector, ..opts(Backend::Gpu) });
    let (ns, _) = batch();
    assert!(e.verdicts(&ns).is_err());
}

#[test]
fn verify_rate_rechecks_composites_without_changing_verdicts() {
    let (ns, want) = batch();
    let mut e = Engine::open(Options { verify_rate: 2, ..opts(Backend::Gpu) });
    assert_eq!(e.verdicts(&ns).unwrap(), want);
    let r = e.report();
    assert_eq!(r.composite_disagreements, 0);
    assert_eq!(r.prp_disagreements, 0);
    assert_eq!(r.gpu_errors, 0, "the GPU path must not have fallen back");
    let (prps, composites) = gpu_path_counts(&ns, &want);
    assert_eq!(r.gpu_tests, prps + composites, "every odd >= 64-bit candidate in a big enough bucket runs on the GPU");
    // Every PRP, plus every 2nd GPU composite (counted across the call).
    assert_eq!(r.rechecks, prps + composites / 2);
    assert!(r.rechecks >= 7);
}
