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

#[test]
fn cpu_backend_matches_the_oracle_including_degenerate_inputs() {
    let (ns, want) = batch();
    let mut e = Engine::open(opts(Backend::Cpu));
    assert_eq!(e.verdicts(&ns).unwrap(), want);
}

#[test]
fn gpu_backend_matches_the_oracle_and_buckets_by_bit_length() {
    let (ns, want) = batch();
    let mut e = Engine::open(Options { cache: Some(std::env::temp_dir().join(format!("bigmont-engine-test-{}.cache", std::process::id()))), ..opts(Backend::Gpu) });
    eprintln!("{}", e.describe());
    assert_eq!(e.verdicts(&ns).unwrap(), want);
    let r = e.report();
    assert_eq!(r.prp_disagreements, 0);
    assert!(r.gpu_tests > 0, "the 1019-bit bucket has 12 entries, above min_gpu_batch 4: it must run on the GPU");
    assert!(r.cpu_tests > 0, "tiny and tiny-bucket candidates run on the CPU");
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
    assert_eq!(e.report().composite_disagreements, 0);
}
