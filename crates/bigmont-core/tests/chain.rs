use bigmont_core::gpu::{Batch, Gpu, run};
use bigmont_core::kernel::KernelConfig;
use bigmont_oracle::*;
use num_bigint::BigUint;
use num_traits::One;

/// Runs `cfg` on the first adapter and compares every instance with the oracle.
/// Skips (loudly) only when there is no adapter or the adapter cannot run this configuration.
fn check(cfg: KernelConfig) {
    let gpu = match Gpu::new() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("SKIP {cfg:?}: no adapter: {e}");
            return;
        }
    };
    if cfg.needs_subgroups() {
        if let Err(e) = gpu.check_subgroups(cfg.tpi) {
            eprintln!("SKIP {cfg:?}: {e}");
            return;
        }
    }
    eprintln!("adapter: {} ({:?}), config {cfg:?}", gpu.info.name, gpu.info.backend);

    let mut values = Batch::random_values(61, 2029, 7);
    // Carry-ripple stress: n = 2^2029 - 1 (every limb all ones) and n = 2^2028 + 1 (long zero runs).
    let all_ones = (BigUint::one() << 2029u32) - 1u32;
    values.push((all_ones.clone(), &all_ones - 1u32));
    let sparse = (BigUint::one() << 2028u32) + 1u32;
    values.push((sparse.clone(), &sparse - 1u32));

    let batch = Batch::from_values(&values);
    let squarings = 3;
    let result = run(&gpu, cfg, &batch, squarings, 1).expect("run failed");
    for (i, (n, x0)) in values.iter().enumerate() {
        let limbs = &result.x[i * LIMBS..(i + 1) * LIMBS];
        assert!(limbs.iter().all(|&l| l <= LIMB_MASK), "instance {i}: limbs not normalised");
        let got = unpack_limbs(limbs);
        assert!(got < n * 2u32, "instance {i}: result not below 2n");
        assert_eq!(&got % n, mont_square_chain(n, x0, squarings), "instance {i}: wrong value");
    }
}

#[test]
fn tpi1_matches_oracle() {
    check(KernelConfig { tpi: 1, wg: 64 });
}
