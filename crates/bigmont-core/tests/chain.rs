use bigmont_core::gpu::{Batch, Gpu, run};
use bigmont_core::kernel::KernelConfig;
use bigmont_oracle::*;
use num_bigint::BigUint;
use num_traits::One;

/// True when env `BIGMONT_REQUIRE_GPU=1`: a missing adapter or unsupported configuration is then a
/// failure instead of a skip, so a skip cannot masquerade as a pass.
fn require_gpu() -> bool {
    std::env::var("BIGMONT_REQUIRE_GPU").is_ok_and(|v| v == "1")
}

/// Skips (loudly) or, under `BIGMONT_REQUIRE_GPU=1`, panics.
fn skip(msg: String) {
    if require_gpu() {
        panic!("{msg} (BIGMONT_REQUIRE_GPU=1)");
    }
    eprintln!("SKIP {msg}");
}

/// Runs `cfg` on the selected adapter and compares every instance with the oracle.
/// Skips (loudly) only when there is no adapter or the adapter cannot run this configuration,
/// and never when `BIGMONT_REQUIRE_GPU=1`.
fn check(cfg: KernelConfig) {
    let gpu = match Gpu::new() {
        Ok(g) => g,
        Err(e) => return skip(format!("{cfg:?}: no adapter: {e}")),
    };
    if cfg.needs_subgroups() {
        if let Err(e) = gpu.check_subgroups(cfg.tpi) {
            return skip(format!("{cfg:?}: {e}"));
        }
    }
    eprintln!("adapter: {} ({:?}), config {cfg:?}", gpu.info.name, gpu.info.backend);

    // 131 random + 2 special = 133 instances: spans many workgroups (wid.x > 0) and leaves a
    // partial last workgroup for every configuration used here.
    let mut values = Batch::random_values(131, 2029, 7);
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

#[test]
fn tpi8_matches_oracle() {
    check(KernelConfig { tpi: 8, wg: 64 });
}

#[test]
fn tpi16_matches_oracle() {
    check(KernelConfig { tpi: 16, wg: 64 });
}

#[test]
fn tpi32_matches_oracle() {
    check(KernelConfig { tpi: 32, wg: 64 });
}

#[test]
fn tpi32_other_workgroup_sizes_match_oracle() {
    check(KernelConfig { tpi: 32, wg: 128 });
    check(KernelConfig { tpi: 32, wg: 256 });
}
