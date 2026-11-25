use bigmont_core::kernel::{KernelConfig, Target, Variant};
use bigmont_core::scheme::{Caps, LimbScheme};
use bigmont_oracle::*;

#[test]
fn limb13_matches_the_oracle_helpers() {
    let s = LimbScheme::Limb13;
    assert_eq!(s.limb_bits(), 13);
    for bits in [2u32, 3, 9, 48, 64, 130, 254, 1019, 2029, 4096] {
        assert_eq!(
            s.limbs_for_bits(bits) as usize,
            limbs_for_bits(bits),
            "{bits} bits"
        );
        let l = s.limbs_for_bits(bits);
        assert!(s.r_bits(l) >= bits + 4, "R must exceed 16n at {bits} bits");
    }
    let mut rng = XorShift64::new(7);
    for bits in [64u32, 2029] {
        let n = random_odd_modulus(&mut rng, bits);
        let limbs = s.limbs_for_bits(bits) as usize;
        let packed = s.pack(&n, limbs);
        assert_eq!(packed, pack_limbs_n(&n, limbs));
        assert_eq!(s.unpack(&packed), n);
        assert!(s.normalised(&packed));
        assert_eq!(s.neg_inv(&n), neg_inv_limb(&n));
    }
    assert!(!s.normalised(&[1 << 13]));
}

#[test]
fn caps_missing_and_satisfies() {
    let none = Caps::default();
    let both = Caps {
        subgroups: true,
        int64: true,
    };
    assert!(both.satisfies(none));
    assert!(both.satisfies(both));
    assert!(!none.satisfies(Caps {
        subgroups: true,
        int64: false
    }));
    assert_eq!(none.missing(both), vec!["subgroups", "int64"]);
    assert_eq!(
        (Caps::BROWSER.subgroups, Caps::BROWSER.int64),
        (true, false)
    );
}

#[test]
fn kernel_config_carries_the_scheme_and_retargets() {
    let cfg = KernelConfig::for_bits(2029, 8, 64, Variant::Unrolled);
    assert_eq!(cfg.scheme, LimbScheme::Limb13);
    assert_eq!(
        cfg,
        KernelConfig::for_bits_scheme(LimbScheme::Limb13, 2029, 8, 64, Variant::Unrolled)
    );
    assert_eq!(
        cfg.requires(),
        Caps {
            subgroups: true,
            int64: false
        }
    );
    assert_eq!(cfg.r_bits(), 13 * cfg.limbs);
    let small = cfg.retarget(100);
    assert_eq!(
        (small.scheme, small.tpi, small.wg, small.variant),
        (cfg.scheme, 8, 64, Variant::Unrolled)
    );
    assert_eq!(small, KernelConfig::for_bits(100, 8, 64, Variant::Unrolled));
    assert_eq!(cfg.try_wgsl_for(Target::Native).unwrap(), cfg.wgsl());
    assert_eq!(
        cfg.try_fermat_wgsl_for(Target::Native).unwrap(),
        cfg.fermat_wgsl_for(Target::Native)
    );
}

#[test]
fn batch_from_values_cfg_matches_the_13_bit_batch() {
    use bigmont_core::gpu::Batch;
    let values = Batch::random_values(5, 130, 3);
    let cfg = KernelConfig::for_bits(130, 4, 64, Variant::Unrolled);
    let a = Batch::from_values_cfg(&values, &cfg);
    let b = Batch::from_values_n(&values, cfg.limbs as usize);
    assert_eq!(
        (a.count, a.limbs, &a.n, &a.ninv, &a.x),
        (b.count, b.limbs, &b.n, &b.ninv, &b.x)
    );
}

#[test]
fn limb32_matches_the_oracle_helpers() {
    let s = LimbScheme::Limb32;
    let mut rng = XorShift64::new(11);
    for bits in [64u32, 254, 2029] {
        let n = random_odd_modulus(&mut rng, bits);
        let limbs = s.limbs_for_bits(bits) as usize;
        assert!(
            s.r_bits(limbs as u32) >= bits + 4,
            "R must exceed 16n at {bits} bits"
        );
        let packed = s.pack(&n, limbs);
        assert_eq!(packed, pack_u32_limbs(&n, limbs));
        assert_eq!(s.unpack(&packed), n);
        assert_eq!(s.neg_inv(&n), neg_inv_u32(&n));
        let x = random_below(&mut rng, &n);
        assert_eq!(
            s.chain(&n, &x, 7, limbs),
            mont_square_chain_bits(&n, &x, 7, 32 * limbs as u32)
        );
    }
}

#[test]
fn limb32_configs_validate_only_as_unrolled_cooperative_kernels() {
    let ok = KernelConfig::for_bits_scheme(LimbScheme::Limb32, 2029, 8, 64, Variant::Unrolled);
    assert_eq!(ok.validate(), Ok(()));
    assert_eq!(
        ok.requires(),
        Caps {
            subgroups: true,
            int64: true
        }
    );
    for tpi in [4u32, 16, 32] {
        assert_eq!(
            KernelConfig::for_bits_scheme(LimbScheme::Limb32, 2029, tpi, 64, Variant::Unrolled)
                .validate(),
            Ok(())
        );
    }
    // the rolled variant and the one-thread configuration are 13-bit only
    assert!(
        KernelConfig::for_bits_scheme(LimbScheme::Limb32, 2029, 8, 64, Variant::Rolled)
            .validate()
            .is_err()
    );
    assert!(
        KernelConfig::for_bits_scheme(LimbScheme::Limb32, 2029, 1, 64, Variant::Rolled)
            .validate()
            .is_err()
    );
}

#[test]
fn check_caps_names_every_missing_capability() {
    let c32 = KernelConfig::for_bits_scheme(LimbScheme::Limb32, 2029, 8, 64, Variant::Unrolled);
    let c13 = KernelConfig::for_bits(2029, 8, 64, Variant::Unrolled);
    let both = Caps {
        subgroups: true,
        int64: true,
    };
    assert_eq!(c32.check_caps(both), Ok(()));
    assert_eq!(
        c13.check_caps(Caps {
            subgroups: true,
            int64: false
        }),
        Ok(())
    );
    let e = c32
        .check_caps(Caps {
            subgroups: true,
            int64: false,
        })
        .unwrap_err();
    assert!(e.contains("int64") && !e.contains("subgroups"), "{e}");
    let e = c32.check_caps(Caps::default()).unwrap_err();
    assert!(e.contains("int64") && e.contains("subgroups"), "{e}");
    assert!(
        c13.check_caps(Caps::default())
            .unwrap_err()
            .contains("subgroups")
    );
}
