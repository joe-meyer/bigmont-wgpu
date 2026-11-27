use bigmont_core::kernel::{KernelConfig, Squaring, Target, Variant};
use bigmont_core::scheme::{Caps, LimbScheme};

const PRELUDE: &str = "enable subgroups;\ndiagnostic(off, subgroup_uniformity);\n";

fn validate(src: &str, caps: Caps) -> Result<(), String> {
    let src = src.strip_prefix(PRELUDE).unwrap_or(src); // naga 30 rejects the browser-only directives
    let module = naga::front::wgsl::parse_str(src).map_err(|e| e.emit_to_string(src))?;
    let mut c = naga::valid::Capabilities::empty();
    if caps.subgroups {
        c |= naga::valid::Capabilities::SUBGROUP;
    }
    if caps.int64 {
        c |= naga::valid::Capabilities::SHADER_INT64;
    }
    naga::valid::Validator::new(naga::valid::ValidationFlags::all(), c)
        .validate(&module)
        .map(|_| ())
        .map_err(|e| format!("{e:?}"))
}

#[test]
fn every_scheme_and_target_that_generates_validates_with_exactly_that_targets_capabilities() {
    let native = Caps {
        subgroups: true,
        int64: true,
    };
    let mut generated = 0;
    for &scheme in LimbScheme::ALL {
        for tpi in [4u32, 8, 16, 32] {
            for bits in [64u32, 254, 1019, 2029] {
                let cfg = KernelConfig::for_bits_scheme(scheme, bits, tpi, 64, Variant::Unrolled);
                for (target, caps) in [(Target::Native, native), (Target::Browser, Caps::BROWSER)] {
                    match cfg.try_wgsl_for(target) {
                        Ok(src) => {
                            validate(&src, caps)
                                .unwrap_or_else(|e| panic!("{cfg:?} {target:?}: {e}"));
                            generated += 1;
                        }
                        Err(_) => assert!(
                            !caps.satisfies(cfg.requires()),
                            "{cfg:?} {target:?} refused although the target can run it"
                        ),
                    }
                }
            }
        }
    }
    assert!(generated > 0);
}

#[test]
fn every_fermat_source_that_generates_validates_with_exactly_that_targets_capabilities() {
    let native = Caps {
        subgroups: true,
        int64: true,
    };
    let mut generated = 0;
    for &scheme in LimbScheme::ALL {
        for tpi in [4u32, 8, 16, 32] {
            for bits in [64u32, 254, 1019, 2029] {
                let cfg = KernelConfig::for_bits_scheme(scheme, bits, tpi, 64, Variant::Unrolled);
                for (target, caps) in [(Target::Native, native), (Target::Browser, Caps::BROWSER)] {
                    match cfg.try_fermat_wgsl_for(target) {
                        Ok(src) => {
                            validate(&src, caps)
                                .unwrap_or_else(|e| panic!("{cfg:?} {target:?} fermat: {e}"));
                            generated += 1;
                        }
                        Err(_) => assert!(
                            !caps.satisfies(cfg.requires()),
                            "{cfg:?} {target:?} fermat refused although the target can run it"
                        ),
                    }
                }
            }
        }
    }
    assert!(generated > 0);
}

#[test]
fn every_multiply_source_that_generates_validates_with_exactly_that_targets_capabilities() {
    let native = Caps {
        subgroups: true,
        int64: true,
    };
    let mut generated = 0;
    for &scheme in LimbScheme::ALL {
        for tpi in [4u32, 8, 16, 32] {
            for bits in [64u32, 254, 1019, 2029] {
                let cfg = KernelConfig::for_bits_scheme(scheme, bits, tpi, 64, Variant::Unrolled);
                for (target, caps) in [(Target::Native, native), (Target::Browser, Caps::BROWSER)] {
                    match cfg.try_mul_wgsl_for(target) {
                        Ok(src) => {
                            assert!(src.contains("b_limbs"), "{cfg:?} {target:?}");
                            validate(&src, caps)
                                .unwrap_or_else(|e| panic!("{cfg:?} {target:?} mul: {e}"));
                            generated += 1;
                        }
                        Err(_) => assert!(
                            !caps.satisfies(cfg.requires()),
                            "{cfg:?} {target:?} mul refused although the target can run it"
                        ),
                    }
                }
            }
        }
    }
    assert!(generated > 0);
}

#[test]
fn a_32_bit_kernel_is_never_generated_for_the_browser() {
    let cfg = KernelConfig::for_bits_scheme(LimbScheme::Limb32, 2029, 8, 64, Variant::Unrolled);
    let e = cfg.try_wgsl_for(Target::Browser).unwrap_err();
    assert!(e.contains("int64"), "{e}");
    assert!(cfg.try_wgsl_for(Target::Native).unwrap().contains("u64"));
    // and its source does need the int64 capability: validating it without that capability fails.
    let src = cfg.try_wgsl_for(Target::Native).unwrap();
    assert!(
        validate(
            &src,
            Caps {
                subgroups: true,
                int64: false
            }
        )
        .is_err()
    );
    // the 13-bit source never mentions u64.
    let c13 = KernelConfig::for_bits(2029, 8, 64, Variant::Unrolled)
        .try_wgsl_for(Target::Browser)
        .unwrap();
    assert!(!c13.contains("u64"));

    // the same for the Fermat ladder.
    let e = cfg.try_fermat_wgsl_for(Target::Browser).unwrap_err();
    assert!(e.contains("int64"), "{e}");
    let fermat = cfg.try_fermat_wgsl_for(Target::Native).unwrap();
    assert!(fermat.contains("u64") && fermat.contains("e_words"));
    assert!(
        validate(
            &fermat,
            Caps {
                subgroups: true,
                int64: false
            }
        )
        .is_err()
    );
    let f13 = KernelConfig::for_bits(2029, 8, 64, Variant::Unrolled)
        .try_fermat_wgsl_for(Target::Browser)
        .unwrap();
    assert!(!f13.contains("u64"));

    // and for the multiply kernel.
    let e = cfg.try_mul_wgsl_for(Target::Browser).unwrap_err();
    assert!(e.contains("int64"), "{e}");
    let mul = cfg.try_mul_wgsl_for(Target::Native).unwrap();
    assert!(mul.contains("u64") && mul.contains("b_limbs"));
    assert!(
        validate(
            &mul,
            Caps {
                subgroups: true,
                int64: false
            }
        )
        .is_err()
    );
    let m13 = KernelConfig::for_bits(2029, 8, 64, Variant::Unrolled)
        .try_mul_wgsl_for(Target::Browser)
        .unwrap();
    assert!(!m13.contains("u64"));
}

#[test]
fn limb32_geometry() {
    let s = LimbScheme::Limb32;
    assert_eq!((s.limb_bits(), s.name()), (32, "32"));
    assert_eq!(s.limbs_for_bits(2029), 64);
    assert_eq!(s.limbs_for_bits(254), 9); // (254 + 4) / 32 = 8.06: R > 16n needs a ninth limb
    assert_eq!(LimbScheme::parse("32"), Some(s));
    assert_eq!(
        s.requires(),
        Caps {
            subgroups: false,
            int64: true
        }
    );
    assert!(s.normalised(&[u32::MAX]));
    assert_eq!(
        KernelConfig::for_bits_scheme(s, 2029, 8, 64, Variant::Unrolled).limbs,
        64
    );
    assert_eq!(
        KernelConfig::for_bits_scheme(s, 254, 8, 64, Variant::Unrolled).limbs,
        16
    ); // 9 rounded up to a multiple of 8
}

#[test]
fn symmetric_squaring_sources_validate_are_native_only_and_leave_multiply_mode_fused() {
    let native = Caps {
        subgroups: true,
        int64: true,
    };
    for tpi in [4u32, 8, 16, 32] {
        for bits in [64u32, 254, 1019, 2029] {
            let cfg =
                KernelConfig::for_bits_scheme(LimbScheme::Limb32, bits, tpi, 64, Variant::Unrolled)
                    .with_squaring(Squaring::Symmetric);
            assert!(cfg.validate().is_ok(), "{cfg:?}");
            let squaring = cfg
                .try_wgsl_for(Target::Native)
                .unwrap_or_else(|e| panic!("{cfg:?}: {e}"));
            assert!(
                squaring.contains("limb32sq.rs"),
                "{cfg:?} used the fused generator"
            );
            validate(&squaring, native).unwrap_or_else(|e| panic!("{cfg:?} squaring: {e}"));
            let fermat = cfg
                .try_fermat_wgsl_for(Target::Native)
                .unwrap_or_else(|e| panic!("{cfg:?}: {e}"));
            assert!(
                fermat.contains("limb32sq.rs"),
                "{cfg:?} used the fused generator"
            );
            validate(&fermat, native).unwrap_or_else(|e| panic!("{cfg:?} fermat: {e}"));
            // Multiply mode (the Gerbicz check) always runs the fused code.
            let mul = cfg
                .try_mul_wgsl_for(Target::Native)
                .unwrap_or_else(|e| panic!("{cfg:?}: {e}"));
            assert!(!mul.contains("limb32sq.rs"), "{cfg:?}");
            // 32-bit kernels, symmetric or not, are never generated for the browser.
            assert!(cfg.try_wgsl_for(Target::Browser).is_err(), "{cfg:?}");
            assert!(cfg.try_fermat_wgsl_for(Target::Browser).is_err(), "{cfg:?}");
        }
    }
}

#[test]
fn symmetric_squaring_is_rejected_for_the_13_bit_scheme() {
    let cfg = KernelConfig::for_bits_scheme(LimbScheme::Limb13, 254, 8, 64, Variant::Unrolled)
        .with_squaring(Squaring::Symmetric);
    assert!(cfg.validate().is_err(), "{cfg:?}");
}
