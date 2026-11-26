use bigmont_core::kernel::{KernelConfig, Target, Variant};

const PRELUDE: &str = "enable subgroups;\ndiagnostic(off, subgroup_uniformity);\n";

#[test]
fn browser_source_is_native_source_plus_the_two_directives() {
    for variant in [Variant::Unrolled, Variant::Rolled] {
        for tpi in [4u32, 8, 16, 32] {
            let cfg = KernelConfig::new(tpi, 64, variant);
            let native = cfg.wgsl_for(Target::Native);
            let browser = cfg.wgsl_for(Target::Browser);
            assert!(browser.starts_with(PRELUDE), "{cfg:?}");
            assert!(!native.contains("enable subgroups"), "{cfg:?}");
            // everything after the prelude is identical, except that the rolled template's own
            // blank first line (the empty `__ENABLE__` placeholder) is replaced.
            assert_eq!(
                browser[PRELUDE.len()..].trim_start(),
                native.trim_start(),
                "{cfg:?}"
            );
        }
    }
}

#[test]
fn fermat_browser_source_is_native_source_plus_the_two_directives() {
    for tpi in [4u32, 8, 16, 32] {
        for bits in [64u32, 1019, 2029] {
            let cfg = KernelConfig::for_bits(bits, tpi, 64, Variant::Unrolled);
            let native = cfg.fermat_wgsl_for(Target::Native);
            let browser = cfg.fermat_wgsl_for(Target::Browser);
            assert!(!native.contains("enable subgroups"), "{cfg:?}");
            assert!(native.contains("var<uniform> fp: vec4<u32>"), "{cfg:?}");
            assert_eq!(browser, format!("{PRELUDE}{native}"), "{cfg:?}");
        }
    }
}

#[test]
fn mul_browser_source_is_native_source_plus_the_two_directives() {
    for tpi in [4u32, 8, 16, 32] {
        for bits in [64u32, 1019, 2029] {
            let cfg = KernelConfig::for_bits(bits, tpi, 64, Variant::Unrolled);
            let native = cfg.try_mul_wgsl_for(Target::Native).unwrap();
            let browser = cfg.try_mul_wgsl_for(Target::Browser).unwrap();
            assert!(!native.contains("enable subgroups"), "{cfg:?}");
            assert!(native.contains("var<storage, read> b_limbs"), "{cfg:?}");
            assert_eq!(browser, format!("{PRELUDE}{native}"), "{cfg:?}");
        }
    }
}

#[test]
fn wgsl_is_native() {
    let cfg = KernelConfig::new(8, 64, Variant::Unrolled);
    assert_eq!(cfg.wgsl(), cfg.wgsl_for(Target::Native));
}
