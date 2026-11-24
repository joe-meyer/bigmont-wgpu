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
            assert_eq!(browser[PRELUDE.len()..].trim_start(), native.trim_start(), "{cfg:?}");
        }
    }
}

#[test]
fn wgsl_is_native() {
    let cfg = KernelConfig::new(8, 64, Variant::Unrolled);
    assert_eq!(cfg.wgsl(), cfg.wgsl_for(Target::Native));
}
