//! Print the generated WGSL for one configuration (used to feed the browser test pages in `web/`).
//!
//!   cargo run --release -p bigmont-core --example dump_wgsl -- TPI WG rolled|unrolled > kernel.wgsl
//!   cargo run --release -p bigmont-core --example dump_wgsl -- TPI WG unrolled fermat-browser > fermat.wgsl
//!
//! The optional fourth word picks the kernel and target: `fermat-browser` (Fermat ladder, `Target::Browser`,
//! with its `enable subgroups;` and `diagnostic(off, ...)` prelude already included), `fermat-native`,
//! `squarings-browser`, `squarings-native`. Without it the squaring kernel for the Phase 0 size class
//! (native target) is printed, as before. Options (any order after the fourth word):
//!
//!   --bits N      operand size in bits for the `*-browser|native` forms (default 2029)
//!   --scheme 13|32  limb scheme (default 13). The 32-bit scheme needs `unrolled` and is native-only: asking for
//!                 a browser form of it prints the refusal to stderr and exits 1.
use bigmont_core::kernel::{KernelConfig, Target, Variant};
use bigmont_core::scheme::LimbScheme;

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let usage = "usage: dump_wgsl TPI WG rolled|unrolled [fermat-browser|fermat-native|squarings-browser|squarings-native] [--bits N] [--scheme 13|32]";
    let tpi: u32 = a.first().and_then(|v| v.parse().ok()).expect(usage);
    let wg: u32 = a.get(1).and_then(|v| v.parse().ok()).expect(usage);
    let variant = match a.get(2).map(String::as_str) {
        Some("rolled") => Variant::Rolled,
        Some("unrolled") => Variant::Unrolled,
        _ => panic!("{usage}"),
    };
    let flag = |name: &str| {
        a.iter().position(|x| x == name).map(|i| {
            a.get(i + 1)
                .unwrap_or_else(|| panic!("{name} needs a value"))
                .clone()
        })
    };
    let bits: u32 = flag("--bits").map_or(2029, |v| v.parse().expect("--bits N"));
    let scheme = flag("--scheme").map_or(LimbScheme::Limb13, |v| {
        LimbScheme::parse(&v).unwrap_or_else(|| panic!("--scheme must be 13 or 32, got {v}"))
    });
    let form = a
        .get(3)
        .map(String::as_str)
        .filter(|s| !s.starts_with("--"));
    let result = match form {
        None => {
            let cfg = KernelConfig::new(tpi, wg, variant);
            cfg.validate().expect("invalid configuration");
            Ok(cfg.wgsl())
        }
        Some(form) => {
            let (kernel, target) = match form {
                "fermat-browser" => ("fermat", Target::Browser),
                "fermat-native" => ("fermat", Target::Native),
                "squarings-browser" => ("squarings", Target::Browser),
                "squarings-native" => ("squarings", Target::Native),
                other => panic!("unknown form {other}; {usage}"),
            };
            let cfg = KernelConfig::for_bits_scheme(scheme, bits, tpi, wg, variant);
            cfg.validate().expect("invalid configuration");
            if kernel == "fermat" {
                cfg.try_fermat_wgsl_for(target)
            } else {
                cfg.try_wgsl_for(target)
            }
        }
    };
    match result {
        Ok(src) => print!("{src}"),
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    }
}
