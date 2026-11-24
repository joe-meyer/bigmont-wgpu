//! Print the generated WGSL for one configuration (used to feed the browser test page in `web/`).
//!   cargo run --release -p bigmont-core --example dump_wgsl -- TPI WG rolled|unrolled > kernel.wgsl
//!   cargo run --release -p bigmont-core --example dump_wgsl -- TPI WG unrolled fermat-browser > fermat.wgsl
//! The `fermat-browser` form prints the Fermat-ladder kernel for 2,029-bit numbers in `Target::Browser`
//! form (with its `enable subgroups;` and `diagnostic(off, ...)` prelude already included).
use bigmont_core::kernel::{KernelConfig, Target, Variant};

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let usage = "usage: dump_wgsl TPI WG rolled|unrolled";
    let tpi: u32 = a.first().and_then(|v| v.parse().ok()).expect(usage);
    let wg: u32 = a.get(1).and_then(|v| v.parse().ok()).expect(usage);
    let variant = match a.get(2).map(String::as_str) {
        Some("rolled") => Variant::Rolled,
        Some("unrolled") => Variant::Unrolled,
        _ => panic!("{usage}"),
    };
    let cfg = KernelConfig::new(tpi, wg, variant);
    cfg.validate().expect("invalid configuration");
    if a.get(3).map(String::as_str) == Some("fermat-browser") {
        let cfg = KernelConfig::for_bits(2029, tpi, wg, variant);
        cfg.validate().expect("invalid configuration");
        print!("{}", cfg.fermat_wgsl_for(Target::Browser));
    } else {
        print!("{}", cfg.wgsl());
    }
}
