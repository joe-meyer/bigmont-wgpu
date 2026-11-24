//! Print the generated WGSL for one configuration (used to feed the browser test page in `web/`).
//!   cargo run --release -p bigmont-core --example dump_wgsl -- TPI WG rolled|unrolled > kernel.wgsl
use bigmont_core::kernel::{KernelConfig, Variant};

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
    let cfg = KernelConfig { tpi, wg, variant };
    cfg.validate().expect("invalid configuration");
    print!("{}", cfg.wgsl());
}
