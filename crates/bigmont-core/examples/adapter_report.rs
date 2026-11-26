//! Prints every adapter wgpu can see and what the selected one supports, so a test run's log says
//! which GPU paths it could cover (CI prints this next to the skip summary). It never fails: a
//! machine with no adapter prints "none".
//!
//!   cargo run -p bigmont-core --example adapter_report
//!   BIGMONT_ADAPTER=llvmpipe BIGMONT_BACKEND=Vulkan cargo run -p bigmont-core --example adapter_report

use bigmont_core::gpu::Gpu;

fn main() {
    println!("adapters wgpu can see:");
    let adapters = pollster::block_on(Gpu::list_adapters_async());
    if adapters.is_empty() {
        println!("  none");
    }
    for a in adapters {
        println!("  {a}");
    }
    match Gpu::new() {
        Ok(g) => {
            println!("selected: {}", g.describe());
            println!(
                "  SUBGROUP: {}  SHADER_INT64: {}  subgroup size: {}..{}",
                g.has_subgroups, g.has_int64, g.info.subgroup_min_size, g.info.subgroup_max_size
            );
            println!(
                "  tpi that fit the subgroups: {:?}",
                [4u32, 8, 16, 32]
                    .into_iter()
                    .filter(|&t| g.check_subgroups(t).is_ok())
                    .collect::<Vec<_>>()
            );
        }
        Err(e) => println!("selected: none ({e})"),
    }
}
