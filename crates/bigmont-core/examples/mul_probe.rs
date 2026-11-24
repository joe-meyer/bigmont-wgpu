//! Instruction-throughput probe: how fast can each GPU issue the operations a big-integer kernel
//! could be built from? Each kernel runs 8 independent dependent chains per lane, so throughput
//! (not latency) is measured. Adapted from the equivalent probe in the gapsieve project.
//!
//!   BIGMONT_ADAPTER=4070 BIGMONT_BACKEND=vulkan cargo run --release -p bigmont-core --example mul_probe
//!
//! Output: billions of chain steps per second per instruction type, and what that implies as an
//! upper bound on squarings/s for each arithmetic design (ops per squaring / measured rate).
//! These are ceilings from a microbenchmark, not predictions of a finished kernel.

use std::time::Instant;

const ITERS: u32 = 1 << 14;
const THREADS: u32 = 1 << 20;
const CHAINS: u32 = 8;

/// (name, needs SHADER_INT64, body with `X` replaced by the chain number). `x`N are u32 chain
/// variables, `a`N are u64 chain variables, `c`/`c64`/`i` are loop-invariant inputs.
const KERNELS: [(&str, bool, &str); 6] = [
    ("add only (baseline)", false, "xX = xX + i;"),
    ("mad24 (13-bit limbs: 24x24+32)", false, "xX = (xX & 0xffffffu) * (c & 0xffffffu) + i;"),
    ("mul_lo_u32 (32x32 low, +add)", false, "xX = xX * c + i;"),
    ("u64 mul-add (32x32->64 + 64)", true, "aX = u64(u32(aX)) * c64 + (aX >> 32u) + u64(i);"),
    ("dot4U8Packed (4 8-bit MACs + add)", false, "xX = dot4U8Packed(xX, c) + i;"),
    ("fma_f32", false, "fX = fma(fX, cf, fi);"),
];

fn shader(body: &str, int64: bool) -> String {
    let chains = |f: &dyn Fn(u32) -> String| (0..CHAINS).map(f).collect::<Vec<_>>().join("\n        ");
    let init = chains(&|k| format!("var x{k} = id.x * {CHAINS}u + {k}u; var f{k} = f32(x{k}); {}", if int64 { format!("var a{k} = u64(x{k}) * 0x100000001lu;") } else { String::new() }));
    let step = chains(&|k| body.replace('X', &k.to_string()));
    let fold = (0..CHAINS)
        .map(|k| format!("x{k} ^ bitcast<u32>(f{k}){}", if int64 { format!(" ^ u32(a{k}) ^ u32(a{k} >> 32u)") } else { String::new() }))
        .collect::<Vec<_>>()
        .join(" ^ ");
    let c64 = if int64 { "let c64 = u64(c);" } else { "" };
    format!(
        "@group(0) @binding(0) var<storage, read_write> buf: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
    let c = buf[0] | 0x8000001u;
    let cf = bitcast<f32>(buf[1]);
    {c64}
    {init}
    for (var i = 0u; i < {ITERS}u; i++) {{
        let fi = f32(i);
        {step}
    }}
    let s = {fold};
    if s == 0x12345678u {{
        buf[2u + id.x % 1024u] = s;
    }}
}}"
    )
}

fn main() {
    pollster::block_on(run());
}

async fn run() {
    let instance = wgpu::Instance::default();
    let name = std::env::var("BIGMONT_ADAPTER").ok().map(|s| s.to_lowercase());
    let backend = std::env::var("BIGMONT_BACKEND").ok().map(|s| s.to_lowercase());
    let adapter = if name.is_some() || backend.is_some() {
        instance
            .enumerate_adapters(wgpu::Backends::all())
            .await
            .into_iter()
            .find(|a| {
                let i = a.get_info();
                name.as_ref().is_none_or(|w| i.name.to_lowercase().contains(w)) && backend.as_ref().is_none_or(|w| format!("{:?}", i.backend).to_lowercase().contains(w))
            })
            .expect("no adapter matches BIGMONT_ADAPTER / BIGMONT_BACKEND")
    } else {
        instance
            .request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, ..Default::default() })
            .await
            .expect("adapter")
    };
    let info = adapter.get_info();
    let has64 = adapter.features().contains(wgpu::Features::SHADER_INT64);
    println!("{} [{:?}] SHADER_INT64: {has64}", info.name, info.backend);
    let features = if has64 { wgpu::Features::SHADER_INT64 } else { wgpu::Features::empty() };
    let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor { required_features: features, ..Default::default() }).await.expect("device");
    let buf = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: 4 * 1100, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    queue.write_buffer(&buf, 0, bytemuck::cast_slice(&[0x9e37_79b9u32, 1.000_001f32.to_bits()]));
    let ops = f64::from(THREADS) * f64::from(ITERS) * f64::from(CHAINS);
    println!("{:<36} {:>9} {:>16}", "instruction", "ms", "G chain-steps/s");
    let mut rates: Vec<(&str, f64)> = Vec::new();
    for (name, int64, body) in KERNELS {
        if int64 && !has64 {
            println!("{name:<36} skipped (no SHADER_INT64)");
            continue;
        }
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(shader(body, int64).into()) });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: None, module: &module, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
        if let Some(e) = scope.pop().await {
            println!("{name:<36} rejected: {e}");
            continue;
        }
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &pipeline.get_bind_group_layout(0), entries: &[wgpu::BindGroupEntry { binding: 0, resource: buf.as_entire_binding() }] });
        let mut best = f64::MAX;
        for _ in 0..5 {
            let mut enc = device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(THREADS / 256, 1, 1);
            }
            let t = Instant::now();
            queue.submit([enc.finish()]);
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            best = best.min(t.elapsed().as_secs_f64());
        }
        let rate = ops / best / 1e9;
        println!("{name:<36} {:>9.2} {:>16.0}", best * 1e3, rate);
        rates.push((name, rate));
    }
    // Ceilings: how many squarings/s the measured rate of each instruction would allow if a kernel
    // spent ALL its time issuing that instruction at the counts below (no shuffles, no overhead).
    println!();
    println!("ceiling squarings/s = measured chain-steps/s / instruction count per squaring (2,029-bit; upper bound, not a prediction)");
    let get = |n: &str| rates.iter().find(|(k, _)| k.starts_with(n)).map(|r| r.1);
    if let Some(r) = get("mad24") {
        // 13-bit limbs, 160 limbs: 2 x 160^2 multiply-adds.
        println!("  13-bit limbs, mad24:            {:>6.0} M sq/s   (51,200 ops)", r * 1e9 / 51_200.0 / 1e6);
    }
    if let Some(r) = get("u64 mul-add") {
        // 32-bit limbs, 64 limbs: 2 x 64^2 limb products, each a u64 mul-add; carry handling roughly doubles the op count (shift + add), assumed 2.0 ops per product.
        println!("  32-bit limbs, u64 mul-add (x2): {:>6.0} M sq/s   (8,192 products x 2 ops = 16,384 ops)", r * 1e9 / 16_384.0 / 1e6);
    }
    if let Some(r) = get("dot4U8Packed") {
        // 8-bit digits, 256 digits: each product needs 256^2 digit MACs / 4 per dot4; two products per Montgomery squaring.
        println!("  8-bit digits, dot4 (x1.0):      {:>6.0} M sq/s   (2 x 16,384 = 32,768 ops, no packing overhead)", r * 1e9 / 32_768.0 / 1e6);
    }
}
