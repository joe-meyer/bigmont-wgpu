//! Reports the subgroup size an adapter really uses for a compute pipeline created through wgpu
//! (the same path the kernels use), by writing the `subgroup_size` and `subgroup_invocation_id`
//! built-ins from a workgroup of 64 invocations.
//!
//!   BIGMONT_ADAPTER=6700 BIGMONT_BACKEND=vulkan cargo run --release -p bigmont-core --example subgroup_probe

use bigmont_core::gpu::Gpu;

const SRC: &str = "
@group(0) @binding(0) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(subgroup_size) sgs: u32, @builtin(subgroup_invocation_id) sgi: u32) {
    out[gid.x * 2u] = sgs;
    out[gid.x * 2u + 1u] = sgi;
}
";

fn main() {
    let gpu = Gpu::new().expect("no usable adapter");
    println!("adapter: {}", gpu.describe());
    if !gpu.has_subgroups {
        println!("no subgroup feature");
        return;
    }
    let (device, queue) = (&gpu.device, &gpu.queue);
    let invocations = 64u32 * 256;
    let size = u64::from(invocations) * 8;
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging = device.create_buffer(&wgpu::BufferDescriptor { label: None, size, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(SRC.into()) });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[wgpu::BindGroupEntry { binding: 0, resource: buf.as_entire_binding() }],
    });
    let mut enc = device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(256, 1, 1);
    }
    enc.copy_buffer_to_buffer(&buf, 0, &staging, 0, size);
    queue.submit([enc.finish()]);
    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map failed"));
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let data: Vec<u32> = bytemuck::cast_slice::<u8, u32>(&slice.get_mapped_range().unwrap()).to_vec();
    let mut sizes = std::collections::BTreeMap::<u32, u32>::new();
    let mut max_id = 0;
    for pair in data.chunks(2) {
        *sizes.entry(pair[0]).or_default() += 1;
        max_id = max_id.max(pair[1]);
    }
    println!("observed subgroup_size values (value: invocations) over {invocations} invocations: {sizes:?}; max subgroup_invocation_id {max_id}");
}
