//! wgpu device setup, input batches, and the kernel runner.

use crate::kernel::KernelConfig;
use bigmont_oracle::{LIMBS, XorShift64, neg_inv_limb, pack_limbs, random_below, random_odd_modulus};
use num_bigint::BigUint;
use std::time::Instant;

pub struct Gpu {
    pub info: wgpu::AdapterInfo,
    pub has_subgroups: bool,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

fn describe(info: &wgpu::AdapterInfo) -> String {
    format!("{} [{:?}, {:?}] subgroup {}..{}", info.name, info.backend, info.device_type, info.subgroup_min_size, info.subgroup_max_size)
}

impl Gpu {
    /// Picks the adapter named by env `BIGMONT_ADAPTER` (substring of the adapter name) and
    /// `BIGMONT_BACKEND` (substring of the backend name, e.g. "Vulkan", "Dx12"); otherwise the
    /// high-performance default. Requests `Features::SUBGROUP` when the adapter has it.
    pub fn new() -> Result<Gpu, String> {
        pollster::block_on(async {
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
                        name.as_ref().is_none_or(|w| i.name.to_lowercase().contains(w))
                            && backend.as_ref().is_none_or(|w| format!("{:?}", i.backend).to_lowercase().contains(w))
                    })
                    .ok_or("no adapter matches BIGMONT_ADAPTER / BIGMONT_BACKEND")?
            } else {
                instance
                    .request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, ..Default::default() })
                    .await
                    .map_err(|e| e.to_string())?
            };
            let info = adapter.get_info();
            let has_subgroups = adapter.features().contains(wgpu::Features::SUBGROUP);
            let mut features = wgpu::Features::empty();
            if has_subgroups {
                features |= wgpu::Features::SUBGROUP;
            }
            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor { required_features: features, ..Default::default() })
                .await
                .map_err(|e| e.to_string())?;
            Ok(Gpu { info, has_subgroups, device, queue })
        })
    }

    pub fn list_adapters() -> Vec<String> {
        pollster::block_on(async {
            let instance = wgpu::Instance::default();
            instance
                .enumerate_adapters(wgpu::Backends::all())
                .await
                .iter()
                .map(|a| {
                    let i = a.get_info();
                    format!("{} subgroups={}", describe(&i), a.features().contains(wgpu::Features::SUBGROUP))
                })
                .collect()
        })
    }

    pub fn describe(&self) -> String {
        describe(&self.info)
    }

    /// Whether `tpi` lanes per number fit this adapter's subgroups.
    pub fn check_subgroups(&self, tpi: u32) -> Result<(), String> {
        if !self.has_subgroups {
            return Err("adapter lacks wgpu::Features::SUBGROUP".into());
        }
        let (lo, hi) = (self.info.subgroup_min_size, self.info.subgroup_max_size);
        if lo < tpi || lo % tpi != 0 || hi % tpi != 0 {
            return Err(format!("subgroup size {lo}..{hi} is not a multiple of tpi {tpi}"));
        }
        Ok(())
    }
}

/// Input arrays, one entry per instance: moduli, -n^-1 mod 2^13, and starting values, as limbs.
pub struct Batch {
    pub count: u32,
    pub n: Vec<u32>,
    pub ninv: Vec<u32>,
    pub x: Vec<u32>,
}

impl Batch {
    pub fn from_values(values: &[(BigUint, BigUint)]) -> Batch {
        let mut b = Batch { count: values.len() as u32, n: Vec::new(), ninv: Vec::new(), x: Vec::new() };
        for (n, x) in values {
            assert!(x < &(n * 2u32), "x must be below 2n");
            b.n.extend(pack_limbs(n));
            b.ninv.push(neg_inv_limb(n));
            b.x.extend(pack_limbs(x));
        }
        b
    }

    /// `count` random (odd modulus of exactly `bits` bits, x below modulus) pairs.
    pub fn random_values(count: u32, bits: u32, seed: u64) -> Vec<(BigUint, BigUint)> {
        let mut rng = XorShift64::new(seed);
        (0..count)
            .map(|_| {
                let n = random_odd_modulus(&mut rng, bits);
                let x = random_below(&mut rng, &n);
                (n, x)
            })
            .collect()
    }
}

pub struct Run {
    pub x: Vec<u32>,
    pub best_secs: f64,
}

/// Uploads `batch`, runs `repeats` dispatches of `squarings` Montgomery squarings each (every
/// dispatch continues from the previous one's output), and reads the result back.
pub fn run(gpu: &Gpu, cfg: KernelConfig, batch: &Batch, squarings: u32, repeats: u32) -> Result<Run, String> {
    cfg.validate()?;
    if cfg.needs_subgroups() {
        gpu.check_subgroups(cfg.tpi)?;
    }
    let (device, queue) = (&gpu.device, &gpu.queue);
    let words = (batch.count as usize * LIMBS) as u64;
    let storage = |label: &str, size: u64, extra: wgpu::BufferUsages| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | extra,
            mapped_at_creation: false,
        })
    };
    let params = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("params"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let n_buf = storage("n", words * 4, wgpu::BufferUsages::empty());
    let ninv_buf = storage("ninv", u64::from(batch.count) * 4, wgpu::BufferUsages::empty());
    let x_buf = storage("x", words * 4, wgpu::BufferUsages::COPY_SRC);
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size: words * 4,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&params, 0, bytemuck::cast_slice(&[batch.count, squarings, 0u32, 0u32]));
    queue.write_buffer(&n_buf, 0, bytemuck::cast_slice(&batch.n));
    queue.write_buffer(&ninv_buf, 0, bytemuck::cast_slice(&batch.ninv));
    queue.write_buffer(&x_buf, 0, bytemuck::cast_slice(&batch.x));

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("mont"), source: wgpu::ShaderSource::Wgsl(cfg.wgsl().into()) });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("mont"),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: n_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: ninv_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: x_buf.as_entire_binding() },
        ],
    });
    let groups = batch.count.div_ceil(cfg.instances_per_workgroup());
    let mut best = f64::MAX;
    for _ in 0..repeats.max(1) {
        let mut enc = device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        let t = Instant::now();
        queue.submit([enc.finish()]);
        device.poll(wgpu::PollType::wait_indefinitely()).map_err(|e| format!("GPU poll failed: {e}"))?;
        best = best.min(t.elapsed().as_secs_f64());
    }
    let mut enc = device.create_command_encoder(&Default::default());
    enc.copy_buffer_to_buffer(&x_buf, 0, &staging, 0, words * 4);
    queue.submit([enc.finish()]);
    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("mapping readback buffer failed"));
    device.poll(wgpu::PollType::wait_indefinitely()).map_err(|e| format!("GPU poll failed: {e}"))?;
    let x = {
        let view = slice.get_mapped_range().expect("readback buffer not mapped");
        bytemuck::cast_slice::<u8, u32>(&view).to_vec()
    };
    staging.unmap();
    Ok(Run { x, best_secs: best })
}
