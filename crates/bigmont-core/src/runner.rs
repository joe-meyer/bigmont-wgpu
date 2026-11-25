//! Reusable Fermat-test runner: shader, pipeline and buffers are built once per (config, bits).
//!
//! Each call to [`Runner::fermat`] prepares the Montgomery inputs on the host (`R mod n`, the start
//! value `2R mod n`, `-n^-1 mod 2^limb_bits`, and the exponent `n - 1`), uploads them into the persistent
//! buffers and runs the `bits - 1` ladder iterations as a sequence of short dispatches ("segments"),
//! so no single submission runs long enough to trip a driver watchdog (Windows TDR).

use crate::gpu::{Gpu, guarded};
use crate::kernel::{KernelConfig, Target, Variant};
use crate::scheme::LimbScheme;
use num_bigint::BigUint;
use num_traits::One;
use std::time::Instant;

#[derive(Clone, Copy, Debug, Default)]
pub struct RunStats {
    /// Wall time of the ladder dispatches (submit to completion).
    pub gpu_secs: f64,
    /// Host-side Montgomery preparation (`R mod n`, `2R mod n`, packing).
    pub host_prep_secs: f64,
    /// Candidates tested.
    pub tests: u64,
}

/// The smallest bit length a [`Runner`] accepts: 2, the length of n = 3, the smallest odd number
/// the Fermat test applies to. Every length from here up is supported (the kernel arithmetic needs
/// only R = 2^(limb_bits * limbs) > 16n, which `KernelConfig::for_bits` guarantees at any size).
pub const MIN_BITS: u32 = 2;

pub struct Runner {
    cfg: KernelConfig,
    bits: u32,
    cap: u32,
    words: u32,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    params: wgpu::Buffer,
    fp: wgpu::Buffer,
    n_buf: wgpu::Buffer,
    ninv_buf: wgpu::Buffer,
    x_buf: wgpu::Buffer,
    e_buf: wgpu::Buffer,
    staging: wgpu::Buffer,
    /// Iterations per dispatch; `None` adapts so one dispatch takes about `target_secs`.
    pub segment_override: Option<u32>,
    pub target_secs: f64,
    pub stats: RunStats,
}

/// Host-side inputs for one candidate.
struct Prep {
    n: Vec<u32>,
    x0: Vec<u32>,
    ninv: u32,
    e: Vec<u32>,
    /// `R mod n`: the Montgomery form of 1, which the ladder result must equal for a pass.
    r: BigUint,
}

impl Runner {
    /// Builds the Fermat shader, pipeline and buffers for `bits`-bit candidates, `bits >= MIN_BITS`
    /// (2). `capacity` (the most candidates per dispatch) is clamped to the dispatch-size and
    /// storage-binding limits.
    pub fn new(gpu: &Gpu, cfg: KernelConfig, bits: u32, capacity: u32) -> Result<Runner, String> {
        cfg.validate()?;
        if cfg.variant != Variant::Unrolled {
            return Err("the Fermat runner needs the unrolled variant".into());
        }
        // The ladder runs `bits - 1` iterations and reads exponent bit `bits - 2 - iteration`, which
        // needs bits >= 2; below that there is no odd n >= 3 anyway.
        if bits < MIN_BITS {
            return Err(format!("{bits} bits is below the {MIN_BITS}-bit minimum (the smallest odd candidate, n = 3, has {MIN_BITS} bits)"));
        }
        if cfg.r_bits() < bits + 4 {
            return Err(format!("{} limbs give R = 2^{}, which is not above 16n for {bits}-bit n (use KernelConfig::for_bits)", cfg.limbs, cfg.r_bits()));
        }
        if capacity == 0 {
            return Err("capacity must be at least 1".into());
        }
        cfg.check_caps(gpu.caps())?;
        gpu.check_subgroups(cfg.tpi)?;
        gpu.check_health()?;
        Self::build(gpu, cfg, bits, capacity).map_err(|e| gpu.explain(e))
    }

    fn build(gpu: &Gpu, cfg: KernelConfig, bits: u32, capacity: u32) -> Result<Runner, String> {
        // Generate the source first: a configuration with no Fermat source is an `Err`, never a panic.
        let source = cfg.try_fermat_wgsl_for(Target::Native)?;
        let words = bits.div_ceil(32);
        let limits = gpu.device.limits();
        let max_groups = u64::from(limits.max_compute_workgroups_per_dimension.min(65535));
        let by_groups = max_groups * u64::from(cfg.instances_per_workgroup());
        let by_limbs = limits.max_storage_buffer_binding_size / (u64::from(cfg.limbs) * 4);
        let by_words = limits.max_storage_buffer_binding_size / (u64::from(words) * 4);
        let cap = u64::from(capacity).min(by_groups).min(by_limbs).min(by_words);
        let cap = u32::try_from(cap).map_err(|_| "capacity overflow".to_string())?;
        if cap == 0 {
            return Err("the device's storage binding limit is too small for one candidate".into());
        }

        let device = &gpu.device;
        let limb_bytes = u64::from(cap) * u64::from(cfg.limbs) * 4;
        let (params, fp, n_buf, ninv_buf, x_buf, e_buf, staging) = guarded(device, "buffer creation", || {
            let storage = |label: &str, size: u64, extra: wgpu::BufferUsages| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(label),
                    size,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | extra,
                    mapped_at_creation: false,
                })
            };
            let uniform = |label: &str| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(label),
                    size: 16,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            };
            let params = uniform("params");
            let fp = uniform("fp");
            let n_buf = storage("n", limb_bytes, wgpu::BufferUsages::empty());
            let ninv_buf = storage("ninv", u64::from(cap) * 4, wgpu::BufferUsages::empty());
            let x_buf = storage("x", limb_bytes, wgpu::BufferUsages::COPY_SRC);
            let e_buf = storage("e", u64::from(cap) * u64::from(words) * 4, wgpu::BufferUsages::empty());
            let staging = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("staging"),
                size: limb_bytes,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            (params, fp, n_buf, ninv_buf, x_buf, e_buf, staging)
        })?;

        let module = guarded(device, &format!("Fermat shader module for {cfg:?}"), || {
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("fermat"),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            })
        })?;
        let pipeline = guarded(device, &format!("Fermat compute pipeline for {cfg:?}"), || {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("fermat"),
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            })
        })?;
        let bind_group = guarded(device, "bind group creation", || {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("fermat"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: n_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: ninv_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: x_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: e_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: fp.as_entire_binding() },
                ],
            })
        })?;
        Ok(Runner {
            cfg,
            bits,
            cap,
            words,
            pipeline,
            bind_group,
            params,
            fp,
            n_buf,
            ninv_buf,
            x_buf,
            e_buf,
            staging,
            segment_override: None,
            target_secs: 0.1,
            stats: RunStats::default(),
        })
    }

    /// Most candidates per dispatch (after clamping to the device limits).
    pub fn capacity(&self) -> u32 {
        self.cap
    }

    /// Base-2 Fermat verdicts for `ns` (each odd with exactly `bits` bits), in order. Any number of
    /// candidates; they are processed in chunks of `capacity()`. A lost device, a timed-out wait
    /// or an uncaptured wgpu error is an `Err` (see `Gpu::check_health`); the runner's buffers are
    /// then in an unknown state and it should be dropped.
    pub fn fermat(&mut self, gpu: &Gpu, ns: &[BigUint]) -> Result<Vec<bool>, String> {
        gpu.check_health()?;
        let mut out = Vec::with_capacity(ns.len());
        for (k, chunk) in ns.chunks(self.cap as usize).enumerate() {
            out.extend(self.fermat_chunk(gpu, chunk, k * self.cap as usize).map_err(|e| gpu.explain(e))?);
        }
        Ok(out)
    }

    /// One dispatch-sized chunk; `base` is the index of `ns[0]` in the caller's slice (for errors).
    fn fermat_chunk(&mut self, gpu: &Gpu, ns: &[BigUint], base: usize) -> Result<Vec<bool>, String> {
        let bits = self.bits;
        for (i, n) in ns.iter().enumerate() {
            if !n.bit(0) || n.bits() != u64::from(bits) {
                return Err(format!("candidate {} must be odd with exactly {bits} bits (has {} bits, odd = {})", base + i, n.bits(), n.bit(0)));
            }
        }
        let count = ns.len() as u32;
        if count == 0 {
            return Ok(Vec::new());
        }
        let limbs = self.cfg.limbs as usize;
        let words = self.words as usize;

        let t = Instant::now();
        let preps = prepare(ns, self.cfg.scheme, limbs, words);
        let mut n_all = Vec::with_capacity(ns.len() * limbs);
        let mut x_all = Vec::with_capacity(ns.len() * limbs);
        let mut ninv_all = Vec::with_capacity(ns.len());
        let mut e_all = Vec::with_capacity(ns.len() * words);
        for p in &preps {
            n_all.extend_from_slice(&p.n);
            x_all.extend_from_slice(&p.x0);
            ninv_all.push(p.ninv);
            e_all.extend_from_slice(&p.e);
        }
        self.stats.host_prep_secs += t.elapsed().as_secs_f64();

        let (device, queue) = (&gpu.device, &gpu.queue);
        queue.write_buffer(&self.n_buf, 0, bytemuck::cast_slice(&n_all));
        queue.write_buffer(&self.x_buf, 0, bytemuck::cast_slice(&x_all));
        queue.write_buffer(&self.ninv_buf, 0, bytemuck::cast_slice(&ninv_all));
        queue.write_buffer(&self.e_buf, 0, bytemuck::cast_slice(&e_all));

        let groups = count.div_ceil(self.cfg.instances_per_workgroup());
        let total = bits - 1;
        let mut done = 0u32;
        // Start with one iteration per dispatch: the first timing is cheap even on a CPU adapter.
        let mut seg = self.segment_override.unwrap_or(1).max(1);
        while done < total {
            let this = seg.min(total - done);
            queue.write_buffer(&self.params, 0, bytemuck::cast_slice(&[count, this, 0u32, 0u32]));
            queue.write_buffer(&self.fp, 0, bytemuck::cast_slice(&[done, bits, self.words, 0u32]));
            let mut enc = device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &self.bind_group, &[]);
                pass.dispatch_workgroups(groups, 1, 1);
            }
            let t = Instant::now();
            queue.submit([enc.finish()]);
            gpu.wait()?;
            let dt = t.elapsed().as_secs_f64();
            self.stats.gpu_secs += dt;
            done += this;
            if self.segment_override.is_none() {
                let per_iter = dt / f64::from(this);
                // Grow by at most 2x per dispatch so one slow measurement cannot overshoot far.
                seg = ((self.target_secs / per_iter) as u32).clamp(1, seg.saturating_mul(2).max(2));
            }
        }

        let bytes = u64::from(count) * limbs as u64 * 4;
        let mut enc = device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&self.x_buf, 0, &self.staging, 0, bytes);
        queue.submit([enc.finish()]);
        let x = gpu.read_back(&self.staging, bytes)?;

        let scheme = self.cfg.scheme;
        if !scheme.normalised(&x) {
            return Err("kernel returned an unnormalised limb".into());
        }
        let verdicts = ns.iter().zip(&preps).enumerate().map(|(i, (n, p))| scheme.unpack(&x[i * limbs..(i + 1) * limbs]) % n == p.r).collect();
        self.stats.tests += u64::from(count);
        Ok(verdicts)
    }
}

/// Montgomery inputs for every candidate, split across the available cores in contiguous slices
/// (results stay in input order).
fn prepare(ns: &[BigUint], scheme: LimbScheme, limbs: usize, words: usize) -> Vec<Prep> {
    let r_pow = BigUint::one() << scheme.r_bits(limbs as u32);
    let one = |n: &BigUint| {
        let r = &r_pow % n;
        let x0 = (&r << 1u32) % n;
        let mut e = (n - 1u32).to_u32_digits();
        e.resize(words, 0);
        Prep { n: scheme.pack(n, limbs), x0: scheme.pack(&x0, limbs), ninv: scheme.neg_inv(n), e, r }
    };
    let threads = std::thread::available_parallelism().map_or(1, |t| t.get()).min(ns.len()).max(1);
    if threads == 1 {
        return ns.iter().map(one).collect();
    }
    let per = ns.len().div_ceil(threads);
    std::thread::scope(|s| {
        let handles: Vec<_> = ns.chunks(per).map(|part| s.spawn(|| part.iter().map(one).collect::<Vec<_>>())).collect();
        handles.into_iter().flat_map(|h| h.join().expect("host prep thread panicked")).collect()
    })
}
