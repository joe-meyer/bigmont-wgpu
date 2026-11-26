//! Reusable Fermat-test runner: shader, pipeline and buffers are built once per (config, bits).
//!
//! Each call to [`Runner::fermat_async`] prepares the Montgomery inputs on the host (`R mod n`, the start
//! value `2R mod n`, `-n^-1 mod 2^limb_bits`, and the exponent `n - 1`), uploads them into the persistent
//! buffers and runs the `bits - 1` ladder iterations as a sequence of short dispatches ("segments"),
//! so no single submission runs long enough to trip a driver watchdog (Windows TDR).
//!
//! [`Runner::fermat_checked_async`] (after [`Runner::enable_gec_async`]) adds the Gerbicz-Li error check: every
//! block of [`GEC_BLOCK`] ladder iterations ends with `A <- A * y_prev`, `B <- B * x`, `y_prev <- x`,
//! and after the last block `A^(2^GEC_BLOCK) * 2^(sum V)` must equal `B` mod n. The iterations after
//! the last full block (the tail) are run twice, on `x` and on the copy in `y_prev`, and the two
//! results must agree mod n. A candidate failing either check gets [`Verdict::Alarm`].
//!
//! Every ladder-kernel submission of either path runs at most one segment, so the checked path
//! splits its blocks, its tail and its check passes the same way (a block's boundary multiplies go
//! into the submission of its last segment). The checked path keeps its adapted segment length
//! across blocks, passes and chunks, so a device fast enough for a whole block per submission makes
//! one submission per block.
//!
//! Every ladder dispatch gets `fp = [first iteration, bits, words, role]`. The generated kernels
//! ignore the role (`fp.w`): 0 for the primary computation on `x`, 1 for the tail recomputation's
//! copy, 2 for the check's own passes (`A^(2^GEC_BLOCK)` and `2^(sum V)`). It lets a test that
//! builds the runner from a mutated kernel ([`Runner::new_with_source_async`]) simulate a transient fault
//! in exactly one of these computations.

use crate::gpu::{Gpu, guarded};
use crate::kernel::{KernelConfig, Target, Variant};
use crate::scheme::LimbScheme;
use bigmont_oracle::{GEC_BLOCK, gec_blocks, gec_window_sum_words};
use num_bigint::BigUint;
use num_traits::One;
use web_time::Instant;

#[derive(Clone, Copy, Debug, Default)]
pub struct RunStats {
    /// Wall time of the ladder dispatches (submit to completion).
    pub gpu_secs: f64,
    /// Host-side Montgomery preparation (`R mod n`, `2R mod n`, packing).
    pub host_prep_secs: f64,
    /// Candidates tested.
    pub tests: u64,
    /// Submissions containing the ladder kernel (plain and checked runs).
    pub dispatches: u64,
}

/// The smallest bit length a [`Runner`] accepts: 2, the length of n = 3, the smallest odd number
/// the Fermat test applies to. Every length from here up is supported (the kernel arithmetic needs
/// only R = 2^(limb_bits * limbs) > 16n, which `KernelConfig::for_bits_scheme` and
/// `KernelConfig::retarget` guarantee at any size).
pub const MIN_BITS: u32 = 2;

/// The error of a checked run on a runner without the Gerbicz state.
const NO_GEC: &str = "checked runs need Runner::enable_gec first";

/// The outcome of one checked Fermat test ([`Runner::fermat_checked_async`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// `2^(n-1) != 1 (mod n)`: n is composite.
    Composite,
    /// `2^(n-1) == 1 (mod n)`: n is a base-2 probable prime.
    Prp,
    /// The Gerbicz check or the tail recomputation failed: the computation is not trustworthy and
    /// the candidate needs a retest.
    Alarm,
}

impl Verdict {
    /// The verdict byte of the driver's output: 0 composite, 1 probable prime, 2 alarm.
    pub fn byte(self) -> u8 {
        match self {
            Verdict::Composite => 0,
            Verdict::Prp => 1,
            Verdict::Alarm => 2,
        }
    }
}

/// `fp.w` of the primary computation's ladder dispatches.
const ROLE_PRIMARY: u32 = 0;
/// `fp.w` of the tail recomputation's copy.
const ROLE_TAIL_COPY: u32 = 1;
/// `fp.w` of the span check's passes over `A` and `y_prev`.
const ROLE_CHECK: u32 = 2;

/// The extra device state for checked runs ([`Runner::enable_gec_async`]).
struct Gec {
    mul_pipeline: wgpu::ComputePipeline,
    /// `[count, 1, 0, 0]` for the multiply kernel.
    mul_params: wgpu::Buffer,
    /// `A`, the product of the block starts.
    a_buf: wgpu::Buffer,
    /// `B`, the product of the block ends.
    b_buf: wgpu::Buffer,
    /// `y_prev`, the ladder state at the last block boundary (then the tail copy, then `2^(sum V)`).
    y_buf: wgpu::Buffer,
    /// `[lo, hi]` of `sum V` per candidate.
    sumv: wgpu::Buffer,
    /// `fp` of the tail recomputation's copy (role [`ROLE_TAIL_COPY`]).
    fp_tail: wgpu::Buffer,
    /// Readback for `A` and `B` (or the tail copy).
    staging2: wgpu::Buffer,
    /// Multiply `A <- A * y_prev`.
    bg_mul_a: wgpu::BindGroup,
    /// Multiply `B <- B * x`.
    bg_mul_b: wgpu::BindGroup,
    /// Ladder over `A` with an all-zero exponent (pure squarings).
    bg_lad_a: wgpu::BindGroup,
    /// Ladder over `y_prev` with the `sum V` exponent.
    bg_lad_y: wgpu::BindGroup,
    /// Ladder over `y_prev` with the real exponent (the tail recomputation).
    bg_lad_tail: wgpu::BindGroup,
}

/// A reusable batched Fermat tester for one kernel configuration and bit length.
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
    /// Iterations per ladder submission; `None` adapts so one submission takes about
    /// `target_secs`. Applies to plain and checked runs.
    pub segment_override: Option<u32>,
    /// Seconds one ladder dispatch should take when `segment_override` is `None`.
    pub target_secs: f64,
    /// Accumulated timings and counts.
    pub stats: RunStats,
    /// Device state for checked runs; `None` until [`Runner::enable_gec_async`].
    gec: Option<Gec>,
    /// The checked path's adapted segment length, kept across blocks, passes and chunks.
    checked_segment: u32,
}

/// One ladder pass of a segmented submission: the bind group, the `fp` uniform it reads and that
/// uniform's `[bits, words, role]` (its `fp.x` is each segment's first iteration).
struct LadderPass<'a> {
    bg: &'a wgpu::BindGroup,
    fp: &'a wgpu::Buffer,
    fp_yzw: [u32; 3],
}

/// What [`Runner::ladder_segments`] did.
struct Segmented {
    /// Submit-to-completion wall time of its submissions.
    secs: f64,
    /// Its submissions.
    dispatches: u64,
    /// The adapted segment length to start the next range with.
    seg: u32,
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
    pub async fn new_async(
        gpu: &Gpu,
        cfg: KernelConfig,
        bits: u32,
        capacity: u32,
    ) -> Result<Runner, String> {
        // Checked before the source is generated, so a bad configuration gets the runner's error.
        Self::check_new(gpu, cfg, bits, capacity)?;
        let source = cfg
            .try_fermat_wgsl_for(Target::host())
            .map_err(|e| gpu.explain(e))?;
        Self::new_with_source_async(gpu, cfg, bits, capacity, source).await
    }

    /// [`Runner::new_async`] with a caller-supplied Fermat ladder source instead of the generated one.
    ///
    /// `fermat_wgsl` must follow the binding contract of [`KernelConfig::try_fermat_wgsl_for`]
    /// for `cfg` (bindings 0 to 5, the same instance layout and `fp` meaning; `fp.w` carries the
    /// pass role described in the module docs). This is a seam for experiments and for simulating
    /// faulty devices in tests; production code uses [`Runner::new_async`].
    pub async fn new_with_source_async(
        gpu: &Gpu,
        cfg: KernelConfig,
        bits: u32,
        capacity: u32,
        fermat_wgsl: String,
    ) -> Result<Runner, String> {
        Self::check_new(gpu, cfg, bits, capacity)?;
        Self::build(gpu, cfg, bits, capacity, fermat_wgsl)
            .await
            .map_err(|e| gpu.explain(e))
    }

    /// The configuration, size and device checks of [`Runner::new_async`].
    fn check_new(gpu: &Gpu, cfg: KernelConfig, bits: u32, capacity: u32) -> Result<(), String> {
        cfg.validate()?;
        if cfg.variant != Variant::Unrolled {
            return Err("the Fermat runner needs the unrolled variant".into());
        }
        // The ladder runs `bits - 1` iterations and reads exponent bit `bits - 2 - iteration`, which
        // needs bits >= 2; below that there is no odd n >= 3 anyway.
        if bits < MIN_BITS {
            return Err(format!(
                "{bits} bits is below the {MIN_BITS}-bit minimum (the smallest odd candidate, n = 3, has {MIN_BITS} bits)"
            ));
        }
        if cfg.r_bits() < bits + 4 {
            return Err(format!(
                "{} limbs give R = 2^{}, which is not above 16n for {bits}-bit n (use KernelConfig::for_bits)",
                cfg.limbs,
                cfg.r_bits()
            ));
        }
        if capacity == 0 {
            return Err("capacity must be at least 1".into());
        }
        cfg.check_caps(gpu.caps())?;
        gpu.check_subgroups(cfg.tpi)?;
        gpu.check_health()
    }

    async fn build(
        gpu: &Gpu,
        cfg: KernelConfig,
        bits: u32,
        capacity: u32,
        source: String,
    ) -> Result<Runner, String> {
        let words = bits.div_ceil(32);
        let limits = gpu.device.limits();
        let max_groups = u64::from(limits.max_compute_workgroups_per_dimension.min(65535));
        let by_groups = max_groups * u64::from(cfg.instances_per_workgroup());
        let by_limbs = limits.max_storage_buffer_binding_size / (u64::from(cfg.limbs) * 4);
        let by_words = limits.max_storage_buffer_binding_size / (u64::from(words) * 4);
        let cap = u64::from(capacity)
            .min(by_groups)
            .min(by_limbs)
            .min(by_words);
        let cap = u32::try_from(cap).map_err(|_| "capacity overflow".to_string())?;
        if cap == 0 {
            return Err("the device's storage binding limit is too small for one candidate".into());
        }

        let device = &gpu.device;
        let limb_bytes = u64::from(cap) * u64::from(cfg.limbs) * 4;
        let (params, fp, n_buf, ninv_buf, x_buf, e_buf, staging) =
            guarded(device, "buffer creation", || {
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
                let e_buf = storage(
                    "e",
                    u64::from(cap) * u64::from(words) * 4,
                    wgpu::BufferUsages::empty(),
                );
                let staging = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("staging"),
                    size: limb_bytes,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                (params, fp, n_buf, ninv_buf, x_buf, e_buf, staging)
            })
            .await?;

        let module = guarded(device, &format!("Fermat shader module for {cfg:?}"), || {
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("fermat"),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            })
        })
        .await?;
        let pipeline = guarded(
            device,
            &format!("Fermat compute pipeline for {cfg:?}"),
            || {
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("fermat"),
                    layout: None,
                    module: &module,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                })
            },
        )
        .await?;
        let bind_group = guarded(device, "bind group creation", || {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("fermat"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: n_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: ninv_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: x_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: e_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: fp.as_entire_binding(),
                    },
                ],
            })
        })
        .await?;
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
            gec: None,
            checked_segment: 1,
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
    pub async fn fermat_async(&mut self, gpu: &Gpu, ns: &[BigUint]) -> Result<Vec<bool>, String> {
        gpu.check_health()?;
        let mut out = Vec::with_capacity(ns.len());
        for (k, chunk) in ns.chunks(self.cap as usize).enumerate() {
            out.extend(
                self.fermat_chunk(gpu, chunk, k * self.cap as usize)
                    .await
                    .map_err(|e| gpu.explain(e))?,
            );
        }
        Ok(out)
    }

    /// One dispatch-sized chunk; `base` is the index of `ns[0]` in the caller's slice (for errors).
    async fn fermat_chunk(
        &mut self,
        gpu: &Gpu,
        ns: &[BigUint],
        base: usize,
    ) -> Result<Vec<bool>, String> {
        self.validate_chunk(ns, base)?;
        let count = ns.len() as u32;
        if count == 0 {
            return Ok(Vec::new());
        }
        let preps = self.upload_chunk(gpu, ns);
        self.ladder_range(gpu, count, 0, self.bits - 1).await?;
        let x = self.read_x(gpu, count).await?;
        let limbs = self.cfg.limbs as usize;
        let scheme = self.cfg.scheme;
        let verdicts = ns
            .iter()
            .zip(&preps)
            .enumerate()
            .map(|(i, (n, p))| scheme.unpack(&x[i * limbs..(i + 1) * limbs]) % n == p.r)
            .collect();
        self.stats.tests += u64::from(count);
        Ok(verdicts)
    }

    /// Every candidate must be odd with exactly `bits` bits; `base` is the index of `ns[0]` in the
    /// caller's slice (for errors).
    fn validate_chunk(&self, ns: &[BigUint], base: usize) -> Result<(), String> {
        let bits = self.bits;
        for (i, n) in ns.iter().enumerate() {
            if !n.bit(0) || n.bits() != u64::from(bits) {
                return Err(format!(
                    "candidate {} must be odd with exactly {bits} bits (has {} bits, odd = {})",
                    base + i,
                    n.bits(),
                    n.bit(0)
                ));
            }
        }
        Ok(())
    }

    /// Host preparation for `ns` and the uploads of `n`, the start values, `ninv` and the exponents.
    fn upload_chunk(&mut self, gpu: &Gpu, ns: &[BigUint]) -> Vec<Prep> {
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

        let queue = &gpu.queue;
        queue.write_buffer(&self.n_buf, 0, bytemuck::cast_slice(&n_all));
        queue.write_buffer(&self.x_buf, 0, bytemuck::cast_slice(&x_all));
        queue.write_buffer(&self.ninv_buf, 0, bytemuck::cast_slice(&ninv_all));
        queue.write_buffer(&self.e_buf, 0, bytemuck::cast_slice(&e_all));
        preps
    }

    /// Ladder iterations `from..to` on the first `count` candidates of `x_buf`, as a sequence of
    /// short dispatches whose length adapts so each takes about `target_secs`. Every call starts
    /// with one iteration per dispatch: the first timing is cheap even on a CPU adapter.
    async fn ladder_range(
        &mut self,
        gpu: &Gpu,
        count: u32,
        from: u32,
        to: u32,
    ) -> Result<(), String> {
        let pass = self.primary_pass();
        let done = self
            .ladder_segments(gpu, count, (from, to), &[pass], false, 1)
            .await?;
        self.record(&done);
        Ok(())
    }

    /// The ladder pass over `x` with the real exponent (role [`ROLE_PRIMARY`]).
    fn primary_pass(&self) -> LadderPass<'_> {
        LadderPass {
            bg: &self.bind_group,
            fp: &self.fp,
            fp_yzw: [self.bits, self.words, ROLE_PRIMARY],
        }
    }

    /// Ladder iterations `from..to` of every pass in `passes` on the first `count` candidates, as a
    /// sequence of submissions of at most one segment each; every submission runs all the passes on
    /// the same segment with the same `params`. With `boundary`, the submission of the last segment
    /// ends with `A <- A * y_prev`, `B <- B * x` and `y_prev <- x` (in that order: the multiplies use
    /// the old `y_prev`, the copy comes last).
    ///
    /// The segment length is `segment_override` if set; otherwise it starts at `seg` and after each
    /// submission moves toward `target_secs` from the measured time per iteration, at most doubling.
    async fn ladder_segments(
        &self,
        gpu: &Gpu,
        count: u32,
        (from, to): (u32, u32),
        passes: &[LadderPass<'_>],
        boundary: bool,
        seg: u32,
    ) -> Result<Segmented, String> {
        let (device, queue) = (&gpu.device, &gpu.queue);
        let groups = count.div_ceil(self.cfg.instances_per_workgroup());
        let gec = match boundary {
            true => Some(self.gec.as_ref().ok_or(NO_GEC)?),
            false => None,
        };
        let mut out = Segmented {
            secs: 0.0,
            dispatches: 0,
            seg: seg.max(1),
        };
        let mut len = self.segment_override.unwrap_or(out.seg).max(1);
        let mut done = from;
        while done < to {
            let this = len.min(to - done);
            queue.write_buffer(
                &self.params,
                0,
                bytemuck::cast_slice(&[count, this, 0u32, 0u32]),
            );
            let mut enc = device.create_command_encoder(&Default::default());
            for p in passes {
                let [y, z, w] = p.fp_yzw;
                queue.write_buffer(p.fp, 0, bytemuck::cast_slice(&[done, y, z, w]));
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, p.bg, &[]);
                pass.dispatch_workgroups(groups, 1, 1);
            }
            if let Some(gec) = gec.filter(|_| done + this == to) {
                for bg in [&gec.bg_mul_a, &gec.bg_mul_b] {
                    let mut pass = enc.begin_compute_pass(&Default::default());
                    pass.set_pipeline(&gec.mul_pipeline);
                    pass.set_bind_group(0, bg, &[]);
                    pass.dispatch_workgroups(groups, 1, 1);
                }
                let bytes = u64::from(count) * u64::from(self.cfg.limbs) * 4;
                enc.copy_buffer_to_buffer(&self.x_buf, 0, &gec.y_buf, 0, bytes);
            }
            let t = Instant::now();
            queue.submit([enc.finish()]);
            gpu.wait_async().await?;
            let dt = t.elapsed().as_secs_f64();
            out.secs += dt;
            out.dispatches += 1;
            done += this;
            if self.segment_override.is_none() {
                let per_iter = dt / f64::from(this);
                // Grow by at most 2x per dispatch so one slow measurement cannot overshoot far.
                len = ((self.target_secs / per_iter) as u32).clamp(1, len.saturating_mul(2).max(2));
                out.seg = len;
            }
        }
        Ok(out)
    }

    /// Adds a [`Runner::ladder_segments`] result to the stats.
    fn record(&mut self, done: &Segmented) {
        self.stats.gpu_secs += done.secs;
        self.stats.dispatches += done.dispatches;
    }

    /// [`Runner::record`] for the checked path, which also keeps the adapted segment length.
    fn record_checked(&mut self, done: &Segmented) {
        self.record(done);
        self.checked_segment = done.seg;
    }

    /// Reads back the ladder state of the first `count` candidates (`count * limbs` words) and
    /// checks that every limb is normalised.
    async fn read_x(&self, gpu: &Gpu, count: u32) -> Result<Vec<u32>, String> {
        let (device, queue) = (&gpu.device, &gpu.queue);
        let bytes = u64::from(count) * u64::from(self.cfg.limbs) * 4;
        let mut enc = device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&self.x_buf, 0, &self.staging, 0, bytes);
        queue.submit([enc.finish()]);
        let x = gpu.read_back_async(&self.staging, bytes).await?;
        if !self.cfg.scheme.normalised(&x) {
            return Err("kernel returned an unnormalised limb".into());
        }
        Ok(x)
    }

    /// Allocates the state for checked runs ([`Runner::fermat_checked_async`]): the multiply pipeline and
    /// three more numbers per candidate (`A`, `B`, `y_prev`) plus small buffers. Calling it again
    /// does nothing. Errors (including a lost device) are an `Err`.
    pub async fn enable_gec_async(&mut self, gpu: &Gpu) -> Result<(), String> {
        if self.gec.is_some() {
            return Ok(());
        }
        let source = self
            .cfg
            .try_mul_wgsl_for(Target::host())
            .map_err(|e| gpu.explain(e))?;
        self.enable_gec_with_source_async(gpu, source).await
    }

    /// [`Runner::enable_gec_async`] with a caller-supplied multiply source instead of the generated one.
    ///
    /// `mul_wgsl` must follow the binding contract of [`KernelConfig::try_mul_wgsl_for`] for this
    /// runner's configuration (bindings 0 to 4). This is a seam for experiments and for simulating
    /// faulty devices in tests. Does nothing when the check state already exists.
    pub async fn enable_gec_with_source_async(
        &mut self,
        gpu: &Gpu,
        mul_wgsl: String,
    ) -> Result<(), String> {
        if self.gec.is_some() {
            return Ok(());
        }
        gpu.check_health()?;
        let gec = self
            .build_gec(gpu, mul_wgsl)
            .await
            .map_err(|e| gpu.explain(e))?;
        self.gec = Some(gec);
        Ok(())
    }

    async fn build_gec(&self, gpu: &Gpu, source: String) -> Result<Gec, String> {
        let cfg = self.cfg;
        let device = &gpu.device;
        let cap = u64::from(self.cap);
        let limb_bytes = cap * u64::from(cfg.limbs) * 4;
        let (mul_params, fp_tail, a_buf, b_buf, y_buf, zero_e, sumv, staging2) =
            guarded(device, "Gerbicz buffer creation", || {
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
                let mul_params = uniform("gec mul params");
                let fp_tail = uniform("gec tail fp");
                // A and B are read back for the comparison; x is copied into y_prev, and y_prev
                // (the tail copy) is read back.
                let a_buf = storage("gec A", limb_bytes, wgpu::BufferUsages::COPY_SRC);
                let b_buf = storage("gec B", limb_bytes, wgpu::BufferUsages::COPY_SRC);
                let y_buf = storage("gec y_prev", limb_bytes, wgpu::BufferUsages::COPY_SRC);
                // Never written: wgpu zero-initialises buffers, so this is an all-zero exponent.
                let zero_e = storage("gec zero exponent", cap * 8, wgpu::BufferUsages::empty());
                let sumv = storage("gec sum V", cap * 8, wgpu::BufferUsages::empty());
                // A and B side by side (or the tail copy alone).
                let staging2 = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("gec staging"),
                    size: 2 * limb_bytes,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                (
                    mul_params, fp_tail, a_buf, b_buf, y_buf, zero_e, sumv, staging2,
                )
            })
            .await?;
        let module = guarded(
            device,
            &format!("multiply shader module for {cfg:?}"),
            || {
                device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("mont_mul"),
                    source: wgpu::ShaderSource::Wgsl(source.into()),
                })
            },
        )
        .await?;
        let mul_pipeline = guarded(device, &format!("multiply pipeline for {cfg:?}"), || {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("mont_mul"),
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            })
        })
        .await?;
        // Bindings 0.. in order: the bind groups keep their buffers (e.g. `zero_e`) alive.
        let group = |layout: &wgpu::BindGroupLayout, bufs: &[&wgpu::Buffer]| {
            let entries: Vec<wgpu::BindGroupEntry> = bufs
                .iter()
                .zip(0u32..)
                .map(|(buf, binding)| wgpu::BindGroupEntry {
                    binding,
                    resource: buf.as_entire_binding(),
                })
                .collect();
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gec"),
                layout,
                entries: &entries,
            })
        };
        let (bg_mul_a, bg_mul_b, bg_lad_a, bg_lad_y, bg_lad_tail) =
            guarded(device, "Gerbicz bind group creation", || {
                let mul = mul_pipeline.get_bind_group_layout(0);
                let lad = self.pipeline.get_bind_group_layout(0);
                let (p, n, ninv, fp) = (&self.params, &self.n_buf, &self.ninv_buf, &self.fp);
                (
                    group(&mul, &[&mul_params, n, ninv, &a_buf, &y_buf]),
                    group(&mul, &[&mul_params, n, ninv, &b_buf, &self.x_buf]),
                    group(&lad, &[p, n, ninv, &a_buf, &zero_e, fp]),
                    group(&lad, &[p, n, ninv, &y_buf, &sumv, fp]),
                    group(&lad, &[p, n, ninv, &y_buf, &self.e_buf, &fp_tail]),
                )
            })
            .await?;
        Ok(Gec {
            mul_pipeline,
            mul_params,
            fp_tail,
            a_buf,
            b_buf,
            y_buf,
            sumv,
            staging2,
            bg_mul_a,
            bg_mul_b,
            bg_lad_a,
            bg_lad_y,
            bg_lad_tail,
        })
    }

    /// Base-2 Fermat verdicts for `ns` with the Gerbicz check: like [`Runner::fermat_async`], but a
    /// candidate whose computation fails the check gets [`Verdict::Alarm`] instead of a verdict.
    /// Needs [`Runner::enable_gec_async`] first (else `Err`).
    pub async fn fermat_checked_async(
        &mut self,
        gpu: &Gpu,
        ns: &[BigUint],
    ) -> Result<Vec<Verdict>, String> {
        if self.gec.is_none() {
            return Err(NO_GEC.into());
        }
        gpu.check_health()?;
        let cap = self.cap as usize;
        let mut out = Vec::with_capacity(ns.len());
        for (k, chunk) in ns.chunks(cap).enumerate() {
            out.extend(
                self.checked_chunk(gpu, chunk, k * cap)
                    .await
                    .map_err(|e| gpu.explain(e))?,
            );
        }
        Ok(out)
    }

    /// One checked chunk; `base` is the index of `ns[0]` in the caller's slice (for errors).
    async fn checked_chunk(
        &mut self,
        gpu: &Gpu,
        ns: &[BigUint],
        base: usize,
    ) -> Result<Vec<Verdict>, String> {
        self.validate_chunk(ns, base)?;
        let count = ns.len() as u32;
        if count == 0 {
            return Ok(Vec::new());
        }
        let bits = self.bits;
        let limbs = self.cfg.limbs as usize;
        let scheme = self.cfg.scheme;
        let nb = gec_blocks(bits, GEC_BLOCK);
        let tail = bits - 1 - nb * GEC_BLOCK;
        let bytes = u64::from(count) * limbs as u64 * 4;

        // 1. Upload: the ladder inputs, A = B = 1~, y_prev = x0, and sum V.
        let preps = self.upload_chunk(gpu, ns);
        let t = Instant::now();
        let mut one_all = Vec::with_capacity(ns.len() * limbs);
        let mut x0_all = Vec::with_capacity(ns.len() * limbs);
        let mut sum_all = Vec::with_capacity(ns.len() * 2);
        for p in &preps {
            one_all.extend(scheme.pack(&p.r, limbs));
            x0_all.extend_from_slice(&p.x0);
            let sum = gec_window_sum_words(&p.e, bits, GEC_BLOCK);
            sum_all.extend([sum as u32, (sum >> 32) as u32]);
        }
        self.stats.host_prep_secs += t.elapsed().as_secs_f64();
        {
            let gec = self.gec.as_ref().ok_or(NO_GEC)?;
            let queue = &gpu.queue;
            queue.write_buffer(&gec.a_buf, 0, bytemuck::cast_slice(&one_all));
            queue.write_buffer(&gec.b_buf, 0, bytemuck::cast_slice(&one_all));
            queue.write_buffer(&gec.y_buf, 0, bytemuck::cast_slice(&x0_all));
            queue.write_buffer(&gec.sumv, 0, bytemuck::cast_slice(&sum_all));
            queue.write_buffer(
                &gec.mul_params,
                0,
                bytemuck::cast_slice(&[count, 1u32, 0u32, 0u32]),
            );
        }

        // 2. The blocks: ladder, then (at the boundary) A <- A * y_prev, B <- B * x, y_prev <- x.
        for blk in 0..nb {
            self.block_step(gpu, count, blk * GEC_BLOCK).await?;
        }

        // 3. The tail runs on `x` and on its copy in `y_prev` (both equal x0 when nb == 0); the two
        // must agree mod n.
        let mut tail_copy = Vec::new();
        if tail > 0 {
            self.tail_step(gpu, count, nb * GEC_BLOCK, tail).await?;
            let gec = self.gec.as_ref().ok_or(NO_GEC)?;
            let mut enc = gpu.device.create_command_encoder(&Default::default());
            enc.copy_buffer_to_buffer(&gec.y_buf, 0, &gec.staging2, 0, bytes);
            gpu.queue.submit([enc.finish()]);
            tail_copy = gpu.read_back_async(&gec.staging2, bytes).await?;
            if !scheme.normalised(&tail_copy) {
                return Err("kernel returned an unnormalised limb".into());
            }
        }

        // 4. The span check: A^(2^GEC_BLOCK) * 2^(sum V) == B (mod n).
        let mut alarm = vec![false; ns.len()];
        if nb > 0 {
            // a. A <- A^(2^GEC_BLOCK): exponent bits 40..1 of the all-zero `zero_e`, so no doubling.
            self.check_pass(gpu, count, |g| &g.bg_lad_a, GEC_BLOCK, GEC_BLOCK + 2)
                .await?;
            // b. y_prev <- (2^(sum V))~: from 1~, read the bits L-1..0 of `sumv`, where
            // L = 40 + bitlen(nb) bounds the bit length of sum V < nb * 2^40 (and is at most 64).
            let gec = self.gec.as_ref().ok_or(NO_GEC)?;
            gpu.queue
                .write_buffer(&gec.y_buf, 0, bytemuck::cast_slice(&one_all));
            let l = GEC_BLOCK + (32 - nb.leading_zeros());
            self.check_pass(gpu, count, |g| &g.bg_lad_y, l, l + 1)
                .await?;
            let gec = self.gec.as_ref().ok_or(NO_GEC)?;
            // c. A <- A * 2^(sum V).
            let groups = count.div_ceil(self.cfg.instances_per_workgroup());
            let mut enc = gpu.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&gec.mul_pipeline);
                pass.set_bind_group(0, &gec.bg_mul_a, &[]);
                pass.dispatch_workgroups(groups, 1, 1);
            }
            let t = Instant::now();
            gpu.queue.submit([enc.finish()]);
            gpu.wait_async().await?;
            let dt = t.elapsed().as_secs_f64();
            // d. Read A and B back and compare them (both are in [0, 2n)).
            let mut enc = gpu.device.create_command_encoder(&Default::default());
            enc.copy_buffer_to_buffer(&gec.a_buf, 0, &gec.staging2, 0, bytes);
            enc.copy_buffer_to_buffer(&gec.b_buf, 0, &gec.staging2, bytes, bytes);
            gpu.queue.submit([enc.finish()]);
            let ab = gpu.read_back_async(&gec.staging2, 2 * bytes).await?;
            if !scheme.normalised(&ab) {
                return Err("kernel returned an unnormalised limb".into());
            }
            let (a, b) = ab.split_at(ab.len() / 2);
            for (i, n) in ns.iter().enumerate() {
                let range = i * limbs..(i + 1) * limbs;
                alarm[i] |= !same_mod_n(&a[range.clone()], &b[range], n, scheme);
            }
            self.stats.gpu_secs += dt;
        }

        // 5. The verdicts, with the tail recomputation's comparison.
        let x = self.read_x(gpu, count).await?;
        let verdicts = ns
            .iter()
            .zip(&preps)
            .enumerate()
            .map(|(i, (n, p))| {
                let range = i * limbs..(i + 1) * limbs;
                let tail_differs = tail > 0
                    && !same_mod_n(&x[range.clone()], &tail_copy[range.clone()], n, scheme);
                if alarm[i] || tail_differs {
                    Verdict::Alarm
                } else if scheme.unpack(&x[range]) % n == p.r {
                    Verdict::Prp
                } else {
                    Verdict::Composite
                }
            })
            .collect();
        self.stats.tests += u64::from(count);
        Ok(verdicts)
    }

    /// The [`GEC_BLOCK`] ladder iterations of the block starting at iteration `first` on `x`, then
    /// (in the submission of the block's last segment) `A <- A * y_prev`, `B <- B * x` and
    /// `y_prev <- x`.
    async fn block_step(&mut self, gpu: &Gpu, count: u32, first: u32) -> Result<(), String> {
        let pass = self.primary_pass();
        let range = (first, first + GEC_BLOCK);
        let done = self
            .ladder_segments(gpu, count, range, &[pass], true, self.checked_segment)
            .await?;
        self.record_checked(&done);
        Ok(())
    }

    /// `iters` tail iterations from iteration `first` on both `x` (role [`ROLE_PRIMARY`]) and its
    /// copy in `y_prev` (role [`ROLE_TAIL_COPY`]); each segment's submission runs both.
    async fn tail_step(
        &mut self,
        gpu: &Gpu,
        count: u32,
        first: u32,
        iters: u32,
    ) -> Result<(), String> {
        let gec = self.gec.as_ref().ok_or(NO_GEC)?;
        let copy = LadderPass {
            bg: &gec.bg_lad_tail,
            fp: &gec.fp_tail,
            fp_yzw: [self.bits, self.words, ROLE_TAIL_COPY],
        };
        let passes = [self.primary_pass(), copy];
        let range = (first, first + iters);
        let done = self
            .ladder_segments(gpu, count, range, &passes, false, self.checked_segment)
            .await?;
        self.record_checked(&done);
        Ok(())
    }

    /// `iters` iterations from 0 of one of the span check's passes over `bg` (role
    /// [`ROLE_CHECK`]), as if on a `bits`-bit exponent of two words: it reads exponent bits
    /// `bits - 2` down to `bits - 1 - iters`.
    async fn check_pass(
        &mut self,
        gpu: &Gpu,
        count: u32,
        bg: fn(&Gec) -> &wgpu::BindGroup,
        iters: u32,
        bits: u32,
    ) -> Result<(), String> {
        let gec = self.gec.as_ref().ok_or(NO_GEC)?;
        let pass = LadderPass {
            bg: bg(gec),
            fp: &self.fp,
            fp_yzw: [bits, 2, ROLE_CHECK],
        };
        let done = self
            .ladder_segments(gpu, count, (0, iters), &[pass], false, self.checked_segment)
            .await?;
        self.record_checked(&done);
        Ok(())
    }
}

/// Whether two kernel outputs (both in `[0, 2n)`) are equal mod `n`: equal, or `n` apart.
fn same_mod_n(a: &[u32], b: &[u32], n: &BigUint, scheme: LimbScheme) -> bool {
    if a == b {
        return true;
    }
    let (a, b) = (scheme.unpack(a), scheme.unpack(b));
    let d = if a > b { a - b } else { b - a };
    d == *n
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
        Prep {
            n: scheme.pack(n, limbs),
            x0: scheme.pack(&x0, limbs),
            ninv: scheme.neg_inv(n),
            e,
            r,
        }
    };
    // A browser page has no worker threads here: prepare on the one it has.
    #[cfg(target_arch = "wasm32")]
    {
        ns.iter().map(one).collect()
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let threads = std::thread::available_parallelism()
            .map_or(1, |t| t.get())
            .min(ns.len())
            .max(1);
        if threads == 1 {
            return ns.iter().map(one).collect();
        }
        let per = ns.len().div_ceil(threads);
        std::thread::scope(|s| {
            let handles: Vec<_> = ns
                .chunks(per)
                .map(|part| s.spawn(|| part.iter().map(one).collect::<Vec<_>>()))
                .collect();
            handles
                .into_iter()
                .flat_map(|h| h.join().expect("host prep thread panicked"))
                .collect()
        })
    }
}
