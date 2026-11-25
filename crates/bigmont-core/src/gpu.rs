//! wgpu device setup, input batches, and the kernel runner.

use crate::kernel::KernelConfig;
use bigmont_oracle::{LIMBS, XorShift64, neg_inv_limb, pack_limbs_n, random_below, random_odd_modulus};
use num_bigint::BigUint;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

/// Serialises instance/adapter/device creation. Mitigation; root cause not investigated: creating
/// several wgpu instances at once from parallel test threads intermittently made
/// `enumerate_adapters` miss the requested adapter. Benchmarks should run in a single process
/// (the spike example does).
static INIT_LOCK: Mutex<()> = Mutex::new(());

/// Which adapter to open. All fields `None` means the high-performance default.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AdapterSelector {
    /// Case-insensitive substring of the adapter name.
    pub name: Option<String>,
    /// Case-insensitive substring of the backend name ("Vulkan", "Dx12", "Gl", ...).
    pub backend: Option<String>,
    /// Index among the non-CPU adapters (after the name/backend filters), in `enumerate_adapters`
    /// order: Beacon's `--device N`.
    pub index: Option<usize>,
}

impl AdapterSelector {
    /// `BIGMONT_ADAPTER` and `BIGMONT_BACKEND`.
    pub fn from_env() -> AdapterSelector {
        AdapterSelector { name: std::env::var("BIGMONT_ADAPTER").ok(), backend: std::env::var("BIGMONT_BACKEND").ok(), index: None }
    }
}

pub struct Gpu {
    pub info: wgpu::AdapterInfo,
    pub has_subgroups: bool,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

/// Parses a `BIGMONT_INSTANCE_BACKENDS` value: a comma list of `vulkan`, `dx12`, `metal`, `gl`,
/// `primary` (Vulkan, Dx12 and Metal) and `all` (every backend, GL included).
pub fn parse_instance_backends(list: &str) -> Result<wgpu::Backends, String> {
    let mut backends = wgpu::Backends::empty();
    for token in list.split(',').map(|t| t.trim().to_lowercase()).filter(|t| !t.is_empty()) {
        backends |= match token.as_str() {
            "vulkan" => wgpu::Backends::VULKAN,
            "dx12" => wgpu::Backends::DX12,
            "metal" => wgpu::Backends::METAL,
            "gl" => wgpu::Backends::GL,
            "primary" => wgpu::Backends::PRIMARY,
            "all" => wgpu::Backends::all(),
            other => return Err(format!("unknown backend '{other}' (use vulkan, dx12, metal, gl, primary, all)")),
        };
    }
    if backends.is_empty() {
        return Err("no backends listed".into());
    }
    Ok(backends)
}

/// The backends the wgpu instance initialises: `Backends::PRIMARY` (Vulkan, Dx12, Metal) unless
/// `BIGMONT_INSTANCE_BACKENDS` says otherwise.
///
/// GL is left out on purpose. Our kernels need subgroup operations, which the GL backend does not
/// offer, so it can never run them; and initialising it (EGL on Linux, WGL on Windows) alongside
/// Vulkan made test binaries crash at exit inside the AMD driver on Windows (6 of 60 runs with all
/// backends, 0 of 60 with Vulkan only and 0 of 60 with Vulkan + Dx12) and was associated with
/// NVIDIA Xid 32 device losses on Linux (8 of 160 runs with all backends, 1 of 160 with Vulkan
/// only). See docs/superpowers/results/2026-11-24-device-loss-investigation.md.
pub fn instance_backends() -> wgpu::Backends {
    match std::env::var("BIGMONT_INSTANCE_BACKENDS") {
        Ok(list) if !list.trim().is_empty() => parse_instance_backends(&list).unwrap_or_else(|e| {
            eprintln!("bigmont: ignoring BIGMONT_INSTANCE_BACKENDS ({e}); using the default");
            wgpu::Backends::PRIMARY
        }),
        _ => wgpu::Backends::PRIMARY,
    }
}

fn make_instance() -> wgpu::Instance {
    wgpu::Instance::new(wgpu::InstanceDescriptor { backends: instance_backends(), ..wgpu::InstanceDescriptor::new_without_display_handle() })
}

/// The one wgpu `Instance` of this process, created on first use and kept alive.
///
/// Why not one `Instance` per `Gpu::open`: each new Instance makes the Vulkan loader `dlopen` the
/// NVIDIA driver library and dropping it `dlclose`s it again. On Linux glibc cannot give back the
/// static thread-local storage such a library takes, so after about ten load/unload cycles in one
/// process the next load fails with "cannot allocate memory in static TLS block", the loader
/// silently drops the NVIDIA driver, and adapter selection finds no NVIDIA adapter (or, without a
/// name filter, silently picks another one). Verified on the RTX 4070 laptop: 10 cycles succeed and
/// the 11th fails; with `GLIBC_TUNABLES=glibc.rtld.optional_static_tls=65536` 40 of 40 succeed.
/// Sharing one Instance loads the driver once. See
/// docs/superpowers/results/2026-11-24-device-loss-investigation.md.
fn shared_instance() -> wgpu::Instance {
    static INSTANCE: OnceLock<wgpu::Instance> = OnceLock::new();
    INSTANCE.get_or_init(make_instance).clone()
}

fn describe(info: &wgpu::AdapterInfo) -> String {
    format!("{} [{:?}, {:?}] subgroup {}..{}", info.name, info.backend, info.device_type, info.subgroup_min_size, info.subgroup_max_size)
}

impl Gpu {
    /// Opens the adapter named by env `BIGMONT_ADAPTER` (substring of the adapter name) and
    /// `BIGMONT_BACKEND` (substring of the backend name, e.g. "Vulkan", "Dx12"); otherwise the
    /// high-performance default. Requests `Features::SUBGROUP` when the adapter has it.
    pub fn new() -> Result<Gpu, String> {
        Gpu::open(&AdapterSelector::from_env())
    }

    /// Opens the adapter chosen by `selector` (see [`AdapterSelector`]); the high-performance
    /// default when the selector is empty. Requests `Features::SUBGROUP` when the adapter has it.
    pub fn open(selector: &AdapterSelector) -> Result<Gpu, String> {
        let _init = INIT_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        pollster::block_on(async {
            let instance = shared_instance();
            let name = selector.name.as_ref().map(|s| s.to_lowercase());
            let backend = selector.backend.as_ref().map(|s| s.to_lowercase());
            let adapter = if name.is_some() || backend.is_some() || selector.index.is_some() {
                let mut matching: Vec<wgpu::Adapter> = instance
                    .enumerate_adapters(wgpu::Backends::all())
                    .await
                    .into_iter()
                    .filter(|a| {
                        let i = a.get_info();
                        name.as_ref().is_none_or(|w| i.name.to_lowercase().contains(w))
                            && backend.as_ref().is_none_or(|w| format!("{:?}", i.backend).to_lowercase().contains(w))
                    })
                    .collect();
                if let Some(index) = selector.index {
                    matching.retain(|a| a.get_info().device_type != wgpu::DeviceType::Cpu);
                    if index >= matching.len() {
                        return Err(format!("no adapter matches {selector:?} ({} non-CPU adapters match the name/backend filters)", matching.len()));
                    }
                    matching.swap_remove(index)
                } else {
                    matching.into_iter().next().ok_or_else(|| format!("no adapter matches {selector:?}"))?
                }
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
            let instance = shared_instance();
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
    /// Limbs per number.
    pub limbs: usize,
    pub n: Vec<u32>,
    pub ninv: Vec<u32>,
    pub x: Vec<u32>,
}

impl Batch {
    /// Batch at the Phase 0 size class (160 limbs).
    pub fn from_values(values: &[(BigUint, BigUint)]) -> Batch {
        Self::from_values_n(values, LIMBS)
    }

    /// Batch of `limbs`-limb numbers (every value must fit in `limbs` limbs).
    pub fn from_values_n(values: &[(BigUint, BigUint)], limbs: usize) -> Batch {
        let mut b = Batch { count: values.len() as u32, limbs, n: Vec::new(), ninv: Vec::new(), x: Vec::new() };
        for (n, x) in values {
            assert!(x < &(n * 2u32), "x must be below 2n");
            b.n.extend(pack_limbs_n(n, limbs));
            b.ninv.push(neg_inv_limb(n));
            b.x.extend(pack_limbs_n(x, limbs));
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

/// Runs `f` with Validation, Internal and OutOfMemory error scopes active, so that errors raised
/// while creating shader modules, pipelines, buffers or bind groups come back as `Err` instead of
/// reaching wgpu's uncaptured-error handler (which panics).
pub(crate) fn guarded<T>(device: &wgpu::Device, what: &str, f: impl FnOnce() -> T) -> Result<T, String> {
    let scopes = [wgpu::ErrorFilter::OutOfMemory, wgpu::ErrorFilter::Internal, wgpu::ErrorFilter::Validation].map(|filter| device.push_error_scope(filter));
    let out = f();
    let mut first = None;
    for scope in scopes.into_iter().rev() {
        let popped = pollster::block_on(scope.pop());
        first = first.or(popped);
    }
    match first {
        Some(e) => Err(format!("{what} rejected: {e}")),
        None => Ok(out),
    }
}

/// Uploads `batch`, runs `repeats` dispatches of `squarings` Montgomery squarings each (every
/// dispatch continues from the previous one's output), and reads the result back.
pub fn run(gpu: &Gpu, cfg: KernelConfig, batch: &Batch, squarings: u32, repeats: u32) -> Result<Run, String> {
    cfg.validate()?;
    let count = batch.count as usize;
    let limbs = cfg.limbs as usize;
    if count == 0 {
        return Err("batch is empty".into());
    }
    if batch.limbs != limbs {
        return Err(format!("batch has {} limbs per number but the configuration expects {limbs}", batch.limbs));
    }
    if batch.x.len() != count * limbs || batch.n.len() != count * limbs || batch.ninv.len() != count {
        return Err(format!(
            "batch lengths do not match count {count}: x {}, n {}, ninv {} (expected {}, {}, {count})",
            batch.x.len(),
            batch.n.len(),
            batch.ninv.len(),
            count * limbs,
            count * limbs
        ));
    }
    let groups = batch.count.div_ceil(cfg.instances_per_workgroup());
    if groups > 65535 {
        return Err(format!("{groups} workgroups exceeds the 65535 per dispatch limit"));
    }
    if cfg.needs_subgroups() {
        gpu.check_subgroups(cfg.tpi)?;
    }
    let (device, queue) = (&gpu.device, &gpu.queue);
    let words = (count * limbs) as u64;
    let (params, n_buf, ninv_buf, x_buf, staging) = guarded(device, "buffer creation", || {
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
        (params, n_buf, ninv_buf, x_buf, staging)
    })?;

    let module = guarded(device, &format!("shader module for {cfg:?}"), || {
        device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("mont"), source: wgpu::ShaderSource::Wgsl(cfg.wgsl().into()) })
    })?;
    let pipeline = guarded(device, &format!("compute pipeline for {cfg:?}"), || {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("mont"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        })
    })?;
    let bind_group = guarded(device, "bind group creation", || {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: n_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: ninv_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: x_buf.as_entire_binding() },
            ],
        })
    })?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_backend_lists() {
        assert_eq!(parse_instance_backends("vulkan").unwrap(), wgpu::Backends::VULKAN);
        assert_eq!(parse_instance_backends(" Vulkan , DX12 ").unwrap(), wgpu::Backends::VULKAN | wgpu::Backends::DX12);
        assert_eq!(parse_instance_backends("primary").unwrap(), wgpu::Backends::PRIMARY);
        assert_eq!(parse_instance_backends("primary,gl").unwrap(), wgpu::Backends::PRIMARY | wgpu::Backends::GL);
        assert_eq!(parse_instance_backends("all").unwrap(), wgpu::Backends::all());
        assert_eq!(parse_instance_backends("vulkan,,").unwrap(), wgpu::Backends::VULKAN);
    }

    #[test]
    fn rejects_unknown_or_empty_lists() {
        assert!(parse_instance_backends("opengl").is_err());
        assert!(parse_instance_backends("").is_err());
        assert!(parse_instance_backends(" , ").is_err());
    }

    #[test]
    fn the_default_leaves_gl_out() {
        // The env var is not set by the test harness; the default must be PRIMARY, which has no GL.
        if std::env::var_os("BIGMONT_INSTANCE_BACKENDS").is_none() {
            assert_eq!(instance_backends(), wgpu::Backends::PRIMARY);
            assert!(!instance_backends().contains(wgpu::Backends::GL));
        }
    }
}
