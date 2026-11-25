//! The driver engine: chooses GPU or CPU, buckets candidates by bit length, and cross-checks the
//! GPU against the CPU reference.
//!
//! Verdicts: `n < 2` or even `n != 2` is 0 (composite), `n == 2` is 1, everything else comes from
//! a backend. The engine never returns 2.
//!
//! Device-level GPU failures (a lost device, a GPU wait longer than the poll timeout, an
//! uncaptured wgpu error; see `bigmont_core::gpu::Gpu::check_health`) are recovered: the device
//! and its runners are dropped, the device is re-opened and the whole bucket is re-run. After
//! `Options::device_retries` re-opens within one `verdicts` call the engine gives up on the GPU and
//! answers on the CPU for the rest of the process. Verdicts are correct in every path: a failed
//! GPU run contributes nothing, the bucket is redone from scratch.

use crate::tune;
use bigmont_core::gpu::{AdapterSelector, Gpu, is_device_error};
use bigmont_core::kernel::{KernelConfig, Variant};
use bigmont_core::runner::Runner;
use num_bigint::BigUint;
use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    /// GPU when one opens (and has subgroups), otherwise the CPU.
    Auto,
    /// GPU only: `verdicts` fails if no adapter opens at `Engine::open`. Buckets still fall back
    /// to the CPU when autotuning or the runner returns an ordinary `Err` (counted in
    /// `EngineReport::gpu_errors`), and the whole engine falls back to the CPU when a lost device
    /// cannot be recovered (see `Options::device_retries`).
    Gpu,
    Cpu,
}

#[derive(Clone, Debug)]
pub struct Options {
    pub backend: Backend,
    pub selector: AdapterSelector,
    /// Tuning cache file; `None` tunes every run. (The `bigmont` binary fills this from `--cache`,
    /// `BIGMONT_CACHE` or the per-user default; the library default is no cache.)
    pub cache: Option<PathBuf>,
    pub retune: bool,
    /// One in `verify_rate` GPU composites is re-checked on the CPU (0 = never). Every GPU
    /// probable prime is always re-checked.
    pub verify_rate: u32,
    /// Bit-length buckets with fewer candidates than this run on the CPU (tuning and dispatch
    /// overhead is not worth it). This is the only size-based reason a bucket skips the GPU: the bit
    /// length itself does not matter, the GPU is used from 2-bit numbers (n = 3) up. (A failed
    /// autotune or runner `Err` also sends a bucket to the CPU; see `Backend::Gpu`.)
    pub min_gpu_batch: usize,
    /// Most candidates per GPU dispatch.
    pub capacity: u32,
    /// CPU worker threads, 0 = all cores.
    pub threads: usize,
    /// Device re-opens allowed per `verdicts` call after device-level GPU failures (successful
    /// re-opens count too, so a device that keeps dying cannot loop). When they are used up the
    /// failing bucket, every later bucket and every later call run on the CPU.
    pub device_retries: u32,
    /// Device re-opens allowed over the engine's whole life (the CLI calls `verdicts` once per
    /// chunk, so the per-call budget alone would let a device that dies on every chunk be re-opened,
    /// and leaked, without bound). Reaching it makes the engine CPU-only for the process.
    pub device_retries_total: u32,
    /// The longest one GPU wait may take (`Gpu::poll_timeout`); `None` keeps the `Gpu` default
    /// (`BIGMONT_POLL_TIMEOUT_MS`, else 30 s). A longer wait counts as a device-level failure.
    pub poll_timeout: Option<Duration>,
    /// Test only: destroy the next N freshly opened devices (the first one and re-opened ones)
    /// right after opening, which makes wgpu report them lost.
    #[doc(hidden)]
    pub inject_loss_on_open: u32,
}

impl Default for Options {
    fn default() -> Options {
        Options {
            backend: Backend::Auto,
            selector: AdapterSelector::default(),
            cache: None,
            retune: false,
            verify_rate: 4096,
            min_gpu_batch: 64,
            capacity: 16384,
            threads: 0,
            device_retries: 2,
            device_retries_total: 8,
            poll_timeout: None,
            inject_loss_on_open: 0,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct EngineReport {
    /// Odd candidates of at least 3 tested on the GPU. Degenerate inputs resolved by the verdict
    /// rules (below 2, even, 2) are in neither count. A bucket counts once, when its GPU run
    /// succeeded (a run lost with its device and redone does not count twice).
    pub gpu_tests: u64,
    /// Odd candidates of at least 3 tested on the CPU as the primary backend.
    pub cpu_tests: u64,
    /// CPU re-runs of GPU verdicts (every GPU PRP plus the sampled composites).
    pub rechecks: u64,
    /// GPU said PRP, CPU said composite (the CPU verdict was used).
    pub prp_disagreements: u64,
    /// GPU said composite, CPU said PRP (the CPU verdict was used).
    pub composite_disagreements: u64,
    /// Buckets that fell back to the CPU after a GPU error: an ordinary `Err` from autotuning or the
    /// runner, or the bucket on which the device could not be recovered (the later buckets of an
    /// engine that has switched to the CPU are not counted again).
    pub gpu_errors: u64,
    /// Seconds spent in autotuning (inside `verdicts`).
    pub tune_secs: f64,
    /// Limb classes autotuned (or loaded from the cache) by this engine: one per distinct
    /// `tune::limb_class` among the GPU buckets, however many bit lengths share it.
    pub tuned_classes: u64,
    /// Device-level GPU failures seen on a device in use: a lost device, a timed-out wait (also
    /// counted in `poll_timeouts`) or an uncaptured wgpu error. Each one drops the device.
    pub device_losses: u64,
    /// Devices successfully re-opened after a device-level failure.
    pub device_recoveries: u64,
    /// Device-level failures that were a GPU wait longer than the poll timeout.
    pub poll_timeouts: u64,
}

/// What one GPU attempt at a bucket gave.
enum GpuRun {
    Done(Vec<bool>),
    /// Use the CPU: no GPU, a small bucket, or an ordinary error (already counted).
    Cpu,
    /// The device failed; re-open it and re-run the bucket.
    DeviceFailure(String),
}

/// What to do with a healthy re-opened device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Reopen {
    /// The same adapter: keep the tuned configurations.
    Keep,
    /// A different adapter: forget the tuned configurations (autotune and self-test again).
    Retune,
    /// Unusable under `Backend::Auto` (no subgroups): CPU for the rest of the process.
    CpuOnly,
    /// Unusable under `Backend::Gpu` (a different adapter without subgroups): `verdicts` fails.
    Fail,
}

/// The rule for a re-opened device. It mirrors `Engine::open`, where `Auto` refuses an adapter
/// without subgroups and `Gpu` accepts any adapter. Under `Gpu`, a re-open that silently switched
/// to a different adapter without subgroups is refused: that is not the GPU the run was started on.
fn reopen_decision(backend: Backend, same_adapter: bool, has_subgroups: bool) -> Reopen {
    match (has_subgroups, same_adapter, backend) {
        (false, _, Backend::Auto | Backend::Cpu) => Reopen::CpuOnly,
        (false, false, Backend::Gpu) => Reopen::Fail,
        (false, true, Backend::Gpu) | (true, true, _) => Reopen::Keep,
        (true, false, _) => Reopen::Retune,
    }
}

/// How a recovery attempt ended.
enum Recovery {
    Recovered,
    CpuOnly,
    Fatal(String),
}

/// Whether `e`, returned by a GPU call on `gpu`, means the device itself failed.
fn device_failure(gpu: &Gpu, e: &str) -> bool {
    gpu.check_health().is_err() || is_device_error(e)
}

pub struct Engine {
    opts: Options,
    gpu: Option<Gpu>,
    /// Why no GPU is in use (open failure, or no subgroups for `Auto`).
    gpu_err: Option<String>,
    /// Set when a device-level failure could not be recovered: the engine is CPU-only from then on.
    cpu_only: Option<String>,
    /// One runner per bit length (a runner bakes its exact `bits`); `None` records that building it
    /// failed.
    runners: HashMap<u32, Option<Runner>>,
    /// The tuned kernel shape per limb class (`tune::limb_class`), shared by every bit length in the
    /// class; the error is why tuning the class failed. Kept across device re-opens.
    tuned: HashMap<u32, Result<KernelConfig, String>>,
    report: EngineReport,
    /// Freshly opened devices still to destroy (`Options::inject_loss_on_open`).
    inject_on_open: u32,
    /// Device re-opens left in the current `verdicts` call.
    reopens_left: u32,
    /// Device re-opens over the engine's life (capped by `Options::device_retries_total`).
    reopens_done: u32,
    /// Test hook: the next recovery treats the re-opened device as a different adapter.
    inject_adapter_change: bool,
}

impl Engine {
    pub fn open(opts: Options) -> Engine {
        let mut e = Engine {
            inject_on_open: opts.inject_loss_on_open,
            reopens_left: opts.device_retries,
            reopens_done: 0,
            inject_adapter_change: false,
            opts,
            gpu: None,
            gpu_err: None,
            cpu_only: None,
            runners: HashMap::new(),
            tuned: HashMap::new(),
            report: EngineReport::default(),
        };
        if e.opts.backend != Backend::Cpu {
            match e.open_device() {
                Ok(g) if e.opts.backend == Backend::Auto && !g.has_subgroups => e.gpu_err = Some(format!("{} lacks subgroup support", g.describe())),
                Ok(g) => e.gpu = Some(g),
                Err(why) => e.gpu_err = Some(why),
            }
        }
        e
    }

    /// Opens the selected adapter with the engine's poll timeout (and the test hook applied).
    fn open_device(&mut self) -> Result<Gpu, String> {
        let mut gpu = Gpu::open(&self.opts.selector)?;
        if let Some(timeout) = self.opts.poll_timeout {
            gpu.set_poll_timeout(timeout);
        }
        if self.inject_on_open > 0 {
            self.inject_on_open -= 1;
            eprintln!("bigmont: test hook: destroying the freshly opened device");
            gpu.device.destroy();
        }
        Ok(gpu)
    }

    fn cpu_threads(&self) -> usize {
        if self.opts.threads == 0 { std::thread::available_parallelism().map_or(1, |n| n.get()) } else { self.opts.threads }
    }

    pub fn describe(&self) -> String {
        match (&self.gpu, &self.gpu_err, &self.cpu_only) {
            (Some(g), _, _) => format!("GPU: {} [{:?}]", g.info.name, g.info.backend),
            (None, _, Some(why)) => format!("CPU: {} threads (GPU disabled: {why})", self.cpu_threads()),
            (None, Some(why), None) if self.opts.backend == Backend::Gpu => format!("GPU unavailable: {why}"),
            (None, Some(why), None) => format!("CPU: {} threads (GPU unavailable: {why})", self.cpu_threads()),
            (None, None, None) => format!("CPU: {} threads", self.cpu_threads()),
        }
    }

    /// Whether a GPU is in use (the backend resolved to the GPU rather than the CPU, and the engine
    /// has not switched to the CPU after an unrecoverable device failure).
    pub fn gpu_active(&self) -> bool {
        self.gpu.is_some()
    }

    /// Test only: destroys the current device (wgpu then reports it lost), as if the driver had
    /// lost it. The next GPU bucket detects the loss and recovers.
    #[doc(hidden)]
    pub fn inject_device_loss(&mut self) {
        if let Some(g) = &self.gpu {
            eprintln!("bigmont: test hook: destroying the current device");
            g.device.destroy();
        }
    }

    /// Test only: the next GPU wait behaves as if `Device::poll` had panicked, as wgpu 30 does on
    /// a real driver-side loss (see `Gpu::inject_poll_panic`).
    #[doc(hidden)]
    pub fn inject_poll_panic(&mut self) {
        if let Some(g) = &self.gpu {
            eprintln!("bigmont: test hook: the next GPU wait panics");
            g.inject_poll_panic();
        }
    }

    /// Test only: the next recovery treats the re-opened device as a different adapter (one host
    /// cannot portably make `Gpu::open` return another adapter).
    #[doc(hidden)]
    pub fn inject_adapter_change(&mut self) {
        self.inject_adapter_change = true;
    }

    pub fn report(&self) -> EngineReport {
        self.report.clone()
    }

    fn cpu(&mut self, nums: &[BigUint]) -> Vec<bool> {
        self.report.cpu_tests += nums.len() as u64;
        bigmont_cpu::fermat_batch(nums, self.opts.threads)
    }

    /// Verdicts (0 composite, 1 probable prime) for `ns`, in input order.
    pub fn verdicts(&mut self, ns: &[BigUint]) -> Result<Vec<u8>, String> {
        if self.opts.backend == Backend::Gpu {
            if let Some(why) = &self.gpu_err {
                return Err(format!("GPU backend requested but unavailable: {why}"));
            }
        }
        self.reopens_left = self.opts.device_retries;
        let two = BigUint::from(2u32);
        let mut out = vec![0u8; ns.len()];
        let mut buckets: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
        for (i, n) in ns.iter().enumerate() {
            if n < &two || !n.bit(0) {
                out[i] = u8::from(n == &two);
            } else {
                buckets.entry(u32::try_from(n.bits()).unwrap_or(u32::MAX)).or_default().push(i);
            }
        }
        let mut composites_seen = 0u64;
        for (bits, idx) in buckets {
            let nums: Vec<BigUint> = idx.iter().map(|&i| ns[i].clone()).collect();
            let verdicts = loop {
                match self.try_gpu(bits, &nums, &mut composites_seen) {
                    GpuRun::Done(v) => break v,
                    GpuRun::Cpu => break self.cpu(&nums),
                    // Re-run the whole bucket on the re-opened device, or answer it on the CPU.
                    GpuRun::DeviceFailure(why) => match self.recover(bits, &why) {
                        Recovery::Recovered => {}
                        Recovery::CpuOnly => {
                            self.report.gpu_errors += 1;
                            break self.cpu(&nums);
                        }
                        Recovery::Fatal(e) => return Err(e),
                    },
                }
            };
            for (&i, v) in idx.iter().zip(verdicts) {
                out[i] = u8::from(v);
            }
        }
        Ok(out)
    }

    /// After a device-level failure on the `bits`-bit bucket: drops the device and every runner and
    /// re-opens the device, at most `reopens_left` times in this call and `device_retries_total`
    /// times over the engine's life. On the same adapter the tuned configurations are kept; on a
    /// different one they are dropped, so the next bucket autotunes and self-tests again, and the
    /// `Auto`/subgroup rule of `Engine::open` applies (see `reopen_decision`).
    fn recover(&mut self, bits: u32, why: &str) -> Recovery {
        self.report.device_losses += 1;
        if self.gpu.as_ref().is_some_and(Gpu::is_poisoned) {
            self.report.poll_timeouts += 1;
        }
        eprintln!("bigmont: GPU device failure on the {bits}-bit bucket: {why}");
        // Runners first: they hold the old device's buffers. A poisoned or driver-lost Gpu leaks
        // its device on drop instead of blocking (see `Gpu`).
        self.runners.clear();
        let lost_info = self.gpu.take().map(|g| g.info.clone());
        while self.reopens_left > 0 && self.reopens_done < self.opts.device_retries_total {
            self.reopens_left -= 1;
            self.reopens_done += 1;
            let g = match self.open_device() {
                Ok(g) => g,
                Err(e) => {
                    eprintln!("bigmont: cannot re-open the GPU device: {e}");
                    continue;
                }
            };
            if let Err(e) = g.check_health() {
                eprintln!("bigmont: the re-opened GPU device is unusable: {e}");
                continue;
            }
            let forced_change = std::mem::take(&mut self.inject_adapter_change);
            let same = !forced_change && lost_info.as_ref().is_some_and(|old| bigmont_core::gpu::same_adapter(old, &g.info));
            match reopen_decision(self.opts.backend, same, g.has_subgroups) {
                Reopen::Keep => {}
                Reopen::Retune => {
                    eprintln!("bigmont: the re-opened GPU is a different adapter ({}); tuning again", g.describe());
                    self.tuned.clear();
                }
                Reopen::CpuOnly => {
                    let reason = format!("the re-opened GPU {} lacks subgroup support (after: {why})", g.describe());
                    eprintln!("bigmont: {reason}; using the CPU for the rest of this process");
                    self.cpu_only = Some(reason);
                    return Recovery::CpuOnly;
                }
                Reopen::Fail => {
                    let reason = format!("the GPU device failed ({why}) and re-opening it gave a different adapter without subgroup support: {}", g.describe());
                    eprintln!("bigmont: {reason}");
                    self.gpu_err = Some(reason.clone());
                    return Recovery::Fatal(format!("GPU backend requested but unavailable: {reason}"));
                }
            }
            self.report.device_recoveries += 1;
            eprintln!("bigmont: GPU device re-opened ({}); re-running the {bits}-bit bucket", g.describe());
            self.gpu = Some(g);
            return Recovery::Recovered;
        }
        let limit = if self.reopens_done >= self.opts.device_retries_total {
            format!("the engine's limit of {} re-opens (device_retries_total) is used up", self.opts.device_retries_total)
        } else {
            format!("not recovered within {} re-opens in this call (device_retries)", self.opts.device_retries)
        };
        let reason = format!("device failure, {limit}: {why}");
        eprintln!("bigmont: GPU {reason}; using the CPU for the rest of this process");
        self.cpu_only = Some(reason);
        Recovery::CpuOnly
    }

    /// Runs the bucket on the GPU with integrity checks. `Cpu` means "use the CPU": the GPU is not
    /// in play for this bucket, or autotuning / `Runner::new` / `Runner::fermat` returned an
    /// ordinary `Err` (counted in `gpu_errors`, then the bucket is redone on the CPU).
    /// `DeviceFailure` means the device itself failed (lost, a timed-out wait, an uncaptured wgpu
    /// error): nothing of the attempt is kept, a failed tuning is not cached, and the caller
    /// re-opens the device and calls again.
    fn try_gpu(&mut self, bits: u32, nums: &[BigUint], composites_seen: &mut u64) -> GpuRun {
        let Some(gpu) = self.gpu.as_ref() else { return GpuRun::Cpu };
        // The only size rule: a bucket under min_gpu_batch is not worth the GPU. Any bit length from 2
        // (n = 3) up is fine; buckets only hold odd n >= 3.
        if nums.len() < self.opts.min_gpu_batch {
            return GpuRun::Cpu;
        }
        if !self.runners.contains_key(&bits) {
            let t = Instant::now();
            // Tune once per limb class, not per bit length; the Runner is still per exact length.
            let class = tune::limb_class(bits);
            if !self.tuned.contains_key(&class) {
                let tuned = tune::autotune(gpu, bits, self.opts.cache.as_deref(), self.opts.retune).map(|t| t.cfg);
                if let Err(e) = &tuned
                    && device_failure(gpu, e)
                {
                    self.report.tune_secs += t.elapsed().as_secs_f64();
                    return GpuRun::DeviceFailure(e.clone());
                }
                self.report.tuned_classes += 1;
                self.tuned.insert(class, tuned);
            }
            let built = match &self.tuned[&class] {
                // The class's kernel shape (tpi, wg) at this exact length: same limb count by construction.
                Ok(tuned) => {
                    let cfg = KernelConfig::for_bits(bits, tuned.tpi, tuned.wg, Variant::Unrolled);
                    Runner::new(gpu, cfg, bits, self.opts.capacity)
                }
                Err(e) => Err(e.clone()),
            };
            self.report.tune_secs += t.elapsed().as_secs_f64();
            if let Err(e) = &built {
                if device_failure(gpu, e) {
                    return GpuRun::DeviceFailure(e.clone());
                }
                eprintln!("bigmont: GPU unusable for {bits}-bit candidates, using the CPU: {e}");
            }
            self.runners.insert(bits, built.ok());
        }
        let Some(runner) = self.runners.get_mut(&bits).and_then(Option::as_mut) else {
            self.report.gpu_errors += 1;
            return GpuRun::Cpu;
        };
        let mut verdicts = match runner.fermat(gpu, nums) {
            Ok(v) => v,
            Err(e) if device_failure(gpu, &e) => return GpuRun::DeviceFailure(e),
            Err(e) => {
                eprintln!("bigmont: GPU error on a {bits}-bit bucket of {} candidates, using the CPU: {e}", nums.len());
                // The runner's buffers may be in an unknown state: drop it, later buckets use the CPU.
                self.runners.insert(bits, None);
                self.report.gpu_errors += 1;
                return GpuRun::Cpu;
            }
        };
        self.report.gpu_tests += nums.len() as u64;

        // Integrity: every PRP, and every verify_rate-th composite (counted across the call).
        let mut recheck = Vec::new();
        for (i, &v) in verdicts.iter().enumerate() {
            if v {
                recheck.push(i);
            } else {
                *composites_seen += 1;
                if self.opts.verify_rate > 0 && *composites_seen % u64::from(self.opts.verify_rate) == 0 {
                    recheck.push(i);
                }
            }
        }
        let subset: Vec<BigUint> = recheck.iter().map(|&i| nums[i].clone()).collect();
        let truth = bigmont_cpu::fermat_batch(&subset, self.opts.threads);
        self.report.rechecks += subset.len() as u64;
        for (&i, cpu) in recheck.iter().zip(truth) {
            if verdicts[i] != cpu {
                if verdicts[i] {
                    self.report.prp_disagreements += 1;
                    eprintln!("bigmont: GPU/CPU disagreement: GPU said PRP, CPU says composite ({bits}-bit candidate); using the CPU verdict");
                } else {
                    self.report.composite_disagreements += 1;
                    eprintln!("bigmont: GPU/CPU disagreement: GPU said composite, CPU says PRP ({bits}-bit candidate); using the CPU verdict");
                }
                verdicts[i] = cpu;
            }
        }
        GpuRun::Done(verdicts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_same_adapter_keeps_its_tuning_and_a_different_one_is_retuned() {
        for backend in [Backend::Auto, Backend::Gpu] {
            assert_eq!(reopen_decision(backend, true, true), Reopen::Keep, "{backend:?}");
            assert_eq!(reopen_decision(backend, false, true), Reopen::Retune, "{backend:?}");
        }
    }

    #[test]
    fn a_reopened_adapter_without_subgroups_follows_the_open_rules() {
        // Auto refuses an adapter without subgroups, as Engine::open does: CPU from then on.
        assert_eq!(reopen_decision(Backend::Auto, false, false), Reopen::CpuOnly);
        assert_eq!(reopen_decision(Backend::Auto, true, false), Reopen::CpuOnly);
        // Gpu: a silent switch to a different adapter without subgroups is an error; the same
        // adapter is what Engine::open already accepted.
        assert_eq!(reopen_decision(Backend::Gpu, false, false), Reopen::Fail);
        assert_eq!(reopen_decision(Backend::Gpu, true, false), Reopen::Keep);
    }
}
