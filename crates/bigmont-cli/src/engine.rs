//! The driver engine: chooses GPU or CPU, buckets candidates by bit length, and cross-checks the
//! GPU against the CPU reference.
//!
//! Verdicts: `n < 2` or even `n != 2` is 0 (composite), `n == 2` is 1, everything else comes from
//! a backend. The engine returns 2 (a Gerbicz alarm) only with `Options::gec`, for a candidate whose
//! checked run (on the GPU or on the CPU) or whose CPU re-check failed the check.
//!
//! Device-level GPU failures (a lost device, a GPU wait longer than the poll timeout, an
//! uncaptured wgpu error; see `bigmont_core::gpu::Gpu::check_health`) are recovered: the device
//! and its runners are dropped, the device is re-opened and the whole bucket is re-run. After
//! `Options::device_retries` re-opens within one `verdicts` call the engine gives up on the GPU and
//! answers on the CPU for the rest of the process. Verdicts are correct in every path: a failed
//! GPU run contributes nothing, the bucket is redone from scratch.

use crate::tune;
use bigmont_core::gpu::{AdapterSelector, Gpu, is_device_error};
use bigmont_core::kernel::KernelConfig;
use bigmont_core::runner::{Runner, Verdict};
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
    /// Which limb schemes the tuner may pick from (`--limbs`): `Auto` (the default) takes whichever
    /// the adapter runs fastest, `Only(s)` forces one. A forced scheme has its own cache entries.
    pub scheme: tune::SchemeChoice,
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
    /// Runs every Fermat test under the Gerbicz error check: GPU buckets through
    /// `Runner::fermat_checked_async`, CPU buckets (`Backend::Cpu`, buckets smaller than `min_gpu_batch`,
    /// and buckets answered on the CPU after a GPU error or a device that could not be recovered)
    /// and the CPU re-checks of GPU verdicts through `bigmont_cpu::fermat_batch_gec`. A candidate
    /// whose computation fails the check gets verdict 2 instead of a verdict, is counted in
    /// `EngineReport::gec_alarms` and stays 2 (the caller retests or arbitrates it). A GPU alarm is
    /// not re-checked on the CPU; a CPU re-check that alarms turns the candidate into 2 too, since
    /// neither computation can then be trusted. Off by default; with it off nothing changes.
    pub gec: bool,
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
            scheme: tune::SchemeChoice::Auto,
            verify_rate: 4096,
            min_gpu_batch: 64,
            capacity: 16384,
            threads: 0,
            device_retries: 2,
            device_retries_total: 8,
            poll_timeout: None,
            gec: false,
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
    /// Extra self-tests of a retargeted kernel: the tuner self-tests a shape at its class's
    /// representative length, and a shorter length can build a program with fewer limbs (32-bit
    /// kernels only), which is self-tested once per (limb class, limb count) before it is used.
    pub retarget_self_tests: u64,
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
    /// Candidates given verdict 2 (a Gerbicz alarm) with `Options::gec`: by a checked GPU run, a
    /// checked CPU run, or a checked CPU re-check of a GPU verdict. Each candidate counts once.
    pub gec_alarms: u64,
    /// Candidates tested on the CPU as the primary backend with the Gerbicz check (`Options::gec`);
    /// at most `cpu_tests`.
    pub cpu_checked: u64,
    /// CPU re-checks of GPU verdicts that ran with the Gerbicz check (`Options::gec`); at most
    /// `rechecks`.
    pub rechecks_checked: u64,
}

fn to_verdict(prp: bool) -> Verdict {
    if prp {
        Verdict::Prp
    } else {
        Verdict::Composite
    }
}

fn to_verdicts(v: &[bool]) -> Vec<Verdict> {
    v.iter().map(|&b| to_verdict(b)).collect()
}

fn from_checked(c: bigmont_cpu::Checked) -> Verdict {
    match c {
        bigmont_cpu::Checked::Composite => Verdict::Composite,
        bigmont_cpu::Checked::Prp => Verdict::Prp,
        bigmont_cpu::Checked::Alarm => Verdict::Alarm,
    }
}

/// CPU verdicts for `nums` on `threads` threads (0 = all cores): with `gec` from the checked batch
/// `bigmont_cpu::fermat_batch_gec` (an alarm is `Verdict::Alarm`), adding the candidates it tested to
/// `checked`; otherwise from the plain `bigmont_cpu::fermat_batch`, which never alarms.
fn cpu_verdicts(gec: bool, threads: usize, nums: &[BigUint], checked: &mut u64) -> Vec<Verdict> {
    if gec {
        *checked += nums.len() as u64;
        bigmont_cpu::fermat_batch_gec(nums, threads)
            .into_iter()
            .map(from_checked)
            .collect()
    } else {
        to_verdicts(&bigmont_cpu::fermat_batch(nums, threads))
    }
}

/// Counts the alarms in `verdicts` (the `bits`-bit candidates of one bucket, or of its CPU
/// re-checks) in `report.gec_alarms` and logs one line for them, `bigmont: N GEC alarms among M
/// B-bit candidates` followed by `context`. Returns that line, or `None` (no count, no line) when
/// there is no alarm.
fn count_alarms(
    report: &mut EngineReport,
    bits: u32,
    verdicts: &[Verdict],
    context: &str,
) -> Option<String> {
    let alarms = verdicts.iter().filter(|&&v| v == Verdict::Alarm).count();
    if alarms == 0 {
        return None;
    }
    report.gec_alarms += alarms as u64;
    let line = format!(
        "bigmont: {alarms} GEC alarms among {} {bits}-bit candidates{context}",
        verdicts.len()
    );
    eprintln!("{line}");
    Some(line)
}

/// What one GPU attempt at a bucket gave.
enum GpuRun {
    Done(Vec<Verdict>),
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

/// Self-tests `cfg` (the tuned shape `tuned` retargeted to some bit length of limb class `class`) when
/// it is a different program from the one the tuner self-tested, i.e. when its limb count differs
/// (13-bit kernels never differ within a class). It runs once per (class, limb count) and the outcome
/// is remembered in `done`, except when the device itself failed (that says nothing about `cfg`).
/// The test length is the largest the retargeted kernel supports, capped at the class's
/// representative length: `min(class_bits(class), cfg.r_bits() - 4)`. A 32-bit kernel has at least 4
/// limbs, so that length is at least 124 bits, above the self-test's 32-bit minimum.
fn retarget_check(
    gpu: &Gpu,
    done: &mut HashMap<(u32, u32), Result<(), String>>,
    report: &mut EngineReport,
    class: u32,
    tuned: &KernelConfig,
    cfg: &KernelConfig,
) -> Result<(), String> {
    if cfg.limbs == tuned.limbs {
        return Ok(());
    }
    if let Some(known) = done.get(&(class, cfg.limbs)) {
        return known.clone();
    }
    let test_bits = tune::class_bits(class).min(cfg.r_bits().saturating_sub(4));
    let outcome = tune::self_test(gpu, *cfg, test_bits);
    if let Err(e) = &outcome {
        // A device-level failure says nothing about `cfg`: not a verdict, not counted (the bucket is
        // redone after recovery and this check runs again).
        if device_failure(gpu, e) {
            return outcome;
        }
        // `e` already names the configuration (see `tune::self_test`).
        eprintln!("bigmont: a retargeted kernel failed its self-test at {test_bits} bits: {e}");
    }
    report.retarget_self_tests += 1;
    done.insert((class, cfg.limbs), outcome.clone());
    outcome
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
    /// The outcome of the extra self-test of a retargeted kernel, per (limb class, limb count) of the
    /// retargeted kernel (see `retarget_check`). Cleared with `tuned`.
    retarget_checked: HashMap<(u32, u32), Result<(), String>>,
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
            retarget_checked: HashMap::new(),
            report: EngineReport::default(),
        };
        if e.opts.backend != Backend::Cpu {
            match e.open_device() {
                Ok(g) if e.opts.backend == Backend::Auto && !g.has_subgroups => {
                    e.gpu_err = Some(format!("{} lacks subgroup support", g.describe()))
                }
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
        if self.opts.threads == 0 {
            std::thread::available_parallelism().map_or(1, |n| n.get())
        } else {
            self.opts.threads
        }
    }

    pub fn describe(&self) -> String {
        match (&self.gpu, &self.gpu_err, &self.cpu_only) {
            (Some(g), _, _) => format!("GPU: {} [{:?}]", g.info.name, g.info.backend),
            (None, _, Some(why)) => {
                format!("CPU: {} threads (GPU disabled: {why})", self.cpu_threads())
            }
            (None, Some(why), None) if self.opts.backend == Backend::Gpu => {
                format!("GPU unavailable: {why}")
            }
            (None, Some(why), None) => format!(
                "CPU: {} threads (GPU unavailable: {why})",
                self.cpu_threads()
            ),
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

    /// Answers the `bits`-bit bucket `nums` on the CPU (checked with `Options::gec`; alarms are
    /// counted in `gec_alarms`, with one log line per bucket).
    fn cpu(&mut self, bits: u32, nums: &[BigUint]) -> Vec<Verdict> {
        self.report.cpu_tests += nums.len() as u64;
        let verdicts = cpu_verdicts(
            self.opts.gec,
            self.opts.threads,
            nums,
            &mut self.report.cpu_checked,
        );
        count_alarms(&mut self.report, bits, &verdicts, " tested on the CPU");
        verdicts
    }

    /// Fails if the GPU backend was demanded (`--backend gpu`) but could not be opened. `verdicts`
    /// checks this too; callers that may never call it (an empty input) check it up front.
    pub fn require_backend(&self) -> Result<(), String> {
        if self.opts.backend == Backend::Gpu
            && let Some(why) = &self.gpu_err
        {
            return Err(format!("GPU backend requested but unavailable: {why}"));
        }
        Ok(())
    }

    /// Verdicts (0 composite, 1 probable prime, and with `Options::gec` 2 for a Gerbicz alarm) for
    /// `ns`, in input order.
    pub fn verdicts(&mut self, ns: &[BigUint]) -> Result<Vec<u8>, String> {
        self.require_backend()?;
        self.reopens_left = self.opts.device_retries;
        let two = BigUint::from(2u32);
        let mut out = vec![0u8; ns.len()];
        let mut buckets: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
        for (i, n) in ns.iter().enumerate() {
            if n < &two || !n.bit(0) {
                out[i] = u8::from(n == &two);
            } else {
                buckets
                    .entry(u32::try_from(n.bits()).unwrap_or(u32::MAX))
                    .or_default()
                    .push(i);
            }
        }
        let mut composites_seen = 0u64;
        for (bits, idx) in buckets {
            let nums: Vec<BigUint> = idx.iter().map(|&i| ns[i].clone()).collect();
            let verdicts = loop {
                match self.try_gpu(bits, &nums, &mut composites_seen) {
                    GpuRun::Done(v) => break v,
                    GpuRun::Cpu => break self.cpu(bits, &nums),
                    // Re-run the whole bucket on the re-opened device, or answer it on the CPU.
                    GpuRun::DeviceFailure(why) => match self.recover(bits, &why) {
                        Recovery::Recovered => {}
                        Recovery::CpuOnly => {
                            self.report.gpu_errors += 1;
                            break self.cpu(bits, &nums);
                        }
                        Recovery::Fatal(e) => return Err(e),
                    },
                }
            };
            for (&i, v) in idx.iter().zip(verdicts) {
                out[i] = v.byte();
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
            let same = !forced_change
                && lost_info
                    .as_ref()
                    .is_some_and(|old| bigmont_core::gpu::same_adapter(old, &g.info));
            match reopen_decision(self.opts.backend, same, g.has_subgroups) {
                Reopen::Keep => {}
                Reopen::Retune => {
                    eprintln!(
                        "bigmont: the re-opened GPU is a different adapter ({}); tuning again",
                        g.describe()
                    );
                    self.tuned.clear();
                    self.retarget_checked.clear();
                }
                Reopen::CpuOnly => {
                    let reason = format!(
                        "the re-opened GPU {} lacks subgroup support (after: {why})",
                        g.describe()
                    );
                    eprintln!("bigmont: {reason}; using the CPU for the rest of this process");
                    self.cpu_only = Some(reason);
                    return Recovery::CpuOnly;
                }
                Reopen::Fail => {
                    let reason = format!(
                        "the GPU device failed ({why}) and re-opening it gave a different adapter without subgroup support: {}",
                        g.describe()
                    );
                    eprintln!("bigmont: {reason}");
                    self.gpu_err = Some(reason.clone());
                    return Recovery::Fatal(format!(
                        "GPU backend requested but unavailable: {reason}"
                    ));
                }
            }
            self.report.device_recoveries += 1;
            eprintln!(
                "bigmont: GPU device re-opened ({}); re-running the {bits}-bit bucket",
                g.describe()
            );
            self.gpu = Some(g);
            return Recovery::Recovered;
        }
        let limit = if self.reopens_done >= self.opts.device_retries_total {
            format!(
                "the engine's limit of {} re-opens (device_retries_total) is used up",
                self.opts.device_retries_total
            )
        } else {
            format!(
                "not recovered within {} re-opens in this call (device_retries)",
                self.opts.device_retries
            )
        };
        let reason = format!("device failure, {limit}: {why}");
        eprintln!("bigmont: GPU {reason}; using the CPU for the rest of this process");
        self.cpu_only = Some(reason);
        Recovery::CpuOnly
    }

    /// Runs the bucket on the GPU with integrity checks. `Cpu` means "use the CPU": the GPU is not
    /// in play for this bucket, or autotuning / `Runner::new_async` / `Runner::fermat_async` returned an
    /// ordinary `Err` (counted in `gpu_errors`, then the bucket is redone on the CPU).
    /// `DeviceFailure` means the device itself failed (lost, a timed-out wait, an uncaptured wgpu
    /// error): nothing of the attempt is kept, a failed tuning is not cached, and the caller
    /// re-opens the device and calls again.
    fn try_gpu(&mut self, bits: u32, nums: &[BigUint], composites_seen: &mut u64) -> GpuRun {
        let Some(gpu) = self.gpu.as_ref() else {
            return GpuRun::Cpu;
        };
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
                let tuned = tune::autotune(
                    gpu,
                    bits,
                    self.opts.cache.as_deref(),
                    self.opts.retune,
                    self.opts.scheme,
                )
                .map(|t| t.cfg);
                if let Err(e) = &tuned
                    && device_failure(gpu, e)
                {
                    self.report.tune_secs += t.elapsed().as_secs_f64();
                    return GpuRun::DeviceFailure(e.clone());
                }
                self.report.tuned_classes += 1;
                self.tuned.insert(class, tuned);
            }
            let built = match self.tuned[&class].clone() {
                // The class's kernel shape (scheme, tpi, wg) at this exact length. A 32-bit kernel's
                // limb count can be smaller than at the class's representative length, and that is a
                // different program from the one the tuner self-tested: check it once first.
                Ok(tuned) => {
                    let cfg = tuned.retarget(bits);
                    retarget_check(
                        gpu,
                        &mut self.retarget_checked,
                        &mut self.report,
                        class,
                        &tuned,
                        &cfg,
                    )
                    .and_then(|()| {
                        pollster::block_on(Runner::new_async(gpu, cfg, bits, self.opts.capacity))
                    })
                    .and_then(|mut runner| {
                        if self.opts.gec {
                            pollster::block_on(runner.enable_gec_async(gpu))?;
                        }
                        Ok(runner)
                    })
                }
                Err(e) => Err(e),
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
        let run = if self.opts.gec {
            pollster::block_on(runner.fermat_checked_async(gpu, nums))
        } else {
            pollster::block_on(runner.fermat_async(gpu, nums)).map(|v| to_verdicts(&v))
        };
        let verdicts = match run {
            Ok(v) => v,
            Err(e) if device_failure(gpu, &e) => return GpuRun::DeviceFailure(e),
            Err(e) => {
                eprintln!(
                    "bigmont: GPU error on a {bits}-bit bucket of {} candidates, using the CPU: {e}",
                    nums.len()
                );
                // The runner's buffers may be in an unknown state: drop it, later buckets use the CPU.
                self.runners.insert(bits, None);
                self.report.gpu_errors += 1;
                return GpuRun::Cpu;
            }
        };
        self.report.gpu_tests += nums.len() as u64;

        GpuRun::Done(cpu_integrity_pass(
            bits,
            nums,
            verdicts,
            &self.opts,
            composites_seen,
            &mut self.report,
        ))
    }
}

/// [`integrity_pass`] with the real CPU as the re-check: `bigmont_cpu`'s checked batch with
/// `opts.gec` (counted in `report.rechecks_checked`), the plain one otherwise, on `opts.threads`.
fn cpu_integrity_pass(
    bits: u32,
    nums: &[BigUint],
    verdicts: Vec<Verdict>,
    opts: &Options,
    composites_seen: &mut u64,
    report: &mut EngineReport,
) -> Vec<Verdict> {
    let mut checked = 0u64;
    let out = integrity_pass(
        bits,
        nums,
        verdicts,
        opts.verify_rate,
        composites_seen,
        report,
        |subset| cpu_verdicts(opts.gec, opts.threads, subset, &mut checked),
    );
    report.rechecks_checked += checked;
    out
}

/// The integrity pass over one GPU bucket's verdicts: every `Verdict::Prp` and every `verify_rate`-th
/// composite (counted across the whole `verdicts` call by `composites_seen`; 0 = never) is re-run
/// through `cpu_check`, and a disagreement is counted and replaced by the CPU verdict. A
/// `Verdict::Alarm` is counted in `report.gec_alarms` (one log line per bucket), is never passed to
/// `cpu_check` and stays an alarm: the caller retests or arbitrates it. `cpu_check` answers one
/// verdict per candidate of the slice it gets, and alarms only when it is a checked run
/// (`Options::gec`). A re-check that alarms makes the candidate an alarm (counted in `gec_alarms`,
/// not a disagreement): neither computation can then be trusted.
fn integrity_pass(
    bits: u32,
    nums: &[BigUint],
    mut verdicts: Vec<Verdict>,
    verify_rate: u32,
    composites_seen: &mut u64,
    report: &mut EngineReport,
    cpu_check: impl FnOnce(&[BigUint]) -> Vec<Verdict>,
) -> Vec<Verdict> {
    let mut recheck = Vec::new();
    for (i, &v) in verdicts.iter().enumerate() {
        match v {
            Verdict::Prp => recheck.push(i),
            Verdict::Composite => {
                *composites_seen += 1;
                if verify_rate > 0 && (*composites_seen).is_multiple_of(u64::from(verify_rate)) {
                    recheck.push(i);
                }
            }
            Verdict::Alarm => {}
        }
    }
    count_alarms(report, bits, &verdicts, "");
    let subset: Vec<BigUint> = recheck.iter().map(|&i| nums[i].clone()).collect();
    let truth = cpu_check(&subset);
    report.rechecks += subset.len() as u64;
    count_alarms(
        report,
        bits,
        &truth,
        " re-checked on the CPU; they get verdict 2",
    );
    for (&i, cpu) in recheck.iter().zip(truth) {
        if cpu == Verdict::Alarm {
            verdicts[i] = Verdict::Alarm;
            continue;
        }
        if verdicts[i] != cpu {
            if verdicts[i] == Verdict::Prp {
                report.prp_disagreements += 1;
                eprintln!(
                    "bigmont: GPU/CPU disagreement: GPU said PRP, CPU says composite ({bits}-bit candidate); using the CPU verdict"
                );
            } else {
                report.composite_disagreements += 1;
                eprintln!(
                    "bigmont: GPU/CPU disagreement: GPU said composite, CPU says PRP ({bits}-bit candidate); using the CPU verdict"
                );
            }
            verdicts[i] = cpu;
        }
    }
    verdicts
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    /// Runs `integrity_pass` with a fake CPU that calls exactly the candidates in `primes` probable
    /// primes. Returns the verdicts, the report and the candidates the fake CPU was asked about.
    fn pass(
        verdicts: &[Verdict],
        primes: &[u32],
        verify_rate: u32,
        seen_before: u64,
    ) -> (Vec<Verdict>, EngineReport, Vec<u32>, u64) {
        pass_with(verdicts, primes, &[], verify_rate, seen_before)
    }

    /// `pass` whose fake CPU also raises a GEC alarm on the candidates in `alarms`.
    fn pass_with(
        verdicts: &[Verdict],
        primes: &[u32],
        alarms: &[u32],
        verify_rate: u32,
        seen_before: u64,
    ) -> (Vec<Verdict>, EngineReport, Vec<u32>, u64) {
        // Candidate i is the number 1000 + i, so a recheck names its position.
        let nums: Vec<BigUint> = (0..verdicts.len() as u32)
            .map(|i| BigUint::from(1000 + i))
            .collect();
        let asked = RefCell::new(Vec::new());
        let mut report = EngineReport::default();
        let mut seen = seen_before;
        let out = integrity_pass(
            12,
            &nums,
            verdicts.to_vec(),
            verify_rate,
            &mut seen,
            &mut report,
            |subset| {
                subset
                    .iter()
                    .map(|n| {
                        let id = u32::try_from(n).unwrap() - 1000;
                        asked.borrow_mut().push(id);
                        if alarms.contains(&id) {
                            Alarm
                        } else {
                            to_verdict(primes.contains(&id))
                        }
                    })
                    .collect()
            },
        );
        (out, report, asked.into_inner(), seen)
    }

    use Verdict::{Alarm, Composite, Prp};

    #[test]
    fn an_alarm_stays_an_alarm_is_counted_once_and_is_never_rechecked() {
        // The fake CPU calls everything prime: an alarm that were re-checked would be overruled.
        let (out, report, asked, seen) = pass(&[Composite, Alarm, Prp, Alarm], &[0, 1, 2, 3], 1, 0);
        assert_eq!(out, [Prp, Alarm, Prp, Alarm]);
        assert_eq!(report.gec_alarms, 2, "counted once each");
        assert_eq!(asked, [0, 2], "an alarm is never passed to the CPU");
        assert_eq!(
            seen, 1,
            "an alarm is not a composite for the sampling counter"
        );
        assert_eq!(
            report.composite_disagreements, 1,
            "the composite at 0 was wrong"
        );
        assert_eq!(report.prp_disagreements, 0);
        // Alarms only, with every composite sampled: nothing reaches the CPU at all.
        let (out, report, asked, _) = pass(&[Alarm, Alarm], &[], 1, 0);
        assert_eq!(out, [Alarm, Alarm]);
        assert_eq!((report.gec_alarms, report.rechecks), (2, 0));
        assert!(asked.is_empty());
    }

    #[test]
    fn a_prp_the_cpu_calls_composite_becomes_composite() {
        let (out, report, asked, _) = pass(&[Prp, Prp], &[1], 0, 0);
        assert_eq!(out, [Composite, Prp]);
        assert_eq!(report.prp_disagreements, 1);
        assert_eq!(report.composite_disagreements, 0);
        assert_eq!(asked, [0, 1]);
    }

    #[test]
    fn a_sampled_composite_the_cpu_calls_prime_becomes_prp_and_the_sampling_spans_calls() {
        // Every 3rd composite of the whole call is sampled; two were already seen, so the first
        // composite of this bucket (index 0) is the 3rd and the last one (index 4) is the 6th.
        let verdicts = [Composite, Composite, Alarm, Composite, Composite];
        let (out, report, asked, seen) = pass(&verdicts, &[0, 4], 3, 2);
        assert_eq!(seen, 6);
        assert_eq!(asked, [0, 4], "the 3rd and 6th composites of the call");
        assert_eq!(out, [Prp, Composite, Alarm, Composite, Prp]);
        assert_eq!(report.composite_disagreements, 2);
        assert_eq!(report.prp_disagreements, 0);
        // The same bucket with a fresh counter samples the 3rd composite only (index 3).
        let (_, _, asked, seen) = pass(&verdicts, &[], 3, 0);
        assert_eq!((asked, seen), (vec![3], 4));
    }

    #[test]
    fn rechecks_counts_exactly_the_checked_candidates() {
        // PRPs at 0 and 4, composites at 1, 2, 3, 5 (sampled: every 2nd, so 2 and 5), an alarm at 6.
        let verdicts = [Prp, Composite, Composite, Composite, Prp, Composite, Alarm];
        let (out, report, asked, _) = pass(&verdicts, &[0, 4], 2, 0);
        assert_eq!(asked, [0, 2, 4, 5]);
        assert_eq!(report.rechecks, 4);
        assert_eq!(
            (report.prp_disagreements, report.composite_disagreements),
            (0, 0)
        );
        assert_eq!(out, verdicts);
        assert_eq!(report.gec_alarms, 1);
    }

    #[test]
    fn verify_rate_0_rechecks_only_the_prps() {
        let verdicts = [Composite, Prp, Composite, Alarm, Prp];
        let (out, report, asked, seen) = pass(&verdicts, &[1, 4, 0], 0, 0);
        assert_eq!(asked, [1, 4]);
        assert_eq!(report.rechecks, 2);
        assert_eq!(out, verdicts, "an unchecked composite is left alone");
        assert_eq!(seen, 2, "composites are still counted");
    }

    #[test]
    fn a_cpu_recheck_alarm_turns_the_candidate_into_an_alarm_counted_once() {
        // A GPU PRP (0) and a sampled GPU composite (2) whose CPU re-checks alarm: neither side can be
        // trusted, so both become alarms. The PRP at 1 is confirmed.
        let (out, report, asked, _) = pass_with(&[Prp, Prp, Composite], &[1], &[0, 2], 1, 0);
        assert_eq!(out, [Alarm, Prp, Alarm]);
        assert_eq!(asked, [0, 1, 2]);
        assert_eq!(report.gec_alarms, 2, "each alarm counted once");
        assert_eq!(report.rechecks, 3);
        assert_eq!(
            (report.prp_disagreements, report.composite_disagreements),
            (0, 0),
            "an alarm is not a disagreement"
        );
        // A GPU alarm next to a re-check alarm: two alarms, both counted once.
        let (out, report, asked, _) = pass_with(&[Alarm, Prp], &[], &[1], 0, 0);
        assert_eq!(out, [Alarm, Alarm]);
        assert_eq!(asked, [1]);
        assert_eq!(report.gec_alarms, 2);
    }

    #[test]
    fn the_cpu_runs_the_checked_batch_exactly_when_gec_is_on() {
        // Primes, composites, base-2 pseudoprimes (341, 561) and a 127-bit Mersenne prime.
        let mut nums: Vec<BigUint> = [5u32, 7, 9, 341, 561, 1105, 65_537]
            .map(BigUint::from)
            .to_vec();
        nums.push((BigUint::from(1u32) << 127u32) - 1u32);
        let want: Vec<Verdict> = nums
            .iter()
            .map(|n| to_verdict(bigmont_oracle::fermat_base2(n)))
            .collect();
        let mut checked = 0;
        assert_eq!(cpu_verdicts(false, 2, &nums, &mut checked), want);
        assert_eq!(checked, 0, "gec off: the checked batch never runs");
        assert_eq!(cpu_verdicts(true, 2, &nums, &mut checked), want);
        assert_eq!(
            checked,
            nums.len() as u64,
            "gec on: every candidate is checked"
        );
        // A clean checked run never alarms, so the counter is what tells the two batches apart.
    }

    /// The candidates 7 (prime), 9, 11 (prime) and 15 with GPU verdicts Prp, Composite, Composite
    /// (wrong) and Alarm, through the real CPU re-check with every composite sampled.
    fn real_recheck(gec: bool) -> (Vec<Verdict>, EngineReport) {
        let nums = [7u32, 9, 11, 15].map(BigUint::from);
        let opts = Options {
            gec,
            verify_rate: 1,
            threads: 2,
            ..Options::default()
        };
        let mut report = EngineReport::default();
        let mut seen = 0;
        let out = cpu_integrity_pass(
            4,
            &nums,
            vec![Prp, Composite, Composite, Alarm],
            &opts,
            &mut seen,
            &mut report,
        );
        (out, report)
    }

    #[test]
    fn the_real_recheck_is_the_checked_one_exactly_when_gec_is_on() {
        for gec in [false, true] {
            let (out, report) = real_recheck(gec);
            assert_eq!(out, [Prp, Composite, Prp, Alarm], "gec {gec}");
            assert_eq!(report.rechecks, 3, "the alarm is not re-checked");
            assert_eq!(report.composite_disagreements, 1);
            assert_eq!(report.gec_alarms, 1, "only the GPU alarm");
            assert_eq!(
                report.rechecks_checked,
                if gec { 3 } else { 0 },
                "gec {gec}"
            );
            assert_eq!(report.cpu_checked, 0, "re-checks are not primary CPU tests");
        }
    }

    #[test]
    fn count_alarms_counts_each_alarm_once_and_logs_one_line() {
        let mut report = EngineReport::default();
        assert_eq!(
            count_alarms(
                &mut report,
                12,
                &[Prp, Alarm, Composite, Alarm],
                " tested on the CPU"
            )
            .as_deref(),
            Some("bigmont: 2 GEC alarms among 4 12-bit candidates tested on the CPU")
        );
        assert_eq!(report.gec_alarms, 2);
        assert_eq!(
            count_alarms(&mut report, 9, &[Alarm], "").as_deref(),
            Some("bigmont: 1 GEC alarms among 1 9-bit candidates")
        );
        assert_eq!(report.gec_alarms, 3);
        assert_eq!(count_alarms(&mut report, 12, &[Prp, Composite], "x"), None);
        assert_eq!(count_alarms(&mut report, 12, &[], "x"), None);
        assert_eq!(report.gec_alarms, 3, "no alarm, no count");
    }

    #[test]
    fn the_same_adapter_keeps_its_tuning_and_a_different_one_is_retuned() {
        for backend in [Backend::Auto, Backend::Gpu] {
            assert_eq!(
                reopen_decision(backend, true, true),
                Reopen::Keep,
                "{backend:?}"
            );
            assert_eq!(
                reopen_decision(backend, false, true),
                Reopen::Retune,
                "{backend:?}"
            );
        }
    }

    #[test]
    fn a_retargeted_kernel_is_self_tested_once_and_a_failure_is_remembered() {
        use bigmont_core::kernel::Variant;
        use bigmont_core::scheme::LimbScheme;
        let gpu = match Gpu::new() {
            Ok(g) => g,
            Err(e) => {
                assert!(
                    std::env::var("BIGMONT_ALLOW_SKIP").is_ok_and(|v| v == "1"),
                    "no adapter: {e} (set BIGMONT_ALLOW_SKIP=1 to allow skips)"
                );
                eprintln!("SKIP no adapter: {e}");
                return;
            }
        };
        let mut done = HashMap::new();
        let mut report = EngineReport::default();
        // Same limb count as the tuned shape: the same program, nothing to test.
        let tuned =
            KernelConfig::for_bits_scheme(LimbScheme::Limb13, 256, 8, 64, Variant::Unrolled);
        assert!(
            retarget_check(
                &gpu,
                &mut done,
                &mut report,
                20,
                &tuned,
                &tuned.retarget(252)
            )
            .is_ok()
        );
        assert_eq!(report.retarget_self_tests, 0);
        // A different limb count whose self-test fails (the workgroup size is invalid): the error is
        // returned, counted once and remembered for the same (class, limbs).
        let mut broken =
            KernelConfig::for_bits_scheme(LimbScheme::Limb32, 252, 8, 64, Variant::Unrolled);
        broken.wg = 300;
        let wider =
            KernelConfig::for_bits_scheme(LimbScheme::Limb32, 256, 8, 64, Variant::Unrolled);
        assert_ne!(broken.limbs, wider.limbs);
        let first = retarget_check(&gpu, &mut done, &mut report, 20, &wider, &broken);
        assert!(
            first.as_ref().is_err_and(|e| e.contains("workgroup")),
            "{first:?}"
        );
        let second = retarget_check(&gpu, &mut done, &mut report, 20, &wider, &broken);
        assert_eq!(first, second);
        assert_eq!(
            report.retarget_self_tests, 1,
            "the failed self-test is not repeated"
        );
    }

    #[test]
    fn a_reopened_adapter_without_subgroups_follows_the_open_rules() {
        // Auto refuses an adapter without subgroups, as Engine::open does: CPU from then on.
        assert_eq!(
            reopen_decision(Backend::Auto, false, false),
            Reopen::CpuOnly
        );
        assert_eq!(reopen_decision(Backend::Auto, true, false), Reopen::CpuOnly);
        // Gpu: a silent switch to a different adapter without subgroups is an error; the same
        // adapter is what Engine::open already accepted.
        assert_eq!(reopen_decision(Backend::Gpu, false, false), Reopen::Fail);
        assert_eq!(reopen_decision(Backend::Gpu, true, false), Reopen::Keep);
    }
}
