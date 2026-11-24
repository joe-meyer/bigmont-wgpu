//! The driver engine: chooses GPU or CPU, buckets candidates by bit length, and cross-checks the
//! GPU against the CPU reference.
//!
//! Verdicts: `n < 2` or even `n != 2` is 0 (composite), `n == 2` is 1, everything else comes from
//! a backend. The engine never returns 2.

use crate::tune;
use bigmont_core::gpu::{AdapterSelector, Gpu};
use bigmont_core::kernel::{KernelConfig, Variant};
use bigmont_core::runner::Runner;
use num_bigint::BigUint;
use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    /// GPU when one opens (and has subgroups), otherwise the CPU.
    Auto,
    /// GPU only: `verdicts` fails if no adapter opens. Buckets still fall back to the CPU when
    /// autotuning or the runner returns `Err` (counted in `EngineReport::gpu_errors`); a lost device
    /// or an uncaptured wgpu error is not handled that way (see `try_gpu`).
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
    /// overhead is not worth it). This is the only reason a bucket skips the GPU: the bit length
    /// itself does not matter, the GPU is used from 2-bit numbers (n = 3) up.
    pub min_gpu_batch: usize,
    /// Most candidates per GPU dispatch.
    pub capacity: u32,
    /// CPU worker threads, 0 = all cores.
    pub threads: usize,
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
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct EngineReport {
    /// Odd candidates of at least 3 tested on the GPU. Degenerate inputs resolved by the verdict
    /// rules (below 2, even, 2) are in neither count.
    pub gpu_tests: u64,
    /// Odd candidates of at least 3 tested on the CPU as the primary backend.
    pub cpu_tests: u64,
    /// CPU re-runs of GPU verdicts (every GPU PRP plus the sampled composites).
    pub rechecks: u64,
    /// GPU said PRP, CPU said composite (the CPU verdict was used).
    pub prp_disagreements: u64,
    /// GPU said composite, CPU said PRP (the CPU verdict was used).
    pub composite_disagreements: u64,
    /// Buckets that fell back to the CPU after a GPU error.
    pub gpu_errors: u64,
    /// Seconds spent in autotuning (inside `verdicts`).
    pub tune_secs: f64,
    /// Limb classes autotuned (or loaded from the cache) by this engine: one per distinct
    /// `tune::limb_class` among the GPU buckets, however many bit lengths share it.
    pub tuned_classes: u64,
}

pub struct Engine {
    opts: Options,
    gpu: Option<Gpu>,
    /// Why no GPU is in use (open failure, or no subgroups for `Auto`).
    gpu_err: Option<String>,
    /// One runner per bit length (a runner bakes its exact `bits`); `None` records that building it
    /// failed.
    runners: HashMap<u32, Option<Runner>>,
    /// The tuned kernel shape per limb class (`tune::limb_class`), shared by every bit length in the
    /// class; the error is why tuning the class failed.
    tuned: HashMap<u32, Result<KernelConfig, String>>,
    report: EngineReport,
}

impl Engine {
    pub fn open(opts: Options) -> Engine {
        let (gpu, gpu_err) = match opts.backend {
            Backend::Cpu => (None, None),
            backend => match Gpu::open(&opts.selector) {
                Ok(g) if backend == Backend::Auto && !g.has_subgroups => {
                    let why = format!("{} lacks subgroup support", g.describe());
                    (None, Some(why))
                }
                Ok(g) => (Some(g), None),
                Err(e) => (None, Some(e)),
            },
        };
        Engine { opts, gpu, gpu_err, runners: HashMap::new(), tuned: HashMap::new(), report: EngineReport::default() }
    }

    fn cpu_threads(&self) -> usize {
        if self.opts.threads == 0 { std::thread::available_parallelism().map_or(1, |n| n.get()) } else { self.opts.threads }
    }

    pub fn describe(&self) -> String {
        match (&self.gpu, &self.gpu_err) {
            (Some(g), _) => format!("GPU: {} [{:?}]", g.info.name, g.info.backend),
            (None, Some(why)) if self.opts.backend == Backend::Gpu => format!("GPU unavailable: {why}"),
            (None, Some(why)) => format!("CPU: {} threads (GPU unavailable: {why})", self.cpu_threads()),
            (None, None) => format!("CPU: {} threads", self.cpu_threads()),
        }
    }

    /// Whether a GPU is in use (the backend resolved to the GPU rather than the CPU).
    pub fn gpu_active(&self) -> bool {
        self.gpu.is_some()
    }

    pub fn report(&self) -> EngineReport {
        self.report.clone()
    }

    /// Verdicts (0 composite, 1 probable prime) for `ns`, in input order.
    pub fn verdicts(&mut self, ns: &[BigUint]) -> Result<Vec<u8>, String> {
        if self.opts.backend == Backend::Gpu {
            if let Some(why) = &self.gpu_err {
                return Err(format!("GPU backend requested but unavailable: {why}"));
            }
        }
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
            let verdicts = match self.try_gpu(bits, &nums, &mut composites_seen) {
                Some(v) => v,
                None => {
                    self.report.cpu_tests += nums.len() as u64;
                    bigmont_cpu::fermat_batch(&nums, self.opts.threads)
                }
            };
            for (&i, v) in idx.iter().zip(verdicts) {
                out[i] = u8::from(v);
            }
        }
        Ok(out)
    }

    /// Runs the bucket on the GPU with integrity checks. `None` means "use the CPU": the GPU is
    /// not in play for this bucket, or autotuning / `Runner::new` / `Runner::fermat` returned `Err`
    /// (counted in `gpu_errors`, then the bucket is redone on the CPU).
    ///
    /// Only those `Err` returns are caught. A lost device or an uncaptured wgpu error (an
    /// out-of-memory or validation error outside the runner's error scopes, e.g. in a submit,
    /// `write_buffer` or poll) is not: `Gpu::open` installs no device-lost or uncaptured-error
    /// handler, so wgpu's default uncaptured-error handler panics (a lost device reaches it with the
    /// next failing operation) and the run aborts without writing a verdict file (the verdict write
    /// is atomic, so no partial file is left).
    fn try_gpu(&mut self, bits: u32, nums: &[BigUint], composites_seen: &mut u64) -> Option<Vec<bool>> {
        let gpu = self.gpu.as_ref()?;
        // The only size rule: a bucket under min_gpu_batch is not worth the GPU. Any bit length from 2
        // (n = 3) up is fine; buckets only hold odd n >= 3.
        if nums.len() < self.opts.min_gpu_batch {
            return None;
        }
        if !self.runners.contains_key(&bits) {
            let t = Instant::now();
            // Tune once per limb class, not per bit length; the Runner is still per exact length.
            let class = tune::limb_class(bits);
            if !self.tuned.contains_key(&class) {
                self.report.tuned_classes += 1;
                let tuned = tune::autotune(gpu, bits, self.opts.cache.as_deref(), self.opts.retune).map(|t| t.cfg);
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
                eprintln!("bigmont: GPU unusable for {bits}-bit candidates, using the CPU: {e}");
            }
            self.runners.insert(bits, built.ok());
        }
        let Some(runner) = self.runners.get_mut(&bits).and_then(Option::as_mut) else {
            self.report.gpu_errors += 1;
            return None;
        };
        let mut verdicts = match runner.fermat(gpu, nums) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("bigmont: GPU error on a {bits}-bit bucket of {} candidates, using the CPU: {e}", nums.len());
                // The runner's buffers may be in an unknown state: drop it, later buckets use the CPU.
                self.runners.insert(bits, None);
                self.report.gpu_errors += 1;
                return None;
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
        Some(verdicts)
    }
}
