#![allow(dead_code)]
use bigmont_core::gpu::Gpu;
use bigmont_core::kernel::KernelConfig;

/// Strict by default: a missing adapter or an unsupported configuration fails the test, so a skip
/// cannot masquerade as a pass. `BIGMONT_ALLOW_SKIP=1` turns those cases into loud skips.
/// `BIGMONT_REQUIRE_GPU=1` is accepted for compatibility (if both are set, strict wins).
pub fn skips_allowed() -> bool {
    let is_one = |name: &str| std::env::var(name).is_ok_and(|v| v == "1");
    is_one("BIGMONT_ALLOW_SKIP") && !is_one("BIGMONT_REQUIRE_GPU")
}

pub fn skip(msg: String) {
    if !skips_allowed() {
        panic!("{msg} (set BIGMONT_ALLOW_SKIP=1 to allow skips)");
    }
    eprintln!("SKIP {msg}");
}

/// The selected adapter, or `None` after a loud skip when it cannot run `cfg`.
pub fn open_gpu(cfg: KernelConfig) -> Option<Gpu> {
    let gpu = match Gpu::new() {
        Ok(g) => g,
        Err(e) => {
            skip(format!("{cfg:?}: no adapter: {e}"));
            return None;
        }
    };
    if cfg.needs_subgroups() {
        if let Err(e) = gpu.check_subgroups(cfg.tpi) {
            skip(format!("{cfg:?}: {e}"));
            return None;
        }
    }
    eprintln!(
        "adapter: {} ({:?}), config {cfg:?}",
        gpu.info.name, gpu.info.backend
    );
    Some(gpu)
}
