#![allow(dead_code)]
pub mod faults;
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
    log_skip(&msg);
}

/// Appends a tab-separated `binary`, `test`, `reason` line to the file `BIGMONT_SKIP_LOG` names, if
/// set. CI summarises that file so reduced coverage is visible instead of reading as a pass.
fn log_skip(why: &str) {
    use std::io::Write;
    let Some(path) = std::env::var_os("BIGMONT_SKIP_LOG") else {
        return;
    };
    let exe = std::env::current_exe().ok();
    let bin = exe
        .as_deref()
        .and_then(|p| p.file_stem())
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    // Cargo names a test binary `<name>-<16 hex digits>`; keep the name.
    let bin = match bin.rsplit_once('-') {
        Some((name, hash)) if hash.len() == 16 && hash.bytes().all(|b| b.is_ascii_hexdigit()) => {
            name
        }
        _ => &bin,
    };
    // libtest runs every test on a thread named after it.
    let test = std::thread::current().name().unwrap_or("?").to_string();
    let why = why.replace(['\t', '\n'], " ");
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        // One write call, so concurrent test binaries cannot interleave a line.
        let _ = f.write_all(format!("{bin}\t{test}\t{why}\n").as_bytes());
    }
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
    if cfg.needs_subgroups()
        && let Err(e) = gpu.check_subgroups(cfg.tpi)
    {
        skip(format!("{cfg:?}: {e}"));
        return None;
    }
    eprintln!(
        "adapter: {} ({:?}), config {cfg:?}",
        gpu.info.name, gpu.info.backend
    );
    Some(gpu)
}
