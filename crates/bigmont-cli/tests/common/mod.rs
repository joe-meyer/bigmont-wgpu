//! Helpers shared by the GPU-dependent integration tests.
//!
//! The engine's GPU backend needs an adapter *and* subgroup support: the tuned
//! kernels cooperate through subgroup shuffles. A machine with no adapter (a CI
//! runner), or with only a software adapter that lacks subgroups (for example
//! Windows' Basic Render Driver), cannot run those tests. They are strict by
//! default and fail; `BIGMONT_ALLOW_SKIP=1` turns that into a loud skip.
#![allow(dead_code)] // each test crate uses a different subset

use bigmont_core::gpu::Gpu;

pub fn allow_skip() -> bool {
    std::env::var("BIGMONT_ALLOW_SKIP").is_ok_and(|v| v == "1")
}

/// Whether the engine's GPU backend can run here. Returns `false` (the caller should return) when it
/// cannot and `BIGMONT_ALLOW_SKIP=1` is set; fails the test when it cannot and the variable is unset.
pub fn usable_gpu() -> bool {
    match Gpu::new() {
        Ok(g) if g.has_subgroups => true,
        Ok(_) => skip("the adapter lacks subgroup support, so no tuned kernel can run"),
        Err(e) => skip(&format!("no adapter: {e}")),
    }
}

fn skip(why: &str) -> bool {
    record_skip(why);
    false
}

/// A GPU-dependent test cannot run here: fails the test unless `BIGMONT_ALLOW_SKIP=1`, otherwise
/// prints `SKIP <why>` and, when `BIGMONT_SKIP_LOG` names a file, appends a tab-separated
/// `binary`, `test`, `reason` line to it. CI summarises that file so reduced coverage is visible instead of reading as a pass.
pub fn record_skip(why: &str) {
    assert!(
        allow_skip(),
        "{why} (set BIGMONT_ALLOW_SKIP=1 to allow skips)"
    );
    note_skip(why);
}

/// Prints `SKIP <why>` and logs it like `record_skip`, without failing the test: for a case that is
/// skipped even in a strict run (part of a test's matrix does not apply on this adapter).
pub fn note_skip(why: &str) {
    eprintln!("SKIP {why}");
    log_skip(why);
}

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
