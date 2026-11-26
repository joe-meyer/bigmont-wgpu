//! Helpers shared by the GPU-dependent integration tests.
//!
//! The engine's GPU backend needs an adapter *and* subgroup support: the tuned
//! kernels cooperate through subgroup shuffles. A machine with no adapter (a CI
//! runner), or with only a software adapter that lacks subgroups (for example
//! Windows' Basic Render Driver), cannot run those tests. They are strict by
//! default and fail; `BIGMONT_ALLOW_SKIP=1` turns that into a loud skip.
#![allow(dead_code)] // each test crate uses a different subset

use bigmont_core::gpu::Gpu;

// The one copy of the skip-log writer, shared with bigmont-core's tests.
#[path = "../../../bigmont-core/tests/common/skip_log.rs"]
mod skip_log;
use skip_log::log_skip;

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
/// `binary`, `test`, `reason` line to it (see `skip_log.rs`). CI summarises that file so reduced
/// coverage is visible instead of reading as a pass.
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
