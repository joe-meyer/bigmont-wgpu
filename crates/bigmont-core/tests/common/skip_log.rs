//! Records a skipped GPU test in the file named by `BIGMONT_SKIP_LOG`, if set.
//!
//! The one copy of this code: `bigmont-core`'s `tests/common/mod.rs` declares it as a module and
//! `bigmont-cli`'s `tests/common/mod.rs` includes this same file with `#[path]`. CI summarises the
//! file (`.github/scripts/skip_summary.py`) so reduced GPU coverage is visible instead of reading
//! as a pass. The log is only ever appended to, so delete it before a local run that you want to
//! count.

/// Appends a tab-separated `binary`, `test`, `reason` line to the file `BIGMONT_SKIP_LOG` names.
/// Does nothing when the variable is unset, and ignores write errors: a bad log path must not
/// fail a test.
pub fn log_skip(why: &str) {
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
