## What and why

<!-- Link the issue: Fixes #123 -->

## Checklist

- [ ] `cargo fmt --all`, `cargo clippy --workspace --all-targets --all-features -- -D warnings`, and `cargo test --workspace -- --test-threads=1` pass locally
- [ ] Kernel/arithmetic changes: differential test vs the CPU reference included
- [ ] Performance claims include GPU, driver, OS and the command run
- [ ] `CHANGELOG.md` updated under `[Unreleased]` (if user-visible)
- [ ] No machine-identifying details or secrets in code, docs or results

## AI assistance

- [ ] No AI assistance, or only trivial autocomplete
- [ ] AI-assisted: tool/model used: `________` (also in an `Assisted-by:` commit trailer)
- [ ] I have read, run and understand all of this code and can explain it
