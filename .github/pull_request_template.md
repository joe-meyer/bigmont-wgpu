## What and why

<!-- Link the issue: Fixes #123 -->

## Tested on

<!--
Results vary by GPU vendor, architecture and driver. For every PR, list what you
actually ran the tests on, and what you did not, so others can help fill the gaps.
Describe hardware by model only (no hostnames or usernames). "Untested" is fine.
-->

| Vendor | GPU model, driver, OS / backend | Result |
|---|---|---|
| NVIDIA | | tested / not tested |
| AMD | | tested / not tested |
| Intel | | tested / not tested |
| Apple | | tested / not tested |
| Software (lavapipe, WARP) | | tested / not tested |
| Browser (WebGPU) | | tested / not tested / n/a |

Help wanted testing on: <!-- vendors you could not cover -->

## Checklist

- [ ] `cargo fmt --all`, `cargo clippy --workspace --all-targets --all-features -- -D warnings`, and `cargo test --workspace -- --test-threads=1` pass locally
- [ ] Kernel/arithmetic changes: differential test vs the CPU reference included
- [ ] "Tested on" lists the vendors tested and not tested
- [ ] Performance claims include GPU, driver, OS and the command run
- [ ] `CHANGELOG.md` updated under `[Unreleased]` (if user-visible)
- [ ] No machine-identifying details or secrets in code, docs or results

## AI assistance

- [ ] No AI assistance, or only trivial autocomplete
- [ ] AI-assisted: tool/model used: `________` (also in an `Assisted-by:` commit trailer)
- [ ] I have read, run and understand all of this code and can explain it
