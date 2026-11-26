## What and why

<!-- Link the issue: Fixes #123 -->

## Tested on

<!--
Results vary by GPU vendor, architecture and driver. For every PR, list what you
actually ran the tests on, and what you did not, so others can help fill the gaps.
Describe hardware by model only (no hostnames or usernames). "Not tested" is fine.

Required (checked by CI) when the PR touches crates/*/src, Cargo manifests or
browser kernels in web/. Docs-only and CI-only PRs can skip it. If a PR touches
those paths but cannot change behaviour (a pure refactor, a doc comment), instead
add one line anywhere in the description: "No hardware impact: <reason>".

In the Result column tick exactly one option per row by changing "[ ]" to "[x]".
A row you tick "tested" needs the GPU model, driver and OS/backend in the middle
column, and at least one row must be "tested". Leave the middle column empty for
the others.

Browser: if the change can reach a browser (bigmont-core, the 13-bit kernel
generators, bigmont-wasm, web/, the wgpu pin), run the pages in web/ in a WebGPU
browser (see AGENTS.md) and tick "tested" with the browser, version and OS. Tick
"not tested" if you could not, and "n/a" only if the change cannot reach a browser.
-->

| Vendor | GPU model, driver, OS / backend | Result (tick one) |
|---|---|---|
| NVIDIA | | [ ] tested  [ ] not tested |
| AMD | | [ ] tested  [ ] not tested |
| Intel | | [ ] tested  [ ] not tested |
| Apple | | [ ] tested  [ ] not tested |
| Software (lavapipe, WARP) | | [ ] tested  [ ] not tested |
| Browser (WebGPU) | | [ ] tested  [ ] not tested  [ ] n/a |

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
