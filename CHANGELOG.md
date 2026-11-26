# Changelog

All notable changes are documented here. Format based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Minimum supported Rust version declared: 1.90 (`rust-version` in the workspace manifest), the highest minimum among our dependencies; the early-exit lock needs 1.89.
- CI, release automation, `AGENTS.md`, and AI contribution guidelines.
- In-driver early exit for the `bigmont` driver: `--ee SIZES` (with `--bins`, `--chunk`, `--growth`, `--marker`) tests candidates in waves and retires a k at its first probable prime, with output byte-identical to the CUDA driver this one replaces on its fixtures (#16). `--lock FILE` now takes a real exclusive lock in every mode: plain runs and `canary` used to accept and ignore it, and now wait for any other holder. `--marker` accepts 3 to 255, as the CUDA driver does.
- `bigmont-core` compiles for wasm32 and runs in a browser's WebGPU through a new async API (`Gpu::open_async`, `Runner::new_async`, `Runner::fermat_async`, `run_async`; the blocking forms remain on native targets) and the new `bigmont-wasm` crate. `web/rust.html` checks it against a BigInt oracle. `bigmont-core` and `bigmont-cli` now depend on wgpu at a trunk commit (pinned by `rev`), because wgpu's web backend only exposes subgroups since gfx-rs/wgpu#10265, which is not in 30.0.2. Return to a crates.io release once one contains it (see `web/README.md`).
- Opt-in Gerbicz error check (`--gec`, `Options::gec`): GPU buckets run with the Gerbicz-Li check over blocks of 40 ladder iterations, plus a recomputation of the last `(bits - 1) % 40` iterations that belong to no block. A candidate that fails gets verdict byte 2 (retest it), counted in the summary line (`GEC alarms`, `EngineReport::gec_alarms`) and, with `--ee`, in `gec_alarm=`. Off by default; with it off nothing changes. Closes #17.

## [0.1.0-alpha.1] - TBD

First alpha release.

[Unreleased]: https://github.com/joe-meyer/bigmont-wgpu/compare/v0.1.0-alpha.1...HEAD
[0.1.0-alpha.1]: https://github.com/joe-meyer/bigmont-wgpu/releases/tag/v0.1.0-alpha.1
