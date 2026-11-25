# Changelog

All notable changes are documented here. Format based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- CI, release automation, `AGENTS.md`, and AI contribution guidelines.
- `bigmont-core` compiles for wasm32 and runs in a browser's WebGPU through a new async API (`Gpu::open_async`, `Runner::new_async`, `Runner::fermat_async`, `run_async`; the blocking forms remain on native targets) and the new `bigmont-wasm` crate. `web/rust.html` checks it against a BigInt oracle. Running on the GPU needs a one-line stopgap patch to wgpu 30.0.2, whose web backend does not yet expose subgroups; this is fixed on wgpu trunk (gfx-rs/wgpu#10265), so drop the patch when upgrading to the next release (see `web/README.md`).

## [0.1.0-alpha.1] - TBD

First alpha release.

[Unreleased]: https://github.com/joe-meyer/bigmont-wgpu/compare/v0.1.0-alpha.1...HEAD
[0.1.0-alpha.1]: https://github.com/joe-meyer/bigmont-wgpu/releases/tag/v0.1.0-alpha.1
