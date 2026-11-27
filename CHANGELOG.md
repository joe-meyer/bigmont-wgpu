# Changelog

All notable changes are documented here. Format based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- The autotuner times every candidate on the engine's batch size (16,384 instances) instead of a fixed number of workgroups, which had given small workgroups fewer instances and made it favour workgroup 256. On an RTX 4070 Laptop the tuned engine runs about 6% faster end to end at 2,029 bits (workgroup 64 or 128 is picked); an RX 6700 XT is unchanged. The tuning cache key gains a measurement-generation field (`m2`), so lines written by earlier versions are a miss and each adapter is re-tuned once on first use (no manual `bigmont tune` needed). Refs #2.
- On x86-64 CPUs that report BMI2 and ADX, the CPU backend's squaring and reduction rows run in a `mulx`/`adcx`/`adox` assembly routine chosen at run time (about 1.65x on one thread on Zen 4 and Zen 3 over the portable code); every other CPU runs the safe Rust loop. This is the crate's only `unsafe`. `Mont::portable()` forces the portable kernel and `Mont::uses_adx()` reports the choice. Refs #40.
- **Breaking (library API):** `bigmont-core` is async-only. The blocking twins are removed: `Runner::new`, `Runner::new_with_source`, `Runner::fermat`, `Runner::enable_gec`, `Runner::enable_gec_with_source`, `Runner::fermat_checked`, `gpu::run`, `gpu::run_mul` and `Gpu::list_adapters`. Use the `_async` forms (`Gpu::list_adapters` is now `list_adapters_async`) and block with `pollster::block_on` where you need to. `Gpu::new`, `Gpu::open` (it holds the open lock), `Gpu::wait` and `Gpu::read_back` stay.
- CPU backend (`bigmont-cpu`) is about 1.6x faster at 2,029 bits: `Mont::sqr` is now a dedicated squaring with a separate Montgomery reduction instead of `mul(a, a)`, the multiply loops no longer carry per-limb bounds checks, and the multiply-accumulate step is cheaper (still no `unsafe`). **Breaking (library API):** `Mont::sqr` needs `2 w` limbs of scratch; allocate it with the new `Mont::scratch_len()` (`Mont::mul` still needs only `w + 2`). `fermat_base2`, `fermat_base2_gec` and the batch functions are unchanged. Refs #40.

### Added
- Symmetric squaring for the 32-bit cooperative kernel (`Squaring::Symmetric`, `KernelConfig::squaring`): the square is formed once, with about three quarters of the limb products, then reduced. The autotuner offers it at TPI 4 and 8 as an extra candidate next to the fused kernel (cache word `unrolled32sym`; older caches still read), so it is used only where it measures faster. At 2,029 bits on an RTX 4070 Laptop it is about 1.7% (TPI 4) and 3.6% (TPI 8) faster than the fused kernel end to end (5.6% and 3.8% in the kernel-only benchmark; the gap is not explained). On an RX 6700 XT the 32-bit scheme is 17% to 19% faster with it kernel-only, but the 13-bit scheme is still the one `auto` picks there, so the tuned engine is unchanged. The 13-bit scheme and multiply mode are unchanged. `fermat_stats` and `scheme_compare` gain options to force the squaring structure. Refs #2.
- Minimum supported Rust version declared: 1.90 (`rust-version` in the workspace manifest), the highest minimum among our dependencies; the early-exit lock needs 1.89.
- CI, release automation, `AGENTS.md`, and AI contribution guidelines.
- In-driver early exit for the `bigmont` driver: `--ee SIZES` (with `--bins`, `--chunk`, `--growth`, `--marker`) tests candidates in waves and retires a k at its first probable prime, with output byte-identical to the CUDA driver this one replaces on its fixtures (#16). `--lock FILE` now takes a real exclusive lock in every mode: plain runs and `canary` used to accept and ignore it, and now wait for any other holder. `--marker` accepts 3 to 255, as the CUDA driver does.
- `bigmont-core` compiles for wasm32 and runs in a browser's WebGPU through a new async API (`Gpu::open_async`, `Runner::new_async`, `Runner::fermat_async`, `run_async`; the blocking forms remain on native targets) and the new `bigmont-wasm` crate. `web/rust.html` checks it against a BigInt oracle. `bigmont-core` and `bigmont-cli` now depend on wgpu at a trunk commit (pinned by `rev`), because wgpu's web backend only exposes subgroups since gfx-rs/wgpu#10265, which is not in 30.0.2. Return to a crates.io release once one contains it (see `web/README.md`).
- Opt-in Gerbicz error check (`--gec`, `Options::gec`): every Fermat test runs with the Gerbicz-Li check over blocks of 40 ladder iterations, on the GPU and on the CPU (CPU buckets and the CPU re-checks of GPU verdicts; `bigmont_cpu::fermat_base2_gec`, `fermat_batch_gec`), plus a recomputation of the last `(bits - 1) % 40` iterations that belong to no block. A candidate that fails gets verdict byte 2 (retest it), counted in the summary line (`GEC alarms`, `EngineReport::gec_alarms`) and, with `--ee`, in `gec_alarm=`; a CPU re-check that alarms makes the candidate 2 as well. Off by default; with it off nothing changes. `Runner::new_with_source` and `Runner::enable_gec_with_source` build a runner from caller-supplied WGSL (the tests use them to simulate faulty devices). Closes #17.
- The checked runner's submissions adapt to the device like the plain path's (a block of 40 iterations is split when it would take longer than the target time), and `RunStats::dispatches` counts ladder submissions. `bigmont-wasm` gains `Engine::fermat_checked_hex` and `web/rust.html?gec=1` runs the Gerbicz check in a browser.

## [0.1.0-alpha.1] - TBD

First alpha release.

[Unreleased]: https://github.com/joe-meyer/bigmont-wgpu/compare/v0.1.0-alpha.1...HEAD
[0.1.0-alpha.1]: https://github.com/joe-meyer/bigmont-wgpu/releases/tag/v0.1.0-alpha.1
