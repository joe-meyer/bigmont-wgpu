# bigmont-wgpu

A portable GPU engine for batched base-2 Fermat probable-prime tests of large integers (tested up to about 2,000
bits), on any GPU that [wgpu](https://wgpu.rs) supports, with a pure-Rust CPU fallback and kernels written so they can
also run in a browser through WebGPU. It was built for prime-gap searches, but the engine itself is generic: it takes
a list of odd integers and returns one verdict each.

**Status: experimental.** Verdicts are checked against independent references, but performance is still being tuned,
some platforms are untested, and the evidence behind several design choices is thin. Known gaps are tracked in the
[issue tracker](../../issues).

## Why this exists: compatibility first

The fastest implementations of this workload are usually written for one vendor, CUDA on NVIDIA for example, which
ties a search to specific hardware and an operating system. A search of this kind is a volunteer-scale job that can
use whatever hardware people already own: AMD and Intel cards, Apple silicon, laptops with integrated GPUs, and
eventually a browser tab.

This project exists to make that work run on all of them from one code base, and every design choice follows from it:

- **One kernel source, many backends.** The kernels are generated WGSL, run through wgpu on Vulkan, DX12 and Metal and
  through WebGPU in a browser, with no vendor-specific code paths.
- **Capabilities, not vendor names.** What a kernel needs (subgroups, 64-bit integers) is checked against what the
  adapter offers. The 32-bit-limb kernels need 64-bit integers, which WebGPU does not have, so they are native-only
  and the browser gets the 13-bit kernels instead.
- **Measured, not assumed.** Which kernel shape is fastest differs by GPU, driver and generation, so the autotuner
  measures on the actual device and verifies each candidate against a CPU reference before using it.
- **A CPU path that always works.** With no usable GPU the pure-Rust backend gives the same verdicts.
- **Compatible with existing tooling.** The `bigmont` driver reads and writes `G2C1` candidate files and `G2V1`
  verdict files, a binary format used by an existing CUDA-based search pipeline, so it can replace that pipeline's
  Fermat driver without changing the rest.

Raw speed matters, but it comes second: where portability and speed conflict, portability wins.

## What it does

- The engine (`bigmont-core`, `Runner::fermat`) takes plain big integers and returns Fermat verdicts. The `bigmont`
  command-line driver is one ready-made front end on top of it.
- Runs the base-2 Fermat test with cooperative Montgomery arithmetic: several GPU lanes share one number, using
  subgroup shuffles for the cross-lane work, written in WGSL.
- Measures instead of guessing. An autotuner tries every kernel shape the adapter can run (limb width, lanes per
  number, workgroup size), checks each one against a CPU reference on that adapter, and caches the fastest.
- Never trusts the GPU alone: every GPU probable prime is re-checked on the CPU, a sample of composites is
  re-checked too, and a device that is lost mid-run is re-opened (or the work moves to the CPU).

### Two limb schemes

| Scheme | Limbs | Needs | Runs in a browser |
|---|---|---|---|
| 13-bit | 13-bit limbs in `u32`, deferred carries | subgroups | yes |
| 32-bit | 32-bit limbs, `u64` products | subgroups and 64-bit integers in shaders (`SHADER_INT64`) | no (WGSL has no 64-bit integers there) |

Which one is faster depends on the GPU, so it is not hard-coded: `--limbs auto` (the default) measures both and keeps
the faster.

## Quick start

Requires a recent stable Rust (edition 2024) and a GPU with a Vulkan, DX12 or Metal driver. Without a usable GPU the
CPU backend runs instead.

```sh
cargo build --release

# Pick and verify the fastest kernel for this machine (writes a per-user cache)
target/release/bigmont tune --bits 2029

# Test a candidate file; writes a verdict file
target/release/bigmont cand.bin verdict.bin

# Same, with a known-answer check against an expected verdict file
target/release/bigmont canary cand.bin expected.bin

# Make a test file of random odd 2,029-bit numbers plus a few probable primes
cargo run --release -p bigmont-cli --example make_candidates -- --bits 2029 131072 7 cand.bin
```

Useful flags: `--backend auto|gpu|cpu`, `--adapter SUBSTR` and `--vk-backend SUBSTR` to pick a device, `--limbs
auto|13|32`, `--retune`, `--cache PATH | --no-cache`, `--verify-rate N`, `--repeat N`. Run `bigmont` with no
arguments for the full usage. The tuning cache lives at `$BIGMONT_CACHE`, else `%LOCALAPPDATA%\bigmont\tune.cache`
(Windows) or `$XDG_CACHE_HOME/bigmont/tune.cache` / `~/.cache/bigmont/tune.cache`.

## Measurements

Measurements, their methods and their limits are in
[`docs/superpowers/results/`](docs/superpowers/results/). They come from a handful of machines with one to three runs
per figure, so this README does not quote them.

## In a browser

`web/fermat.html` runs the generated 13-bit Fermat kernels from JavaScript and checks every verdict against a BigInt
oracle. `web/rust.html` does the same through the Rust engine itself: `bigmont-core` compiles to wasm32 (async API,
13-bit scheme only, no autotuner yet) and `bigmont-wasm` exposes it to the page. The engine currently depends on wgpu at a trunk commit, because the
browser's `subgroups` feature is only exposed by wgpu's WebGPU backend since gfx-rs/wgpu#10265 (not in the 30.0.2 release). See [`web/README.md`](web/README.md).

## Layout

| Path | What |
|---|---|
| `crates/bigmont-oracle` | CPU reference arithmetic for both limb schemes (used by tests and the self-test) |
| `crates/bigmont-core` | the wgpu engine: kernel generators (`unrolled.rs`, `limb32.rs`), `Runner`, adapter and device handling |
| `crates/bigmont-cpu` | pure-Rust 64-bit-limb Montgomery Fermat test (the CPU backend) |
| `crates/bigmont-cli` | the `bigmont` driver: file formats, autotuner and cache, the engine that buckets and cross-checks |
| `web/` | browser test pages and generated kernels |
| `docs/superpowers/` | design, plans, measurements and notes |

## Testing

```sh
cargo test --release --workspace --no-fail-fast -- --test-threads=1
```

The GPU tests are strict: a missing adapter or capability fails the test. `BIGMONT_ALLOW_SKIP=1` turns that into a
loud skip. `BIGMONT_ADAPTER=SUBSTR` and `BIGMONT_BACKEND=Vulkan|Dx12|...` pick the device. Run the suite in a single
process with `--test-threads=1`.

## Known gaps

Tracked as [issues](../../issues): performance against vendor-specific implementations, the evidence behind the limb
scheme choice, untested platforms, and missing features.

## Background and credits

The first use is prime-gap searching. The cooperative kernel design is shaped after NVIDIA's
[CGBN](https://github.com/NVlabs/CGBN) library; this repository contains none of its source files. The kernels are
generated WGSL; the tests compare them with independent big-integer references.

## License

[MIT](LICENSE).
