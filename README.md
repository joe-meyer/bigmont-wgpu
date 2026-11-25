# bigmont-wgpu

A portable GPU engine for the heavy step of a Beacon-style prime-gap hunt: base-2 Fermat tests of about
2,029-bit numbers, on any GPU that [wgpu](https://wgpu.rs) supports (AMD, NVIDIA, Intel, Apple), with a pure-Rust
CPU fallback and kernels written so they can also run in a browser through WebGPU.

**Status: experimental.** The engine is correct against independent references and runs on four GPUs, but it is
not yet as fast as the CUDA implementation it is modelled on (about half its speed on the one card where both were
measured), and some platforms are untested. See [Limits](#limits).

## What it does

- Reads and writes Beacon's `G2C1` candidate files and `G2V1` verdict files, so `bigmont` can stand in for Beacon's
  `g1_fermat_v3` driver on those files.
- Runs the base-2 Fermat test with cooperative Montgomery arithmetic: several GPU lanes share one number, using
  subgroup shuffles for the cross-lane work (the shape of NVIDIA's CGBN, written in WGSL).
- Measures instead of guessing. An autotuner tries every kernel shape the adapter can run (limb width, lanes per
  number, workgroup size), checks each one against a CPU reference on that adapter, and caches the fastest.
- Never trusts the GPU alone: every GPU probable prime is re-checked on the CPU, a sample of composites is
  re-checked too, and a device that is lost mid-run is re-opened (or the work moves to the CPU).

### Two limb schemes

| Scheme | Limbs | Needs | Runs in a browser |
|---|---|---|---|
| 13-bit | 13-bit limbs in `u32`, deferred carries | subgroups | yes |
| 32-bit | 32-bit limbs, `u64` products | subgroups and 64-bit integers in shaders (`SHADER_INT64`) | no (WGSL has no 64-bit integers there) |

Which one is faster depends on the GPU, so it is not hard-coded: `--limbs auto` (the default) measures both. On the
cards tried so far the tuner picks 32-bit limbs on an RTX 4070 Laptop and a Quadro P4000 and 13-bit limbs on an
RX 6700 XT.

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

## Measured so far

2,029-bit numbers, the engine end to end (`bigmont CAND OUT --backend gpu`, tests per second, excluding the one-time
tuning). One to three runs per cell on one machine each, so treat differences of a few percent as noise. Details,
method and limits: [`docs/superpowers/results/2026-11-25-limb-factory-validation.md`](docs/superpowers/results/2026-11-25-limb-factory-validation.md).

| GPU | 13-bit limbs | 32-bit limbs | Auto picks |
|---|---|---|---|
| RTX 4070 Laptop (Vulkan) | ~35.7k | ~61.1k | 32-bit |
| Quadro P4000 (Vulkan, shared card) | ~5.3k | ~7.8k | 32-bit |
| RX 6700 XT (Vulkan) | ~31.7k | ~20.8k | 13-bit |

For reference, Beacon's CUDA driver measured 132k tests/s (116k for one pass over the file) on the same RTX 4070
Laptop, so the engine is at roughly half of CUDA's speed there. On the CPU the pure-Rust backend does about 2k
tests/s on a 16-thread machine. Verdicts from the GPU engine, the CPU backend and Beacon's own driver were
byte-identical on Beacon's 32,000-candidate canary file.

## In a browser

`web/fermat.html` runs the 13-bit Fermat kernels in a browser's WebGPU and checks every verdict against a BigInt
oracle. Validated so far in Chrome 156 on an RX 6700 XT only (all kernels compile and pass; about 34k tests/s at
2,029 bits there). See [`web/README.md`](web/README.md) and
[`docs/superpowers/results/2026-11-25-browser-fermat-validation.md`](docs/superpowers/results/2026-11-25-browser-fermat-validation.md).

## Layout

| Path | What |
|---|---|
| `crates/bigmont-oracle` | CPU reference arithmetic for both limb schemes (used by tests and the self-test) |
| `crates/bigmont-core` | the wgpu engine: kernel generators (`unrolled.rs`, `limb32.rs`), `Runner`, adapter and device handling |
| `crates/bigmont-cpu` | pure-Rust 64-bit-limb Montgomery Fermat test (the CPU backend) |
| `crates/bigmont-cli` | the `bigmont` driver: file formats, autotuner and cache, the engine that buckets and cross-checks |
| `web/` | browser test pages and generated kernels |
| `docs/superpowers/` | design, plans, measurements and the open-items list |

## Testing

```sh
cargo test --release --workspace --no-fail-fast -- --test-threads=1
```

The GPU tests are strict: a missing adapter or capability fails the test. `BIGMONT_ALLOW_SKIP=1` turns that into a
loud skip. `BIGMONT_ADAPTER=SUBSTR` and `BIGMONT_BACKEND=Vulkan|Dx12|...` pick the device. Run the suite in one
process with `--test-threads=1`; opening many devices at once is what exhausted the Linux NVIDIA driver in earlier
runs.

## Limits

- Not yet CUDA speed: about half of Beacon's driver on the RTX 4070 Laptop, the only card where both were run.
- Tested GPUs: RTX 4070 Laptop, Quadro P4000, RX 6700 XT, Radeon 780M (and software Vulkan). Intel and Apple GPUs,
  and DX12/Metal behaviour of the 32-bit scheme, are untested. Browser testing covers one Chrome build on one GPU.
- The Linux NVIDIA driver 580 occasionally resets the card for reasons unrelated to these kernels (a few percent of
  processes in a stress loop); the engine survives it by re-opening the device, but the real driver-level failure has
  not been seen under the recovery code.
- Not implemented: Beacon's `--ee`, `--bins`, `--chunk`, `--growth`, `--marker`, `--inject` flags; the Gerbicz check.
- Small numbers (tens of bits) run slower on the GPU than on the CPU because of per-call overhead.

The open items are tracked in the issues; the older notes are in `docs/superpowers/phase1-backlog.md`.

## Background and credits

The design follows [Beacon](https://github.com/djuk8/beacon)'s CUDA Fermat driver (which builds on NVIDIA's
[CGBN](https://github.com/NVlabs/CGBN) library) as its reference for interface and performance. This repository contains none of their
source files. The kernels are generated WGSL; the tests compare them with independent big-integer references.

## License

[MIT](LICENSE).
