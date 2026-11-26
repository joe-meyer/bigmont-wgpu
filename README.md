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

- The engine (`bigmont-core`, `Runner::fermat_async`) takes plain big integers and returns Fermat verdicts. The `bigmont`
  command-line driver is one ready-made front end on top of it.
- The driver also implements the in-driver early-exit interface (`--bins`, `--ee SIZES`, `--chunk`, `--growth`, `--marker`,
  `--lock`) used by the existing CUDA pipeline's feeders: candidates are grouped into ks, tested in waves, and a k stops at its
  first probable prime. Its output is byte-identical to the CUDA driver's on that pipeline's own fixtures (see
  [`docs/superpowers/results/2026-11-25-early-exit-validation.md`](docs/superpowers/results/2026-11-25-early-exit-validation.md));
  it is much slower at start-up for now (issue #21).
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
auto|13|32`, `--retune`, `--cache PATH | --no-cache`, `--verify-rate N`, `--gec`, `--repeat N`. Run `bigmont` with no
arguments for the full usage. The tuning cache lives at `$BIGMONT_CACHE`, else `%LOCALAPPDATA%\bigmont\tune.cache`
(Windows) or `$XDG_CACHE_HOME/bigmont/tune.cache` / `~/.cache/bigmont/tune.cache`.

## Error checking (Gerbicz)

`--gec` runs every Fermat test with the Gerbicz-Li error check, on the GPU and on the CPU. It is off by default, as
it is in the CUDA driver this one replaces, and with it off nothing changes. The ladder is split into blocks of 40 squarings; at each block boundary the
state is folded into two running products, and at the end one extra exponentiation checks that they agree mod n, so a
bit flipped anywhere in a block's computation is caught with overwhelming probability. The last `(bits - 1) % 40` iterations belong to no block, so
they are covered by recomputation instead: they are run twice, once on the ladder state and once on a copy taken at the
last block boundary, and the two results must agree. A candidate shorter than one block is covered entirely this way.

A candidate that fails either check gets **verdict byte 2** in the verdict file: neither composite nor probable
prime, a request to test it again (the same contract the CUDA driver has). It is counted in the summary line (`GEC
alarms`) and, with `--ee`, in the `gec_alarm=` field of the `g1_fermat[EE]` line, and a k with a verdict 2 is retired
like one with a probable prime, as the CUDA driver does. Clean runs never alarm, and their verdicts are byte-identical
to a run without `--gec`. Every probable prime is still re-checked on the CPU, as before; an alarm is not re-checked,
since the point of 2 is to leave the decision to the caller. The CPU runs the same check: buckets it answers
(`--backend cpu`, fewer candidates than the GPU threshold, or after a GPU error) and its re-checks of GPU verdicts are
checked too, so with `--gec` every verdict is either checked or 2. A re-check that alarms makes the candidate 2 as well,
since neither computation can then be trusted. On the CPU the check adds two multiplies per block of 40 squarings, about
40 squarings plus the `2^(sum V)` ladder per candidate, and runs the last `(bits - 1) % 40` iterations twice; its cost
was not measured.

How the detector is tested: the runner tests (`crates/bigmont-core/tests/gec.rs`) build the runner from mutated
kernel sources that simulate a faulty device, and the real runner, kernels and host comparison run on top of them. The
simulated faults are a flipped register mid-block and in the tail, a flipped bit arriving from a load at a dispatch
start, a faulty multiply unit, corrupted memory cells (the modulus, `-n^-1` and the exponent), and a modulus that is
wrong on every load while the buffer is intact. Each must alarm exactly the hit candidates, a hit probable prime
included, and a fault condition that never fires must leave the run clean. The algorithm is also checked against a CPU
model in `bigmont-oracle`, and the engine's handling of an alarm (counted, never re-checked, never overruled) is
unit-tested without a GPU. The CPU check (`bigmont_cpu::fermat_base2_gec`) is written over a small arithmetic trait, and
its tests (`crates/bigmont-cpu/tests/cpu.rs`) run it on a wrapper of the real arithmetic that corrupts the output of one
chosen operation: in a block, at a block boundary, in either tail computation and in the final check. Every such fault
must alarm, a hit probable prime included, and a fault position past the end must leave the run clean.

What the check does not cover:

- Work done on the host. The Montgomery inputs (`R mod n`, `2R mod n`, `-n^-1`), the exponent sums and the final
  comparisons are computed on the CPU and are not checked. A fault on the device that keeps the identity intact is
  possible only with negligible probability. That includes a modulus that every kernel reads wrong: the check's products
  start from `R mod n`, computed on the host from the true `n`, so the identity fails for the wrong modulus (the tests
  confirm this). An unchecked run in the same situation gives a silent false composite. Only an independent computation
  catches errors outside the check: the CPU re-check of probable primes, `--verify-rate` samples, and Beacon's canary
  and notary.
- The final read of the result when no tail recomputation runs (a candidate whose `bits - 1` is a multiple of 40, for
  example 41 or 2,001 bits): the last block is covered by the identity, but the copy of the final value back to the host
  is not compared with anything.

Cost: on 2,029-bit candidates the check lowered throughput by about 15 % (mean of three plain and three `--gec` runs of
100,003 candidates, 34,112 against 28,927 tests/s, each pair between 14.9 % and 15.4 %), measured on one AMD Radeon RX 6700 XT
(AMD proprietary driver 26.10.1, Vulkan, 13-bit limbs, TPI 8). The CUDA driver's kit notes report 514,850 to 581,795 tests/s with and without its check on an RTX 5090, about 11.5 %. Other GPUs
and drivers were not measured; see [the results](docs/superpowers/results/2026-11-25-gerbicz-check.md).

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
