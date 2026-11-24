# bigmont-wgpu: portable GPU engine for Beacon-class prime-gap hunting

Date: 2026-11-24
Status: design, awaiting review
Scope: sub-project A (the engine). Sub-project B (web distribution) is out of scope here; section 8 only fixes the boundary.

## 1. Goal

Run the heavy GPU stages of a Beacon-style prime-gap hunt (djuk8/beacon, Apache-2.0) on any GPU wgpu supports (AMD, NVIDIA, Intel, Apple), with kernels that can later run in a browser through WebGPU.

Beacon today is NVIDIA-only: a CUDA Fermat driver built on NVIDIA's CGBN bignum library, a CUDA bucket sieve, and Python/bash orchestration that needs `nvcc`, `nvidia-smi`, root, Linux and `/dev/shm`.

Decisions already made with the user:

- Targets: every vendor, through wgpu/WebGPU (not HIP first).
- Home: a new standalone Rust crate, usable by gapsieve and by a Beacon fork.
- Approach: subgroup-cooperative Montgomery arithmetic (CGBN-shaped), with a one-thread-per-number backend as correctness baseline and fallback, gated by a Phase 0 speed spike.
- Web side (B): open public volunteers, so verification is a real requirement. Specified later.

## 2. What Beacon's GPU code actually depends on

Read from `cuda/g1_fermat_v33.cu` (568 lines), `cuda/gbs.cu` (110 lines) and `deploy/fresh_box.sh`. I have not read the rest of the kit, which is in `deploy/kit_v8/o1_kit_v8.57.tgz`.

- Fermat driver: a thin layer over CGBN calls (`cgbn_mont_sqr`, `cgbn_mont_mul`, `cgbn_extract_bits_ui32`, `cgbn_clz`, `cgbn_rem`, `cgbn_equals`). No hand-written PTX of its own. It performs a windowed base-2 exponentiation with a Gerbicz error check, and has a GMP-backed CPU path (`-DCPU_TEST`).
- Driver interface is file-based: `g1_fermat cand.bin verdict.bin [--device N --lock F --ee sizes.bin --bins ...]`. This is why a drop-in replacement looks feasible; it needs confirming against the kit's callers.
- Sieve: plain integer CUDA. 64-bit stepping and `atomicAnd` on `unsigned long long`. WGSL has no u64 and no u64 atomics, so this needs a two-u32 rewrite.
- Orchestration: bash, Python and cron. NVIDIA-specific only in `nvidia-smi` (card count, compute capability, VRAM) and the `nvcc` build line.
- Existing acceptance material: a 2,029-bit canary set (200 primes, 200 composites, expected verdicts) and an identity gate (GPU sieve byte-identical to an independent CPU sieve).

## 3. Components

| Unit | Purpose | Depends on |
|---|---|---|
| `bigmont-core` | 2,048-bit Montgomery multiply/square, windowed modexp, Gerbicz check, behind a `Backend` trait | wgpu |
| `bigmont-oracle` | CPU reference for every operation (GMP or `num-bigint`) | none |
| `bigmont-sieve` | Bucket sieve with u32-pair atomics, gated byte-identical to Beacon's CPU sieve | wgpu, oracle |
| `bigmont-cli` | Driver speaking Beacon's candidate/verdict file formats; portable device detection instead of `nvidia-smi` | core, sieve |

Backends behind the `Backend` trait:

1. **Subgroup-cooperative.** Several lanes of one subgroup share one number (CGBN's TPI idea). Limbs are spread across lanes and carries move by subgroup shuffles. Lanes per number are chosen at runtime from the adapter's subgroup size, which differs by vendor (32 on NVIDIA and Apple, 32 or 64 on AMD, 8 to 32 on Intel).
2. **Workgroup-memory cooperative.** Same algorithm, carries exchanged through workgroup memory and barriers, for adapters without subgroup support. Slower.
3. **One thread per number.** Whole number in private memory. Simplest to get right; expected to spill registers badly at 2,048 bits (unmeasured). Used as the correctness baseline.

Kernels use only WGSL features that browsers implement. Native-only features (push constants, 64-bit integers, other wgpu extensions) stay out of the kernels, so the same kernels can be reused in a browser build.

A native per-vendor backend (HIP, Metal) remains possible behind the same trait if the spike shows WGSL is too slow. It is not planned.

## 4. Phase 0: speed spike (go/no-go)

Build a 2,048-bit Montgomery squaring in backends 1 and 3 (backend 2 if backend 1 is blocked by subgroup support) and time them on the RX 6700 XT (Windows) and the RTX 4070 laptop.

Measure: squarings per second per card, and the projected full-exponentiation rate (2,048 squarings plus window multiplies).

Reference: CGBN's published per-card figures from Beacon's README (516,000 to 523,000 tests/s per card at the 2,048-bit class, which I read as an early-exit figure, not full exponentiations; to be checked). A direct CGBN run on the laptop needs the CUDA toolkit (several GB); ask the user before downloading.

Gate: the threshold is set with the user once the numbers exist. If every backend lands far below it, stop and reconsider a native backend. Only the spike code is lost.

## 5. Later phases (each gets its own plan after Phase 0)

1. `bigmont-core` and `bigmont-oracle`: full API, randomised tests, canary acceptance.
2. `bigmont-sieve`: port of `gbs.cu`, identity gate against the CPU sieve.
3. `bigmont-cli`: drop-in for `g1_fermat_v3`, verified against the callers in the kit.
4. Portable orchestration: replace `nvidia-smi`, root and `/dev/shm` assumptions (in a wrapper or a Beacon fork).

## 6. Correctness

- Acceptance: Beacon's canary set must reproduce the expected verdicts byte for byte.
- Randomised tests against the oracle at several sizes, including edge moduli (2^n - 1, values near limb boundaries).
- Gerbicz alarms must be zero on real runs. Fault injection on purpose proves the check fires.
- A backend's speed is not reported until it passes everything above.

## 7. Risks

- Subgroup support: wgpu and browser support for subgroup operations is recent. Check current status before relying on it. Fallback is backend 2.
- Speed: a portable 2,048-bit kernel may be several times slower than CGBN. That is a guess; Phase 0 measures it.
- Tuning surface: subgroup size varies by vendor, so lanes per number and window size need per-adapter tuning.
- Upstream drift: Beacon's kit versions change quickly. The drop-in driver depends on file formats that need re-checking each kit version.

## 8. Boundary with sub-project B (web distribution)

The engine exposes only two operations: candidate file in, verdicts out; k-range in, survivors out. It has no networking.

B is a separate spec, written after Phase 0 has numbers. For open public volunteers it will need at least:

- canary work units mixed into every batch;
- redundant re-testing of a sample of results, since "tested and found nothing" cannot be checked cheaply;
- server-side re-derivation and ECPP certification of any claimed gap (Beacon's `rewalk.py` and `certify_batch.py` do this today);
- a k-range allocator replacing Beacon's forum-claimed seeds;
- a browser build of the engine (WebGPU), with the same kernels.

None of these are designed yet.
