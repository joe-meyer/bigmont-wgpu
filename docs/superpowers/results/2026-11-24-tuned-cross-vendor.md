# Tuning pass results across adapters, and in the browser (2026-11-24)

Code: commit `37c987a` (cheaper Montgomery factor with the inverse masked at load, plus TPI 4 with the carry pass inside the unrolled body). Companion to `2026-11-24-tuning-pass-results.md` (the RX 6700 XT study and ISA counts) and `2026-11-24-unrolled-cross-vendor.md` (the previous best rows). The task review approved it with no Critical or Important findings (it re-derived the carry-pass spacing for every width: at most 20 steps between passes against the safe limit of 24). Every native row below passed the oracle check of its actual timed output (25 to 27 sampled instances per row). Measured: squarings/s, wall-clock, one run per cell, run-to-run noise about 1% (earlier repeats). Projected: tests/s = squarings/s / 2029 (not a measured Fermat test).

## 1. Best verified row per adapter, now against the previous best

| Adapter (OS, API) | Now: best row | squarings/s | projected tests/s | Before (previous best) | Change |
|---|---|---|---|---|---|
| RX 6700 XT (Windows, Vulkan) | TPI 4, wg 64 | 96,210,000 | 47,418 | 81.77 M (TPI 8) | +17.7% |
| RX 6700 XT (Windows, Dx12 + dxcompiler) | TPI 4, wg 64 | 87,530,000 | 43,141 | 75.45 M (TPI 8) | +16.0% |
| RTX 4070 Laptop (Linux, Vulkan) | TPI 4, wg 256 (wg 64: 95.56 M) | 96,228,035 | 47,426 | 92.63 M (TPI 8) | +3.9% |
| Radeon 780M (Linux, RADV) | TPI 4, wg 128 | 24,385,980 | 12,019 | 23.13 M (TPI 8) | +5.4% |
| Quadro P4000 (Linux, shared) | **TPI 8**, wg 256 (TPI 4 only 4.45 M) | 13,782,787 | 6,793 | 13.48 M (TPI 8) | +2.2% |
| llvmpipe on Ryzen 7 7840HS (CPU) | TPI 8, wg 64 (TPI 4: 2.95 M) | 5,316,763 | 2,620 | 5.16 M | +3.0% |
| **Chrome 156 WebGPU, RX 6700 XT** | **rolled** TPI 8 (unrolled TPI 8: 68.3 M; unrolled TPI 4: 60.6 M) | 74,400,000 (earlier run) | 36,685 | 70.8 M unrolled TPI 8 | n/a |

Raw files: `other-adapters/tuning/` (desktop, laptop, host2). Browser kernels were re-checked after this change: unrolled TPI 4, 8, 16 and 32 all pass the three correctness checks in Chrome (including the 175 KB TPI 4 kernel); the rolled kernel was measured in the earlier browser run and not re-run after this commit.

## 2. What the tuning pass did and did not do

- **The literal factor change was slower.** `((p & MASK) * ninv) & MASK` alone kept a full 32-bit multiply (the compiler cannot see that `ninv`, read from a buffer, is small) and ran 1.2% to 3.7% slower on Vulkan; masking `ninv` once at load turns it into a 16-bit multiply-add. That second form gave about +5% on Vulkan for TPI 8, 16 and 32 and no measurable change on Dx12 TPI 8 (noise).
- **TPI 4 helps AMD and the 4070 but not the older P4000.** It is the best width on the RX 6700 XT, the Radeon 780M and (marginally, +4%) the RTX 4070, but a third as fast on the Quadro P4000 (register pressure is the likely cause; not measured). The best width is therefore different per vendor and per environment; an autotune would have to discover it.
- **In the browser the best kernel is the rolled one** (about 74 M against 68 M for unrolled TPI 8 and 61 M for TPI 4), because Chrome's compiler unrolls the loops itself. The tuning pass improved the native paths, not the browser's.
- Dx12 TPI 4 at workgroup 256 drops to 75.7 M (87.5 M at wg 64); cause not investigated.

## 3. The ratio to native CUDA on the RTX 4070

Beacon's CUDA/CGBN driver measured 132,100 tests/s on this card (`2026-11-24-test-b-cuda-vs-wgpu.md`). Our best row now projects to 47,426, **about 35.9%** (CUDA about 2.8x faster), up from 34.6%. Same caveats as there: ours is a projection (no complete Fermat test timed), one card, one CUDA configuration.

## 4. Not covered

Intel GPUs, wave64 mode, Firefox and Safari, other GPUs in a browser, the Dx12 default (no dxcompiler) path for the cooperative kernels, repeat runs for variance (single run per cell), the cause of TPI 4's weakness on the P4000 and at Dx12 wg 256, and GPU timestamp queries.
