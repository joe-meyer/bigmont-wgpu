# Unrolled cooperative kernel: results across all adapters (2026-11-24)

Code: commit `ae14ce1` (the unrolled generator from `7654c98`, verification harness from Task 3). Companion to `2026-11-24-unrolled-kernel-results.md` (the RX 6700 XT study, ISA analysis and review corrections). **Every row below passed an oracle comparison of the actual timed output** (25 to 27 sampled instances per row, after 24 to 768 chained squarings, first and last workgroups included, full value mod n plus normalised limbs and a < 2n check; a deliberately wrong oracle made every row fail, so the check can fail). Measured: squarings/s from wall-clock around submit and a blocking poll, best of 3, one run per row, no GPU timestamps. Projected: tests/s = squarings/s / 2029 (one squaring per exponent bit of a base-2 test, no extra multiplications; not a measured Fermat test).

## 1. Best verified row per adapter, unrolled against the earlier rolled kernel

| Adapter (OS, API) | Unrolled best row | squarings/s | projected tests/s | Rolled best (squarings/s) | Speed-up | Raw file |
|---|---|---|---|---|---|---|
| RTX 4070 Laptop (Linux, Vulkan) | TPI 8, wg 64 | 92,633,607 | 45,655 | 3,413,011 | 27x | `unrolled/raw-laptop-vulkan-verified.txt` |
| RX 6700 XT (Windows, Vulkan) | TPI 8, wg 64 | 81,774,171 | 40,303 | 4,881,358 | 17x | `unrolled/raw-desktop-vulkan-verified.txt` |
| RX 6700 XT (Windows, Dx12 + dxcompiler) | TPI 8, wg 128 | 75,445,418 | 37,184 | 5,021,299 | 15x | `unrolled/raw-desktop-dx12-dxc-verified.txt` |
| Radeon 780M (Linux, Mesa RADV) | TPI 8, wg 128 | 23,128,599 | 11,399 | 7,996,262 | 2.9x | `unrolled/raw-laptop-vulkan-verified.txt` |
| Quadro P4000 (Linux, shared GPU) | TPI 8, wg 256 | 13,482,301 | 6,645 | 948,491 | 14x | `unrolled/raw-host2-ct104-verified.txt` |
| llvmpipe on Ryzen 7 7840HS (software, 16 threads) | TPI 8, wg 64 | 5,164,997 | 2,546 | 250,333 | 21x | `unrolled/raw-laptop-vulkan-verified.txt` |
| llvmpipe in CT 104 (4 shared Xeon vCPUs) | TPI 8, wg 64 | 34,638 | 17 | 22,201 | 1.6x | `unrolled/raw-host2-ct104-verified.txt` |

Rolled figures are from `other-adapters/amortised/` and the first addendum; they were measured at 8 squarings per dispatch, which the skeptical review showed does not disadvantage the rolled kernel (its slope column is 2 to 6% above its long-dispatch rate).

Other widths (best wg), squarings/s at long dispatch: 4070: TPI 16 77.1 M, TPI 32 57.9 M. 6700 XT Vulkan: TPI 16 64.1 M, TPI 32 48.4 M. 780M: TPI 16 18.4 M, TPI 32 13.7 M. P4000: TPI 16 12.6 M, TPI 32 10.2 M. On all four GPUs TPI 8 was best or within a few percent of best, which is the opposite of the rolled kernel (there NVIDIA preferred TPI 32).

## 2. How consistent is this across vendors

Per shader lane the results line up, which is evidence the speed is real and not an artefact of one driver: 6700 XT 82 M / 2,560 lanes = 32 k squarings/s per lane; 780M 23 M / 768 = 30 k; RTX 4070 92.6 M / 4,608 = 20 k; P4000 13.5 M / 1,792 = 7.5 k (an older, shared, lower-clocked card). Lane counts are from memory (unverified). The 780M's smaller gain over its rolled kernel (2.9x against 14x to 27x elsewhere) is consistent with Mesa's compiler already doing better with the rolled code, but that is a guess, not tested.

## 3. What was verified about the AMD code (Task 3)

wgpu 30.0.2 sets only `ALLOW_VARYING_SUBGROUP_SIZE` on compute stages and a probe measured subgroup size 32 on Vulkan and on Dx12. In the matching compile mode the unrolled TPI 8 kernel has 840 `v_mad_u32_u24`, 0 `v_readlane`, 44 `ds_bpermute`, 8 to 9 `s_cbranch`; the measured rate is about 85 to 89% of a nominal (unverified) vector-issue bound. The earlier "readlane waterfall" and "60% of peak" statements were withdrawn. Only the Vulkan ISA was inspected; Dx12/DXC code generation and the NVIDIA, Mesa and software compilers were not.

## 4. Not covered

Intel GPUs, wave64 mode, browsers (still not browser-valid: naga rejects `enable subgroups;`), CUDA/CGBN on any card, a full Fermat test with Gerbicz checking, GPU timestamp queries, repeat runs for variance, the desktop OpenGL adapter and WARP (one-thread kernel only), and the NVIDIA and Mesa ISA.
