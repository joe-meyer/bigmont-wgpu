# Test B: our wgpu kernel against Beacon's CUDA/CGBN driver on the same card (2026-11-24)

Raw output: `test-b/raw-rtx4070-cuda-vs-wgpu.txt`. This is the comparison the gate note (`2026-11-24-proposed-gate-v2.md`, Test B) said was missing.

## Result

On the RTX 4070 Laptop GPU, 2,029-bit numbers (Beacon's 2,048-bit size class), no Gerbicz check on either side:

| | tests/s | Kind |
|---|---|---|
| Beacon CUDA/CGBN (`g1_fermat_v3`, kit v8.57) | about 132,100 (130,711; 132,091; 132,150; 132,145; 132,081) | **measured**; canary and all 32,000 verdicts correct |
| Our wgpu unrolled kernel, TPI 8 | about 45,650 (45,637; 45,661) | **projected** from measured squarings/s (92.6 M) divided by 2,029; oracle-verified output |
| **Ratio (ours / CUDA)** | **about 34.6%**, i.e. CUDA is about 2.9x faster | |

Both runs were back to back on the same GPU, whose SM clock was in the same band for both (about 2.35 to 2.47 GHz). Against the proposed gate in `2026-11-24-proposed-gate-v2.md` this lands just above the 33% "clear go" line and about 3.5 times the 10% floor; those thresholds are my proposals, not derived values, and the margin over 33% is small enough that the caveats below could move it either way.

## How it was measured

- Beacon: `djuk8/beacon` at `ad414e4`, kit `o1_kit_v8.57` `g1_fermat_v3.cu`, with NVlabs CGBN at `114601b`, built for `sm_89` with `nvcc` 13.0.48 using Beacon's own build line. Its compile-time `FERMAT_GEC` defaults to 0 in this kit (no Gerbicz-Li check), which is the like-for-like for our kernel (the Beacon notes say turning the check on costs about 13%). The size class for 2,048 bits uses 8 lanes per number, the same width as our best kernel.
- Input: Beacon's own 400-candidate canary (200 known primes, 200 composites) repeated 80 times into a 32,000-candidate file in Beacon's candidate format, run with `--repeat 10` (320,000 tests, about 2.4 s). A 400-candidate launch is too small to fill the GPU (50 k tests/s); the larger batch is the fair one. A base-2 Fermat test does the same squarings whatever the number, so repeating candidates does not change the work.
- Ours: the Task 3 sweep row `unrolled 8 64` with 131,064 instances (oracle-verified, 27 instances after 768 chained squarings).

## Caveats

1. **Measured against projected.** Beacon's figure is a measured end-to-end driver rate (including copying candidates and verdicts). Ours is a projection: squarings/s measured on the kernel, divided by 2,029 (one squaring per exponent bit, doubling assumed nearly free). We have not implemented or timed a complete Fermat test (initial conversion, final comparison, handling the doubling). Those are small next to 2,029 squarings but not zero, so the true ratio is a little lower than 34.6%.
2. **One card.** The RTX 4070 laptop only (power-limited, driver 580). The AMD cards cannot run CUDA; a HIP port of CGBN would be needed for an AMD comparison and was not tried. Beacon reports 486 k tests/s at 2,048 bits on an idle RTX 5090 (its driver notes); that is another card and not compared here.
3. **Beacon's kit is tuned by its author for the RTX 5090**; compile flags here are its own (`-O3`, `sm_89`). No sweep of CUDA launch parameters was done for the 4070, so CUDA might gain a little on this card.
4. **Same-clock check is approximate:** SM clock was sampled once a second during short runs.
5. A first guess in this project (a per-lane-equal CUDA would reach about 82 k tests/s on this card, putting us near 55%) was wrong; the measured CUDA rate is about 1.6x higher than that estimate.

## Environment changes made for this test (on the laptop, user level, no sudo)

`~/cuda-13.0` (toolkit pieces merged from NVIDIA's `cuda_13.0.0_580.65.06_linux.run`, 4,300,799,649 bytes, from developer.download.nvidia.com; the runfile's installer could not run because Ubuntu 26.04 lacks `libxml2.so.2`, so its payload was extracted and merged by hand); one private header patched for glibc 2.41's `noexcept` `rsqrt` (backup `math_functions.h.orig`); `~/beacon-test` (Beacon, CGBN, kit, a user-level `gmp.h`); the runfile and extraction folders were deleted afterwards.
