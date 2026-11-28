# Symmetric squaring for the 13-bit cooperative kernel (2026-11-27)

Issue #2, second half. The 32-bit symmetric kernel (`2026-11-27-symmetric-squaring-32bit.md`) is native-only. This note covers
the 13-bit one (`Squaring::Symmetric` for `LimbScheme::Limb13`, `crates/bigmont-core/src/unrolledsq.rs`). It needs no 64-bit
integers, so it also runs in the browser. It comes in two operand packings (`Packing::Limbs`, `Packing::Pairs`), and the tuner
times both next to the fused kernel.

**Result in short.** All results are at 2,029 bits. "Kernel-only" means the `scheme_compare` slope benchmark, 3 runs each.

| | best fused | best symmetric | gain |
|---|---|---|---|
| RTX 4070 Laptop, native, kernel-only | 9.51e7 to 9.57e7 (TPI 4) | 1.105e8 (pairs, TPI 4) | +15.4% to +16.2% |
| RX 6700 XT, native, kernel-only | 9.52e7 to 9.68e7 (TPI 4) | 9.20e7 to 9.28e7 (limbs, TPI 8) | -3% to -4% (the tuner keeps fused) |
| RX 6700 XT, Chrome WebGPU, `fermat.html`, tests/s | 33,743 (TPI 8) | 38,433 (limbs, TPI 8) | +13.9% (one run) |

The native winner on the 4070 is still the 32-bit scheme (1.80e8 squarings/s symmetric). The 13-bit kernel matters where the
32-bit scheme cannot run: browsers, and devices without 64-bit shader integers. The browser on the 6700 XT has no autotuner, and
its fused optimum is TPI 8, not TPI 4 as natively, so the symmetric kernel gains there although it loses natively.

## 1. What was built

The schedule is the one of the 32-bit kernel: the rediagonalisation of cuda-fixnum's wide squaring, with a lane's whole limb run
as the digit. It is adapted to redundant 13-bit limbs. The module comment of `unrolledsq.rs` has the full argument; the main
points:

- **Phase 1** forms `x^2` in two 32-bit accumulator blocks per lane: block `j` and block `TPI + j`. The lanes are relabelled
  per iteration (lane `j` plays role `(j - 2i) mod TPI`), so the low half of each block product lands in the lane itself and
  only the high half moves, one lane up. Cross products are doubled limbwise (in redundant form that needs no carry), and carries
  are deferred to in-lane light passes.
- **Phase 2** reduces only the low half, with the fused loop minus its `x b` products. It then adds the high half limb by limb:
  `(S + M n) / R = (S_lo + M n) / R + S_hi`, and lane `j` holds both block `j` of the result and block `TPI + j` of `S`. There
  is no injection into the window and no extra shuffle per step. The 32-bit kernel injects the high words; the same trick may
  help it too (not tried).
- **Bounds are checked at generation time.** A tracker (`Bound`) places the light passes and asserts that no accumulator can
  exceed 2^32 for the lane count and limb count being generated. The reduction's light-pass period comes out at about 60 steps,
  against 20 in the fused loop, because each step adds one product instead of two.
- **Products per lane** fall to about 0.75x of the fused loop's, plus the redundant upper-diagonal squares (as in the 32-bit
  kernel).

Shuffles per squaring are slightly more than the fused kernel's: about `3.5 TPI LPL` against `3 TPI LPL`, counted from the
generator without packing. Phase 1 moves the operand blocks and the high halves; phase 2 needs two shuffles per step instead
of three. Packing halves phase 1's operand shuffles.

## 2. Register pressure (the AMD problem and what fixed it)

The first version lost at every width on the 6700 XT: -20% (TPI 4), -4% (TPI 8), -8% (TPI 16), -12% (TPI 32). It won on the
4070 (+14.1% at TPI 4). The AMD driver (VK_KHR_pipeline_executable_properties, `pipeline_stats`) showed 256 VGPRs plus scratch
at TPI 4 and 8. Ablations located the cause: with one cross iteration the kernel used 144 VGPRs and did not spill. Every block
product depends only on `x`, so the compiler overlapped several iterations to hide shuffle latency, and several products were
live at once. Three changes, each checked against the oracle:

1. **Order the iterations** (`ORDER` in `phase1`). Each iteration's lane index gets `(OR of all accumulator words) & params.z`
   added. `params.z` is always 0 (every host writes `[count, iterations, 0, 0]`), but the compiler cannot know that, so an
   iteration waits for the previous one. A cheaper zero, `lo0 >> 31`, was folded away by the compiler (no change at all). Making
   the dependency on one word only was not enough either; it has to cover every word that the previous iteration writes.
2. **Accumulate the low half in place.** The `lo` names are swapped per lane (once, at the iteration where the lane's target block
   changes), so the low-half products go straight into the output block and only the high half needs its own `LPL` words.
3. **Packed operands** (`Packing::Pairs`, optional). `x` is held two 13-bit limbs per word during phase 1. That halves its
   registers and its operand shuffles, for one AND or shift per unpacked limb.

| TPI | fused VGPRs | symmetric VGPRs (scratch) | symmetric-pairs VGPRs (scratch) | first version |
|---|---|---|---|---|
| 4 | 192 | 256 (144 B) | 256 (128 B) | 256 (672 B) |
| 8 | 96 | 160 (0) | 144 (0) | 256 (464 B) |
| 16 | 76 | 96 (0) | 101 (0) | 112 (0) |
| 32 | 45 | 58 (0) | 55 (0) | - |

(`symmetric-squaring-13bit/registers-6700xt.txt`.) Packing helps the 4070 by about 3 points at TPI 4. On the 6700 XT it costs
about 30% even though it uses fewer registers. The likely reason is that the compiler repeats the unpack at every use to save
registers, but that is not established. Unpacking with `extractBits` instead of a mask or shift recovered part of the loss (TPI 8:
8.33e7 against 6.18e7, one run) but not all of it. So both forms are generated and the tuner picks. TPI 4 still spills on the
6700 XT: without packing the kernel's own state is about five values per limb, 200 registers at 40 limbs per lane.

Also tried and dropped: rotating the high half with two constant-offset shuffles (up 1, and down `TPI - 1` for lane 0) instead
of one indexed shuffle was 7% slower at TPI 4 on the 6700 XT. Loading `n` in phase 2 instead of holding it changed nothing (the
loads are loop-invariant).

## 3. Kernel-only results

`scheme_compare --bits 2029 --schemes 13 --squarings fused,symmetric,symmetric-pairs --tpis 4,8,16,32 --wgs 64,128,256 --runs 3`,
65,536 instances, run three times on each machine (each run's summary takes the median of 3 timings at the best workgroup).
Squarings per second; the ratio is against fused at the same TPI.

RTX 4070 Laptop, NVIDIA 580.178.04, Ubuntu 26.04, Vulkan (`kernel-4070-run1..3.txt`):

| TPI | fused | symmetric (ratio) | symmetric-pairs (ratio) |
|---|---|---|---|
| 4 | 9.51e7 to 9.57e7 | 1.076e8 (1.125, 1.132, 1.131) | 1.105e8 (1.154, 1.162, 1.162) |
| 8 | 9.22e7 | 1.037e8 to 1.040e8 (1.125 to 1.128) | 1.019e8 to 1.023e8 (1.106 to 1.110) |
| 16 | 7.59e7 | 7.90e7 to 7.92e7 (1.042 to 1.044) | 8.07e7 to 8.08e7 (1.064) |
| 32 | 5.67e7 | 5.31e7 (0.936, 0.937) | 5.30e7 (0.935, 0.936) |

RX 6700 XT, AMD proprietary driver 26.10.1, Windows 11, Vulkan (`kernel-6700xt-run1..3.txt`):

| TPI | fused | symmetric (ratio) | symmetric-pairs (ratio) |
|---|---|---|---|
| 4 | 9.52e7 to 9.68e7 | 8.86e7 to 8.88e7 (0.917 to 0.931) | 6.09e7 (0.629 to 0.639) |
| 8 | 8.67e7 to 8.71e7 | 9.20e7 to 9.28e7 (1.062 to 1.066) | 6.19e7 to 6.21e7 (0.713, 0.714) |
| 16 | 6.81e7 to 6.87e7 | 6.48e7 to 6.50e7 (0.946 to 0.950) | 4.89e7 (0.713 to 0.718) |
| 32 | 5.20e7 to 5.22e7 | 4.57e7 to 4.60e7 (0.879 to 0.881) | 3.98e7 to 3.99e7 (0.763 to 0.766) |

First version on the 6700 XT, before section 2's changes: `kernel-6700xt-first-version.txt`.

Tuner candidates: symmetric in both packings at TPI 4, 8 and 16 (TPI 32 lost on both machines).

## 4. End to end

RTX 4070 Laptop, `fermat_stats --limbs 13 --squaring S --tpi T --wg W cand2029.bin` (131,075 candidates of 2,029 bits, engine
capacity 16,384), three rounds, wall-clock tests/s (`endtoend-4070.txt`):

| shape | round 1 | round 2 | round 3 |
|---|---|---|---|
| fused, TPI 4, wg 64 | 39,899 | 39,829 | 39,725 |
| fused, TPI 8, wg 128 | 39,883 | 40,088 | 40,199 |
| symmetric, TPI 4, wg 128 | 42,153 | 42,131 | 42,223 |
| symmetric-pairs, TPI 4, wg 128 | 42,537 | 42,611 | 42,550 |
| symmetric, TPI 8, wg 256 | 42,959 | 42,928 | 42,972 |

The best symmetric shape is 6.9% faster than the best fused one end to end, against about 16% kernel-only. The gap is like
the 32-bit kernel's, where it is not explained either. The tuner restricted to the 13-bit scheme (`--limbs 13`) picked
symmetric-pairs at TPI 4 and wg 64 and ran 42,657 tests/s. That is 0.7% below the best shape measured here, because the
tuner's kernel-only timing ranks pairs at TPI 4 first. `auto` still picks the 32-bit symmetric kernel (73,354 tests/s, unchanged).

Not measured end to end: the 6700 XT (natively its tuner keeps the fused kernel, so the tuned engine is unchanged there).

## 5. Browser

Chrome 156 (the Claude desktop app's built-in Chromium), Windows 11, WebGPU on the RX 6700 XT (`browser-6700xt.txt`):

- `rust.html`: both packings at TPI 4, 8 and 16, and at TPI 8 with the Gerbicz check, at 255, 1,019 and 2,029 bits. 0
  mismatches, 0 alarms.
- `fermat.html`: all 32 symmetric kernels (both packings, 64 to 2,029 bits, TPI 4 to 32), the fused 2,029-bit TPI 4 and 8
  kernels, and the 32-bit negative control (rejected as expected). 0 wrong verdicts.

| kernel at 2,029 bits, tests/s | TPI 4 | TPI 8 | TPI 16 | TPI 32 |
|---|---|---|---|---|
| fused | 27,557 | 33,743 | - | - |
| symmetric | 17,403 | 38,433 | 27,617 | 20,004 |
| symmetric-pairs | 12,936 | 25,630 | 21,130 | 17,315 |

One run each. The browser has no tuner, so the page's caller picks the shape (`rust.html?squaring=symmetric&tpi=8`).

## 6. Correctness

- **Oracle tests on both GPUs.** Both packings run in `chain.rs` (TPI 4 to 32, workgroups 64 to 256, 3- and 128-squaring
  chains), in `sizes.rs` and in `fermat.rs`:
  - `sizes.rs` covers every size class, including 200 limbs at TPI 4 (passes inside each block product) and 264 limbs at TPI 4
    (reduction passes inside the unrolled body, with a remainder). It also covers 1 and 2 limbs per lane and odd limb counts
    per lane (a half-filled last pair).
  - `fermat.rs` checks verdicts through the `Runner`, including the Gerbicz check.
- **Generated sources.** `source_matrix.rs` validates every symmetric source on both targets without int64. It also checks that
  multiply mode stays fused, and that packing is rejected outside the 13-bit symmetric squaring.
- **Full workspace suite.** It passed on both machines, except one CLI test whose config rebuild dropped the packing. That was
  a test bug: fixed, and that test file was rerun on both.

## 7. Not tested

Intel, Apple and software rasterisers; other NVIDIA and AMD generations; Firefox and Safari WebGPU. The browser was only run on
the 6700 XT. On an NVIDIA GPU, where the native result suggests packing would win, it was not run.
