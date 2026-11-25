# Limb-scheme experiment: 13-bit limbs vs 32-bit limbs with u64 products (one thread per number)

Date: 2026-11-24. Commit measured: `bbab9d1` (branch of `phase1`). Host: test-laptop (Ubuntu 26.04), RTX 4070
Laptop (NVIDIA 580.178.04, Vulkan) and Radeon 780M (RADV PHOENIX, Vulkan). Raw outputs:
`docs/superpowers/results/phase1/schemes/`.

## What was compared

`crates/bigmont-core/src/onethread.rs` (experimental). The kernels are fully unrolled, one thread per
number, and use no subgroup operations.

- **A (13-bit)**: the `unrolled.rs` arithmetic with TPI = 1. It uses 13-bit limbs in u32,
  `L = limbs_for_bits(bits)` and R = 2^(13 L). Each step is a 24-bit multiply-add (`& 0xffffff` operand
  masks), with the one-limb shift fused into the accumulate. A light carry pass runs every 20 steps and one
  sequential carry pass ends each squaring. The result is in [0, 2n) with normalised limbs.
- **B (32-bit, u64)**: separated CIOS with sequential u64 carries. It needs SHADER_INT64, uses
  `W = ceil((bits + 2) / 32)` limbs and R = 2^(32 W), and ends with a conditional subtraction, so the
  result is in [0, n).
- **B2**: B with the two passes fused per multiplier limb (FIOS: both carry chains in one loop over
  `j`). It is a low-effort variant. It does not reduce the number of carry operations.

| bits | A limbs | B/B2 limbs |
|---|---|---|
| 256 | 20 | 9 |
| 384 | 30 | 13 |
| 512 | 40 | 17 |
| 1024 | 80 | 33 |

Correctness is covered by `crates/bigmont-core/tests/onethread.rs`. Every instance is compared with
`mont_square_chain_bits` at 64, 130, 256, 384, 512 and 1024 bits. The inputs include random moduli,
2^bits - 1 and 2^(bits-1) + 1. Instance counts of 200, 130 and 300 are not multiples of the workgroup
sizes 64, 128 and 256. The runs use chains of 64, 257 and 500 squarings, 2 chained dispatches and 0
squarings. All tests pass on the 4070 and on the 780M.

Method: `examples/scheme_bench.rs` uses the slope method of `spike` on 262,144 instances. For each
workgroup size in {64, 128, 256} it runs 5 times. Each run times a 2-squaring dispatch (best of 5), then
doubles the squarings while the predicted dispatch time stays at or below 0.4 s (best of 3). Every run's
TIMED output is checked against the oracle on 27 sampled instances. No run failed. Each adapter was
benchmarked by 2 separate processes (run1 and run2). The tables give run1, which was designated primary
before seeing data. run2 agrees within 0.01 on every ratio.

## RTX 4070 Laptop (run1; run2 in brackets)

| scheme | bits | best wg | sq/s min | sq/s median | sq/s max | ratio vs A (medians) |
|---|---|---|---|---|---|---|
| A | 256 | 256 | 6.822e9 | 6.825e9 | 6.830e9 | 1.00 |
| B | 256 | 64 | 9.810e9 | 9.821e9 | 9.876e9 | **1.44** [1.45] |
| B2 | 256 | 64 | 1.0035e10 | 1.0046e10 | 1.0053e10 | 1.47 [1.48] |
| A | 384 | 64 | 2.913e9 | 2.916e9 | 2.934e9 | 1.00 |
| B | 384 | 128 | 4.674e9 | 4.675e9 | 4.677e9 | **1.60** [1.60] |
| B2 | 384 | 128 | 4.680e9 | 4.683e9 | 4.685e9 | 1.61 [1.61] |
| A | 512 | 256 | 1.700e9 | 1.702e9 | 1.703e9 | 1.00 |
| B | 512 | 128 | 2.715e9 | 2.716e9 | 2.716e9 | **1.60** [1.60] |
| B2 | 512 | 64 | 2.699e9 | 2.701e9 | 2.702e9 | 1.59 [1.59] |
| A | 1024 | 64 | 2.848e8 | 2.854e8 | 2.856e8 | 1.00 (A spills, see below) |
| B | 1024 | 64 | 7.031e8 | 7.042e8 | 7.054e8 | **2.47** [2.47] |
| B2 | 1024 | 128 | 7.004e8 | 7.009e8 | 7.010e8 | 2.46 [2.46] |

The B/A spread over the runs (min B / max A to max B / min A) is narrow at every size: 256 gives
1.44-1.45, 384 gives 1.59-1.61, 512 gives 1.59-1.60 and 1024 gives 2.46-2.48.

**Pre-registered bucket: INCONCLUSIVE ("inspect the generated code before deciding").** The B/A medians
are 1.44 at 256 bits, 1.60 at 384 bits and 1.60 at 512 bits. That is not >= 1.5 at all three spill-free
sizes, because 256 is below 1.5, and it is not < 1.2 at any of them. B2 gives 1.47, 1.61 and 1.59, so it
lands in the same bucket.

## Radeon 780M (run1; run2 in brackets)

| scheme | bits | best wg | sq/s median | ratio vs A (medians) |
|---|---|---|---|---|
| A | 256 | 64 | 1.699e9 | 1.00 |
| B | 256 | 64 | 9.282e8 | **0.55** [0.55] |
| B2 | 256 | 256 | 9.304e8 | 0.55 [0.55] |
| A | 384 | 64 | 6.125e8 | 1.00 (A spills on RADV) |
| B | 384 | 64 | 4.455e8 | **0.73** [0.72] |
| B2 | 384 | 128 | 4.503e8 | 0.74 [0.72] |
| A | 512 | 128 | 2.996e7 | 1.00 (A spills heavily on RADV) |
| B | 512 | 128 | 2.158e8 | 7.20 [7.19] |
| B2 | 512 | 256 | 2.428e8 | 8.10 [8.10] |
| A | 1024 | 256 | 2.852e6 | 1.00 (A spills) |
| B | 1024 | 64 | 1.438e7 | 5.04 [5.12] |
| B2 | 1024 | 128 | 1.513e7 | 5.31 [5.37] |

On the 780M, B/A is at or below 1 only where A does not spill much (256 bits: 0.55; 384: 0.73). At 512
and 1024 bits, the one-thread A kernel spills (see below) and B wins by 5-8x. That is a register-pressure
effect of one thread per number, not a property of the arithmetic.

## Register use and spills (driver pipeline statistics)

The data comes from `examples/pipeline_stats.rs onethread ...` (raw:
`phase1/schemes/pipeline_stats-4070-780M.txt`). NVIDIA reports "Local Memory Size" with a constant 2^36
added. The values below are the excess over that constant.

| kernel | 4070 regs (wg64) | 4070 local mem/thread | 780M VGPRs | 780M spilled VGPRs |
|---|---|---|---|---|
| A 256 | 96 | 0 | 256 | 0 |
| A 384 | 128 | 0 | 256 | 106 |
| A 512 | 168 | 0 | 256 | 364 |
| A 1024 | 255 | **560 B (wg64), 768 B (wg256)** | 256 | 2832 |
| B 256 | 64 | 0 | 80 | 0 |
| B 384 | 80 | 0 | 168 | 0 |
| B 512 | 80 | 0 | 256 | 0 |
| B 1024 | 168 | 0 (wg64), 384 B (wg256) | 256 | 411 |

- On the 4070, A is spill-free at 256, 384 and 512 bits and spills at 1024 bits. B spills only at 1024
  bits with wg 256. The best B row at 1024 uses wg 64, which has no local memory.
- NVIDIA also reports a nonzero "Shared Memory Size" for kernels that declare no shared memory. Examples
  are 16,640 B for A 512 at wg64 and 6,912 B for B 512 at wg64. It is not known what that is (possibly
  driver spill-to-shared or a reservation). It is recorded raw and left uninterpreted.
- On RADV the 780M compiles these kernels as wave64. A spills from 384 bits up, so on the 780M only the
  256-bit A row is spill-free.

For whoever does the inspection, cheap code facts. On RADV (780M ISA, 256 bits), A is 418 `v_mad_u32_u24`
plus 212 `v_mul_u32_u24`. B has no `v_mad_u64_u32`: each u64 product becomes a `v_mul_lo_u32` /
`v_mul_hi_u32` pair (135 + 126), and the carries become `v_add_co_u32` / `v_add_co_ci_u32` (297 each) plus
152 64-bit shifts (`v_lshrrev_b64`) for `c = a >> 32`. NVIDIA exposes no internal representation
through this extension, so no SASS was inspected. B at 256 bits uses 9 limbs (288 bits) because of the
pre-registered `+2` headroom, where 8 limbs would hold 256-bit moduli if the result only had to stay
below R > n. That is about 81/64 = 1.27x more limb products than a tight 8-limb B. This was not measured.

## Not verified / limits

- One thread per number is not the cooperative (TPI >= 4) kernel the library uses. Register pressure,
  occupancy and memory access patterns differ completely. In particular, the 780M's 512/1024 rows and
  the 4070's 1024 row measure spills, not arithmetic.
- Cross-lane carries for 32-bit limbs (the hard part of a cooperative wide-limb kernel) were not built
  or tested.
- Any comparison with CGBN is by projection only. CGBN was not run in this experiment.
- One GPU of each vendor (a laptop RTX 4070 and an integrated 780M). Neither the RX 6700 XT nor any
  other NVIDIA part was measured: GPU work on the desktop is paused.
- The timings are wall-clock around submit plus a blocking poll, not GPU timestamps. Clocks were not
  locked; the 4070 ran at about 2.2-2.5 GHz.
- B's limb count follows the pre-registered `ceil((bits + 2) / 32)`. A tighter B, or a squaring that
  exploits symmetry (in either scheme), was not tried.
- The full workspace test suite on the 4070 had 13 failures, all "no adapter matches 4070" at
  `Gpu::new` after several devices had been created in the same test process (chain.rs, fermat.rs and
  sizes.rs). The base commit `7d69749` fails the same 4 chain.rs tests the same way, and those tests pass
  when run on their own. Raw: `phase1/schemes/workspace-tests-4070.txt`. One Xid 31 occurred during an
  onethread test run (device creation, passed on rerun), and one Xid 69 occurred during a
  `pipeline_stats` (raw `ash`) run.

## Addendum: fair-size rerun (254, 382, 510, 1022 bits), 2026-11-25

Why: the first run used 256/384/512 bits, where B's pre-registered `ceil((bits + 2) / 32)` rule wastes limbs
(256 bits -> 9 limbs where 8 fit). At bits = 32W - 2 B has no slack: 254 -> 8 limbs, 382 -> 12, 510 -> 16,
1022 -> 32. A's slack is 2 bits at 254 and 382, and 10 bits at 510 (40 limbs). Same code (`bbab9d1`), same method
(262,144 instances, 3 runs, wg 64/128/256, oracle-checked output, 0 failures on either host). Raw:
`phase1/schemes/scheme_bench-fair-4070.txt`, `scheme_bench-fair-6700xt.txt`.

| bits | RTX 4070 Laptop B/A | RX 6700 XT B/A |
|---|---|---|
| 254 | **1.87** | 0.77 |
| 382 | **1.92** | 0.75 |
| 510 | **1.86** | 0.72 |
| 1022 | 2.61 (A may spill) | 0.47 (A may spill) |

B2 is within 0.01 of B everywhere. The min/max spread over runs is under 0.02 at 254-510 bits on the 4070.

- **NVIDIA:** the pre-registered 1.5x bar is met at all three spill-free sizes (1.86-1.92), so the first run's
  "inconclusive" came mostly from B's wasted limb at 256 bits. Slack-free 32-bit limbs with u64 products are
  about 1.9x faster than 13-bit limbs in the one-thread kernel.
- **AMD (RDNA2, proprietary driver 26.10.1):** B is 25-28% slower at 254-510 bits, even without spills in B.
  This matches the instruction-probe estimate (about 0.9x) in direction. The 780M (RADV) was also below 1 in
  the first run (0.55-0.73). So the better limb scheme depends on the GPU vendor: one fixed scheme cannot
  be best everywhere, which supports choosing the limb width per adapter (autotune).
- The 1022-bit rows mix in A's register spills on both hosts and are not an arithmetic comparison.

Not verified: all limits listed above still hold. This is one thread per number, not the cooperative kernel,
so the 1.9x is an upper-bound indication for the NVIDIA lever, not a prediction for 2,029 bits, where the
cross-lane carry cost of 32-bit limbs is unmeasured. The desktop's GPU showed 36% total engine utilisation
before the run (one 3D-engine process while it ran); A's absolute rate on the 6700 XT (6.16e9 at 254 bits)
is close to the 4070's, which suggests little interference, but this was not isolated. Clocks were not locked.
