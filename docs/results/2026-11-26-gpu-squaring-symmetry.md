# Squaring symmetry in the one-thread GPU kernels (2026-11-26)

Issue #2. A Montgomery squaring needs about three quarters of the limb products of a general multiply, because `x_i * x_j`
appears twice in `x^2`. Neither limb scheme used this: both run the general multiply-accumulate loop, so a squaring costs a
full multiplication plus the reduction. The issue asked to try it first on the one-thread kernels (where there are no
cross-lane carries), then on the cooperative kernels. This note covers the first half: two dedicated-squaring kernels, their
correctness checks, and their timings against the existing kernels on an RTX 4070 Laptop and an RX 6700 XT.

**Result in one paragraph.** On the RTX 4070 Laptop the dedicated squaring is 1.15x to 1.27x faster than the same-radix
baseline at 254 to 510 bits (two runs agree within 0.002), and 1.28x (13-bit) and 1.06x (32-bit) at 1022 bits, where the rows
are less clean. On the RX 6700 XT it is 1.03x to 1.15x at 254 to 510 bits. The measured gains are below the 1.28x to 1.33x that the product counts predict, as the CPU result in #46 (about 1.2x)
suggested. This does **not** show a gain for the cooperative kernels the engine runs at 2,029 bits (section 5).

## 1. What was built

`onethread.rs` (experimental, native only, not used by the runner or the CLI; the module comes from the earlier
limb-scheme experiment, `2026-11-24-limb-scheme-experiment.md`) gained two kernels. Both keep the radix, the limb count and
the input and output contract of their baseline, so the oracle comparison is the same.

| Kernel | Radix | Contract (same as its baseline) | Baseline |
|---|---|---|---|
| A-sqr | 13-bit limbs, 24-bit multiply-add | limbs below 2^13, value below 2n | A |
| B-sqr | 32-bit limbs, u64 products (`SHADER_INT64`) | value below n | B |

Both do the squaring in three phases instead of the fused word-by-word loop:

1. **Cross terms once.** `x_i * x_j` for `i < j`, accumulated by row into a 2L-word product.
2. **Double and add the diagonal.** In B-sqr the doubling and the diagonal `x_i^2` are one pass with a carry. In A-sqr the
   multiplicand is the doubled limb `2 x_j` (below 2^14), so doubling is free; the diagonal is added afterwards and one
   sequential carry pass normalises all 2L limbs.
3. **Word-by-word Montgomery reduction** of the 2L-word product, `m = t_i * ninv` per row.

Limb products per squaring, counted from the generators (not an instruction count):

| Kernel | Baseline | Dedicated squaring | Ratio |
|---|---|---|---|
| B, W = 8 / 12 / 16 / 32 limbs | 2 W^2 = 128 / 288 / 512 / 2048 | 1.5 W^2 + W/2 = 100 / 222 / 392 / 1552 | 1.28 / 1.30 / 1.31 / 1.32 |
| A, L = 20 / 30 / 40 / 79 limbs | 2 L^2 = 800 / 1800 / 3200 / 12482 | 1.5 L^2 + L/2 = 610 / 1365 / 2420 / 9401 | 1.31 / 1.32 / 1.32 / 1.33 |

**Carry headroom (the issue's main risk).** B-sqr needs no new argument: every `t + a * b + c` is at most 2^64 - 1 as in B,
and the total `x^2 + M n < n^2 + R n < 2^(64 W)` so the carry bit out of the last reduction row is zero (checked in the
generator's comment). A-sqr is where the bounds change, and they were redone for one thread per number:

- Cross terms use the doubled limb, so each product is below 2^27 (not 2^26). A row adds one product to a column, so a light
  carry pass every 20 rows keeps a column below 2^19 + 2^13 + 20 * 2^27, about 2.68e9, under 2^32.
- The diagonal adds at most 2^26 to a column; the following sequential pass sees at most about 2.75e9 plus a carry below 2^20.
- The reduction adds one product (below 2^26) per row per column, so a light pass every 40 rows keeps a column below about
  2.68e9, and `p + n0 * m` below about 2.75e9. Columns at or below the current row are dead (their carry went up through
  `carry0`), so a pass only covers the live window.

## 2. Correctness

Every kernel is compared with the oracle (`mont_square_chain_bits` at the kernel's radix) on every instance:
`crates/bigmont-core/tests/onethread.rs`, 10 tests, all passing on both GPUs below.

- All sizes of the existing test (64, 130, 256, 384, 512, 1024 bits) with instance counts that are not a multiple of the
  workgroup size, chains of 1 to 500 squarings, chained dispatches, and zero squarings.
- A-sqr at the headroom limits: 254, 256, 508, 510, 512, 520, 780, 1022, 1034 and 1500 bits (limb counts just below and above
  multiples of the 20 and 40 pass periods, and past the largest benchmarked size), with the all-ones modulus `2^bits - 1`, a
  sparse modulus, random moduli, and `x = 2n - 1` (the largest input scheme A allows), `x = n - 1` and random `x < 2n`.
- B-sqr at the edges: 32, 33, 62, 64, 254, 382, 510, 1022 and 1500 bits with the same moduli and `x = n - 1`, `x = 1`, random.
- The benchmark itself re-checks 27 sampled instances of the timed output against the oracle after every run; there were no
  failures and no rejected configurations in any of the three result files.

The naga validation unit tests for all five kernels (`onethread::tests`) were run on the desktop only; they do not depend on
the GPU.

## 3. Setup

| | Laptop | Desktop |
|---|---|---|
| GPU | NVIDIA GeForce RTX 4070 Laptop (8 GB) | AMD Radeon RX 6700 XT |
| Driver / backend | NVIDIA 580.178.04, Vulkan | AMD proprietary driver 26.10.1, Vulkan |
| OS | Ubuntu 26.04 | Windows 11 |
| Runs | 2 separate processes, 3 runs per configuration each | 1 process, 3 runs per configuration |
| Other load | a desktop session was running; `nvidia-smi` showed 12% GPU utilisation and 2175 MHz before each run | not checked |

```sh
BIGMONT_ADAPTER=4070 BIGMONT_BACKEND=Vulkan cargo run --release -p bigmont-core --example scheme_bench -- \
  --runs 3 --count 262144 --sizes 254,382,510,1022 --wgs 64,128,256 --schemes A,As,B,Bs
```

(`BIGMONT_ADAPTER=6700` on the desktop.) Method as in the earlier experiment: slope method on 262,144 instances, per
workgroup size in {64, 128, 256} and run, rate = instances x (S_long - 2) / (t_long - t_short); the table gives the workgroup
with the best median. Sizes 254, 382, 510 and 1022 bits are the slack-free sizes of the earlier fair-size rerun (B has no
unused limb capacity, A has 2 to 10 bits). Times are wall-clock around submit plus a blocking poll; clocks were not locked.

As a cross-check that the restored harness behaves like the original: baseline B/A on the 4070 is 1.86, 1.92, 1.86 and 2.61,
and on the 6700 XT 0.76, 0.74, 0.72 and 0.47, against 1.87, 1.92, 1.86, 2.61 and 0.77, 0.75, 0.72, 0.47 in the earlier
fair-size rerun.

## 4. Results

Squarings per second, median of 3 runs at the best workgroup size. Speedup is the dedicated squaring over its same-radix
baseline.

### RTX 4070 Laptop (run 1; run 2 in brackets)

| Bits | A | A-sqr | A-sqr / A | B | B-sqr | B-sqr / B |
|---|---|---|---|---|---|---|
| 254 | 6.825e9 [6.826e9] | 8.287e9 [8.286e9] | **1.21** [1.21] | 1.272e10 [1.272e10] | 1.462e10 [1.462e10] | **1.15** [1.15] |
| 382 | 2.933e9 [2.933e9] | 3.704e9 [3.703e9] | **1.26** [1.26] | 5.639e9 [5.637e9] | 6.816e9 [6.816e9] | **1.21** [1.21] |
| 510 | 1.701e9 [1.700e9] | 2.154e9 [2.155e9] | **1.27** [1.27] | 3.157e9 [3.157e9] | 3.871e9 [3.871e9] | **1.23** [1.23] |
| 1022 | 2.923e8 [2.919e8] | 3.730e8 [3.732e8] | 1.28 [1.28] (see below) | 7.638e8 [7.642e8] | 8.088e8 [8.090e8] | 1.06 [1.06] |

The min to max spread over the 3 runs inside one process is at most 0.16% at 254 to 510 bits. At 1022 bits it is at most
0.2% in run 2, but in run 1 one configuration (A-sqr) spread 3.9%. The medians of the two processes agree within 0.15%
everywhere. The two processes ran back to back on the same boot, so they are not independent of thermal state.

### RX 6700 XT (single process)

The spread over the 3 runs is at most 0.41% at 254 to 510 bits.

| Bits | A | A-sqr | A-sqr / A | B | B-sqr | B-sqr / B |
|---|---|---|---|---|---|---|
| 254 | 6.197e9 | 6.684e9 | **1.08** | 4.721e9 | 4.868e9 | **1.03** |
| 382 | 2.887e9 | 3.179e9 | **1.10** | 2.145e9 | 2.253e9 | **1.05** |
| 510 | 1.686e9 | 1.937e9 | **1.15** | 1.215e9 | 1.291e9 | **1.06** |
| 1022 | 2.851e8 | 2.890e8 | 1.01, unstable (see below) | 1.350e8 | 3.246e8 | 2.40, not symmetry (see below) |

### The 1022-bit rows are not clean

- **4070, A at 1022 bits:** the earlier experiment found that A's 79-limb kernel may spill registers at this size. A-sqr
  keeps a 2L-word product live, so spills may differ between A and A-sqr and the 1.28x may mix register effects with the
  arithmetic gain. Pipeline statistics were not collected for the new kernels.
- **4070, B-sqr at 1022 bits** gains only 1.06x against 1.15x to 1.23x at the smaller sizes. Not investigated.
- **6700 XT, A-sqr at 1022 bits:** the three runs at the best workgroup size were 3.28e8 (the first), 2.89e8 and 2.89e8 sq/s. The median
  ratio of 1.01 and the best-case ratio of 1.15 differ by a lot; neither should be quoted.
- **6700 XT, B at 1022 bits** is already anomalously slow (B/A of 0.47, as in the earlier experiment), and B-sqr is not. The
  2.40x is a property of the baseline kernel at this size on this driver, not a symmetry gain. Not investigated.

## 5. What this means, and what it does not

- **Gains are below the product-count ratio.** Where the rows are clean, A-sqr gains 1.21x to 1.27x on the 4070 against
  1.31x to 1.32x fewer limb products, and B-sqr 1.15x to 1.23x against 1.28x to 1.31x. The extra work is the doubling, the
  extra carry passes, the 2L-word product (twice the live state of the fused loops) and, for B-sqr, the same final
  conditional subtraction. This was not profiled.
- **The gain depends on the vendor.** NVIDIA gains more (1.15x to 1.27x) than AMD (1.03x to 1.15x) in this one-thread
  microbenchmark, and the 32-bit scheme gains least on AMD. One device per vendor, one driver each.
- **Not shown: the cooperative kernels.** The engine runs a cooperative kernel (TPI of 4 or more lanes per number), not one
  thread per number. These measurements say nothing about it. The reasoning below is untested:
  - *Lane balance.* In the cooperative loop every lane multiplies the broadcast limb of the owning lane by its own limbs.
    Symmetry needs only the limbs above the owner, so lanes whose limbs are all at or below it idle while the others work. With
    lanes in lockstep, skipping those products saves no time unless limbs are redistributed across lanes (for example folding
    lane `l` with lane `TPI - 1 - l`). Only the cross-term phase is triangular; the reduction phase is not.
  - *Carry headroom across lanes.* The 32-bit cooperative kernel defers the carry out of each lane's last limb by one step
    (`pend` below 2^34). Doubling cross terms and separating squaring from reduction changes those bounds; the argument above
    covers one thread per number only. For the 13-bit scheme at 2,029 bits (157 limbs) the light passes need cross-lane
    shuffles.
  - *Registers.* A 2L-word product doubles the per-lane limb state, which may cost occupancy or spill.
- **Not tested:** Intel, Apple, the Radeon 780M, software rasterisers, and a browser. A-sqr uses no 64-bit integers and would
  be a candidate for WebGPU, but no browser run was done. B-sqr is native only (`SHADER_INT64`).
- **Timing caveats:** wall-clock timing, clocks unlocked, other processes sharing the laptop GPU, desktop load unchecked, and
  one driver per vendor.

## 6. Reproducing

The experiment code is on branch `spike/gpu-squaring-symmetry` (commit `c9dc715`), not merged: it restores the earlier
experimental `onethread.rs`, its oracle test and the `scheme_bench` example, and adds A-sqr and B-sqr. Raw outputs are in
`docs/results/squaring-symmetry/` (`scheme_bench-4070-run1.txt`, `scheme_bench-4070-run2.txt`, `scheme_bench-6700xt.txt`).

```sh
git checkout spike/gpu-squaring-symmetry
cargo test --release -p bigmont-core --test onethread -- --test-threads=1
cargo run --release -p bigmont-core --example scheme_bench -- --runs 3 --sizes 254,382,510,1022 --schemes A,As,B,Bs
```

## 7. Recommendation

Do not merge the kernels. They change nothing the runner or the CLI does. A cooperative version is worth trying only if a
cheap probe shows the lane-balance problem can be solved: the 1.15x to 1.27x seen here on NVIDIA is an upper bound for it, and
AMD's 1.03x to 1.15x is a weak case for the 13-bit scheme that browsers use. Issue #2 stays open for that work.
