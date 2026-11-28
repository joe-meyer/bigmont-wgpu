# Symmetric squaring for the 32-bit cooperative kernel (2026-11-27)

Issue #2. A Montgomery squaring needs about three quarters of the limb products of a general multiply, because `x_i * x_j`
appears twice in `x^2`. This note covers the first cooperative kernel that uses it: `Squaring::Symmetric` for the 32-bit limb
scheme (`crates/bigmont-core/src/limb32sq.rs`), offered to the autotuner as an extra candidate next to the existing fused loop.
The one-thread experiment that preceded it is in `2026-11-26-gpu-squaring-symmetry.md`.

**Result in short.** (This change also fixes the tuner's workgroup choice, section 5, worth about 6% on the laptop.) On the RTX 4070 Laptop the symmetric kernel is 5.6% (TPI 4) and 3.8% (TPI 8) faster than the fused one at
2,029 bits in the kernel-only benchmark, and 0.9% to 1.8% (TPI 4) and 2.4% to 3.9% (TPI 8) faster end to end through the real
runner, depending on workgroup size. On the RX 6700 XT, where the 32-bit scheme is not the winner, it is 17% to 19% faster kernel-only and about 14.5% end to
end when the 32-bit scheme is forced. It is a measured candidate: the tuner picks it only where it wins. The end-to-end gain on
NVIDIA is clearly smaller than the kernel-only gain, and the reason is not established (section 4).

## 1. What was built

Phase 1 forms `S = x^2` as `2 L` words; phase 2 is the existing word-by-word reduction without the `x * b` products, fed by the
high words of `S`.

- **Phase 1 schedule:** the "rediagonalisation" of cuda-fixnum's wide squaring (after Ozturk, Guilford and Gopal 2013) with a
  lane's whole limb run as the digit. In iteration `i` of `TPI / 2`, lane `l` multiplies blocks `(i, l + i)` or, when `l + i`
  wraps, `(l + i - TPI/2, TPI/2 + i)`. Both have block column `l + 2 i`, so every product lands at the lane's own position in a
  window that slides down one lane per half-iteration, and no partial product is ever delivered to another lane. Every unordered
  block pair is visited once. Ideas only; no code was copied.
- **Doubling:** the second operand of each block product is doubled before the product (an `LPL`-word shift plus one top bit,
  added separately), so no u64 chain overflows; lane 0, whose pair is `(i, i)`, uses the undoubled operand.
- **Carries:** the carry out of a lane's top word stays in the lane across the window shift (the same deferred-carry idea as the
  fused kernel). The bounds are in the module comment of `limb32sq.rs`.
- **Product counts per lane at 2,029 bits** (counted from the generator): TPI 4, 16 limbs per lane: 1,544 against 2,048 (0.75x);
  TPI 8: 772 against 1,024 (0.75x). Phase 1 as built does 648 products per lane at TPI 4 against an ideal 520, because the upper
  diagonal blocks are squared redundantly by two lanes each.
- **Prior art:** NVIDIA's CGBN does not use symmetry (`mont_sqr` is `mont_mul(a, a)`). cuda-fixnum has the schedule above for one
  limb per lane and reports it pays off only at 32 lanes. Here each lane holds 8 to 16 limbs, so the products dominate the
  shuffles.

Integration: `KernelConfig::squaring` (`Fused` default); the tuner offers `Symmetric` at TPI 4 and 8 for the 32-bit scheme (TPI
16 lost on the NVIDIA card and never beat TPI 4 or 8 on the AMD one); the cache gets the word `unrolled32sym` and older caches
still read; multiply mode (the Gerbicz check) and the 13-bit scheme are unchanged. 32-bit kernels are native-only, so this never
reaches a browser.

## 2. Setup

| | Laptop | Desktop |
|---|---|---|
| GPU | NVIDIA GeForce RTX 4070 Laptop | AMD Radeon RX 6700 XT |
| Driver / backend | NVIDIA 580.178.04, Vulkan | AMD proprietary driver 26.10.1, Vulkan |
| OS | Ubuntu 26.04 | Windows 11 |
| Other load | a desktop session was running | not checked |

Kernel-only timings: `scheme_compare --bits 2029 --count 65536 --runs 3 --tpis 4,8,16 --wgs 64,128,256 --schemes 32
--squarings fused,symmetric` (slope method, oracle-verified after every run, 3 repetitions), median at the best workgroup size per
row. End-to-end: `fermat_stats --limbs 32 --squaring fused|symmetric CAND` on 131,075 random 2,029-bit candidates (`make_candidates
--bits 2029 131072 20261127`), capacity 16,384 (the engine default), alternating fused and symmetric runs. Times are wall-clock;
clocks were not locked. The kernel timings come from the commit that introduced the generator; later commits changed tests and
examples only.

## 3. Results

### Kernel only (squarings per second, best workgroup)

| GPU | TPI | Fused | Symmetric | Gain (3 repetitions) |
|---|---|---|---|---|
| RTX 4070 Laptop | 4 | 1.708e8 | 1.803e8 | +5.6%, +5.7%, +5.6% |
| RTX 4070 Laptop | 8 | 1.564e8 | 1.624e8 | +3.8%, +3.7%, +3.8% |
| RTX 4070 Laptop | 16 | 1.294e8 | 1.224e8 | -5.4% (not offered) |
| RX 6700 XT | 4 | 5.974e7 | 7.006e7 | +17.3%, +17.3%, +17.5% |
| RX 6700 XT | 8 | 5.472e7 | 6.528e7 | +19.3%, +19.0%, +19.3% |
| RX 6700 XT | 16 | 4.670e7 | 5.276e7 | +13.0%, +13.0%, +12.9% |

The 13-bit scheme (unchanged) runs at 9.74e7 (TPI 4) and 8.81e7 (TPI 8) on the 6700 XT, so it stays the faster scheme there.

### The autotuner (`bigmont tune --bits 2029`)

| GPU | `--limbs 32` winner | `--limbs auto` winner |
|---|---|---|
| RTX 4070 Laptop | symmetric, TPI 4, wg 256, 1.563e8 sq/s (best fused: TPI 4, wg 256, 1.516e8) | the same symmetric kernel, 1.577e8 |
| RX 6700 XT | symmetric, TPI 4, wg 256, 5.489e7 (best fused: 5.065e7) | 13-bit, TPI 8, wg 256, 7.537e7 (unchanged) |

Per configuration the tuner's own measurement shows the symmetric kernel ahead everywhere except TPI 8 workgroup 64 on the
laptop (-0.8%); see `tune32-*.txt`.

### End to end (tests per second through `Runner`, 2,029 bits, capacity 16,384)

| GPU | Configuration | Fused | Symmetric | Gain |
|---|---|---|---|---|
| RTX 4070 Laptop | tuner's choice: TPI 4, wg 256 (3 repetitions) | 68,149 / 67,928 / 67,743 | 68,764 / 68,633 / 68,858 | +0.9%, +1.0%, +1.6% |
| RTX 4070 Laptop | TPI 4, wg 64 (2 repetitions) | 71,875 / 71,902 | 73,147 / 73,066 | +1.8%, +1.6% |
| RTX 4070 Laptop | TPI 4, wg 128 | 71,765 / 71,634 | 73,081 / 72,813 | +1.8%, +1.6% |
| RTX 4070 Laptop | TPI 8, wg 64 | 67,736 / 67,823 | 70,267 / 70,245 | +3.7%, +3.6% |
| RTX 4070 Laptop | TPI 8, wg 128 | 67,762 / 67,890 | 70,413 / 69,910 | +3.9%, +3.0% |
| RX 6700 XT (32-bit forced) | TPI 4, wg 256 (3 repetitions) | 25,051 / 24,943 / 24,944 | 28,588 / 28,611 / 28,564 | +14.1%, +14.7%, +14.5% |

Without forcing, the 6700 XT runs the 13-bit scheme (TPI 8) at 36,036 tests/s, far ahead of either 32-bit kernel. Capacity
sweep on the laptop (TPI 4, wg 256): +3.0% at 4,096, +1.2% at 16,384 and +0.3% to +0.9% at 65,536 instances
(`endtoend-4070-capacity-sweep.txt`).

### Registers (driver report, RTX 4070 Laptop, 2,029 bits)

| Kernel | Registers | Binary | Local memory |
|---|---|---|---|
| fused, TPI 4 | 80 | 46 KB | 0 B |
| symmetric, TPI 4 | 128 | 89 KB | 0 to 96 B depending on workgroup and mode |
| fused, TPI 8 | 64 | 15 KB | 0 B |
| symmetric, TPI 8 | 72 to 80 | 42 KB | 0 B |

## 4. What is not explained

- **Why the end-to-end gain on NVIDIA is smaller than the kernel-only gain.** The plain squaring benchmark at 16,384 instances,
  the runner's batch size, still shows +5.4% (TPI 4) and +3.8% (TPI 8), so the batch size is not the cause. Issuing the
  exponent-bit load before the squaring made no difference (68.5k to 68.7k tests/s with it, 68.6k to 68.9k without, three repetitions), and the Fermat-mode kernels do not
  spill at workgroup 64. Fermat mode costs the symmetric kernel about 0.2 ns more per iteration than the fused one, which is
  roughly the missing gain. Without a profiler I could not attribute it. On the 6700 XT the end-to-end gain keeps most of the
  kernel gain (14.5% of 17.3%).
- **Which part of phase 1 costs the most.** Ablations (removing a part and timing phase 1 alone) show the upper diagonals cost
  about 13% of phase 1 on the laptop and about 21% on the 6700 XT at TPI 4; the shuffle, finishing and ripple code costs close to
  nothing (replacing the gather shuffles by ALU operations left the time unchanged). The first ablation I ran, which aliased the
  gathered operands to `x`, overstated the gather cost because it let the compiler share work between mirrored products; it was
  discarded.

## 5. The tuner's workgroup choice (fixed in the same change)

The tuner always picked workgroup 256 on the laptop, and the end-to-end sweep shows that is about 6% slower than workgroup 64 or
128 for both squaring structures (TPI 4 fused: 67.9k against 71.9k tests/s). `tune::measure` timed every configuration with a
fixed number of workgroups (256), so the instance count depended on the workgroup size (at TPI 4, 4,096 instances at wg 64
against 16,384 at wg 256) while the runner always uses a capacity of 16,384: small workgroups were under-filled in the
measurement and looked slower than they are in the engine. It now times every candidate on 16,384 instances (rounded up to whole
workgroups).

| After the fix | Tuner's pick | End to end (tests/s, wall) |
|---|---|---|
| RTX 4070 Laptop | symmetric, TPI 4, wg 64 or 128 | 73,493 / 73,595 / 73,300 (before: 68.7k with the symmetric kernel at wg 256, 67.9k with the fused kernel at wg 256) |
| RX 6700 XT | 13-bit, TPI 8, wg 64 or 256 (near tie) | 35,798 / 35,977 / 35,966 (before: 36,036; no change) |

So on the laptop the real rate is about 8% above the engine before this change (73.5k against 67.9k): roughly 1.7% from the
symmetric kernel and the rest from the tuner. `--limbs 13` picks are affected the same way. Raw: `endtoend-4070-after-tuner-fix.txt`,
`tuneauto-4070-after-tuner-fix.txt`.

## 6. Correctness

- The existing 32-bit chain tests (`limb32.rs`) now run both squaring structures; the Fermat tests (`fermat.rs`) run both shapes,
  including every odd number from 3 to 4,095, Mersenne primes and composites, edge lengths around limb boundaries and a mixed
  batch at 2,029 bits spanning chunks; `source_matrix.rs` validates every symmetric source with naga and checks that it is never
  generated for the browser and that multiply mode keeps the fused generator; the tuner tests cover the new cache word and
  candidates.
- Run on the RTX 4070 Laptop (the whole workspace, plus a rerun of the CLI test files after updating the tests that assumed every cached 32-bit winner was fused) and
  on the RX 6700 XT (the core suites and the CLI engine and tuner files). All pass. One test failed once on the laptop with
  "Parent device is lost" in the middle of a long run; it passes on its own and in the rerun, and I attribute it to a transient
  GPU reset, which I did not investigate.
- During development phase 1 was also compared word for word against the exact `x^2` for random and adversarial full-width inputs
  at TPI 4 to 32 and 254 to 2,029 bits (a debug mode that is not part of this change).

## 7. Not tested

Intel, Apple, the Radeon 780M, software rasterisers; a browser (the 32-bit scheme never runs in one). TPI 16 and 32 symmetric
kernels exist but are not offered to the tuner. Timing is wall-clock with unlocked clocks, one driver per vendor, and the laptop had
a desktop session running.
