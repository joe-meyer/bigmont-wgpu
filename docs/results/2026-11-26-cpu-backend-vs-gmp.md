# CPU backend against GMP, and the squaring speedup (2026-11-26)

Issue #40. The pure-Rust CPU backend (`bigmont-cpu`) had never been timed against GMP on the same machine. This note records
that comparison, an ablation of where the gap came from, and the result of the change that followed.

## Setup

- CPU: AMD Ryzen 7 7840HS (8 cores, 16 threads), idle, Linux (Ubuntu 26.04), governor `powersave` (amd-pstate active),
  left as found. Rust 1.100.0, `cargo build --release` (no `target-cpu` flag).
- GMP 6.3.0 from the Ubuntu package, linked statically; `tools/cpu-baseline/gmp_modexp.c` unchanged
  (`gcc -O2 ... -lgmp`). That build is generic x86-64: its code contains no `mulx`/`adx`, so a CPU-tuned GMP would be faster.
- Work: base-2 Fermat test, 2,029-bit random odd moduli (`bigmont-cpu/examples/rate.rs` for Rust, `gmp_modexp 2029 N seed`
  for GMP; different random generators, same size). Single-thread figures are pinned to one core and interleaved; the first run
  of each in a session is a cold outlier and is dropped. `perf` was not available (`perf_event_paranoid=4`), so the
  attribution below is an ablation, not a profile.

## 1. Before the change (commit 806eee6)

| | GMP `mpz_powm` | `bigmont-cpu` | ratio |
|---|---|---|---|
| 1 thread | 650 to 661 tests/s (7 runs) | 289 to 293 tests/s (8 runs) | 2.25x |
| 16 threads (GMP: 16 processes) | 5,301 to 5,325 tests/s (4 runs) | 1,990 to 2,123 tests/s (5 runs) | 2.5x |

## 2. Where the gap came from (throwaway variants, one pinned core, 40 moduli)

| Variant | tests/s | vs shipped |
|---|---|---|
| shipped `fermat_base2` | 294 | 1.00x |
| shipped, `-C target-cpu=native` (23 `mulx`/`adx` emitted) | 293 | 1.00x |
| same CIOS algorithm, bounds checks removed | 357 | 1.21x |
| + dedicated squaring, then separate reduction | 435 | 1.48x |
| + `target-cpu=native` | 502 | 1.71x |
| GMP | 656 | 2.23x |

The disassembly of the old `Mont::mul` showed two slice bounds checks per limb, the accumulator kept in memory, and
`u128` multiply-add-add compiled to `mulq` plus two `add`/`adc` pairs. A dedicated squaring does about 1.5 w^2 limb
products (w^2/2 for the cross terms and diagonal, w^2 for the reduction) against 2 w^2 for `mul(a, a)`, so about 1.33x
fewer products; 1.22x was measured on top of the bounds-check removal.

## 3. After the change (this branch)

`Mont::sqr` is a dedicated squaring plus a word-by-word reduction that parks each row's carry in the dead low limb;
`Mont::mul` runs its loops over equal-length sub-slices, and the multiply-accumulate step is written for fewer instructions. No `unsafe`. The disassembly of `Mont::sqr` has no
bounds-check panics. Same machine and method as section 1, interleaved with the old build and GMP:

| | before | squaring + no bounds checks | + compact multiply-accumulate (this branch) | GMP |
|---|---|---|---|---|
| 1 thread | 290 to 294 tests/s (5 runs) | 432 to 434 (5 runs, 1.48x) | 477 to 480 (5 runs, 1.63x) | 655 to 663 |
| 16 threads | 2,026 to 2,135 (5 runs) | 3,111 to 3,369 (5 runs, 1.58x) | 3,500 to 3,514 (5 runs, 1.65x) | not re-run |

`mac` is the one-limb multiply-accumulate every inner loop uses. Written as a 128-bit sum of three terms, LLVM emits about
10 instructions per limb product (it builds each carry bit in a register with `xor`/`setb`/`mov`); written as one widening
multiply and two `overflowing_add`s it emits about 9, and the test is 10% faster on one thread (5% on 16, where the cores
share more). Neither form reaches the compact `add`/`adc $0` sequence GMP's loop uses.

The gap to GMP is now about 1.4x on one thread; the rest is GMP's hand-written assembly kernels.

Not a bottleneck, measured in a scratch copy (not in this branch): interleaving 2 to 6 independent candidates in one loop
(each lane's carry chain in registers, checked in the disassembly) gave no speedup (434 against 420 to 444 tests/s), so
the loop is limited by instruction count per limb product, not by carry-chain latency.

## 4. Hardware kernel: `mulx` with `adcx`/`adox` (this PR)

The loop that squares and reduces is one row routine, `t += m * src`. On x86-64 CPUs that report both BMI2 and ADX it is a
four-limb assembly body (`mulx` for the product, `adcx` for the previous limb's high word, `adox` for the accumulator word,
so two independent carry chains); everywhere else, and on request through `Mont::portable()`, it is the safe Rust loop of
section 3. The choice is made once, when a `Mont` is built.

### Why this and not something else (laptop Zen 4, one pinned core, 48 moduli, throwaway variants in a scratch copy)

| Variant | tests/s |
|---|---|
| section 3 code | 482 |
| interleave 2 to 6 independent candidates in one loop | 420 to 444 (no gain) |
| `_addcarry_u64` chain (safe, GMP's `mulq` instruction pattern) | 473 to 487 |
| same, loop unrolled 4x | 469 |
| `mulx` + plain `adc`, one carry chain | 512 |
| `mulx` + `adcx`/`adox`, two chains, unrolled 4x | 738 to 750 |

The same `mulx` + `adc` against `mulx` + `adcx`/`adox` comparison on a Ryzen 7 5800X (Zen 3): 437 to 450, 483 to 487, 704
to 706. On both Zen generations `mulx` alone is worth 6 to 11%; the second carry chain is the lever.

Hardware counters (laptop, `perf stat`, per limb product; a squaring has about 1,552 of them):

| | section 3 code | `mulx` + `adcx`/`adox` | GMP 6.3.0 (Ubuntu build) |
|---|---|---|---|
| instructions | 12.5 | 7.6 | 7.0 |
| micro-ops | 14.4 | 8.7 | 8.6 |
| cycles | 3.31 | 2.16 | 2.43 |

Scheduler-full, load-queue and store-queue stalls were under 1% of cycles for the section 3 code, and retire-blocked cycles
about 5%: the loop was not waiting, it was issuing about 4.3 micro-ops per cycle, so the saving comes from executing fewer of
them. Interleaving candidates therefore had no spare slots to fill.

GMP itself (Valgrind instruction profile of the same Ubuntu build, 2,029 bits): 64% in `mpn_redc_1`, 33% in
`mpn_sqr_basecase`, about 3% in shifts and glue; it special-cases base 2 through `mpn_2powm`. The same algorithm shape as
ours, so what separates the two is kernel quality, not a missing algorithm. That build is a generic x86-64 one with no
`mulx`/`adx`; a GMP built for the CPU would be faster than the 659 tests/s used as the reference here.

### Results (2,029-bit base-2 Fermat, `crates/bigmont-cpu/examples/rate.rs`, release build)

| CPU | main | #46 (portable) | this PR | GMP |
|---|---|---|---|---|
| Ryzen 7 7840HS (Zen 4), Linux, 1 thread | 290 to 293 | 477 to 480 | 730 to 747 | 659 to 663 |
| Ryzen 7 7840HS, 16 threads | 2,107 to 2,139 | 3,298 to 3,519 | 5,266 to 5,698 | 5,301 to 5,325 (16 processes) |
| Ryzen 7 5800X (Zen 3), Windows, 1 thread | 274 to 279 | 446 to 457 | 737 to 744 (three runs of eleven were disturbed by background load: 599 to 729) | not measured |
| Xeon E5-2690 v2 (Ivy Bridge, no BMI2 or ADX), Linux container on a shared host (load about 3.7 on 4 vCPU), 1 thread | 57 to 137 (very noisy) | not run (this CPU runs the same portable code) | 194 to 203 | not re-run |

On the Xeon the hardware kernel is not used (the CPU has no BMI2 or ADX); the run shows that the fallback is selected, passes
the tests, and keeps the section 3 gains. Its timings are noisy because the container shares a host.

## 5. Not covered

CPUs measured: Zen 4, Zen 3, Ivy Bridge. Not measured: Zen 1 and Zen 2, any Intel CPU with ADX (Broadwell and later), Haswell
(BMI2 without ADX; this PR uses the portable code there, since on Zen `mulx` alone gained only 6 to 11%), and any non-x86
CPU (the crate compiles and passes clippy for wasm32; aarch64 was not built). The assembly row is checked against the
portable row on lengths 0 to 40 and on all-ones and alternating operands, and the Fermat and Gerbicz tests run through it
on every CPU that has ADX. The `xor` that clears CF and OF before the first limb is load-bearing, but removing it is not
detected by the tests (the flags happen to be clear at entry), so it carries a comment. The full workspace suite was not run
(only `bigmont-cpu`, `bigmont-cli` and `bigmont-oracle`).

## 6. Earlier caveats (sections 1 to 3)

One CPU model (Zen 4). The Xeon E5-2690 v2 figures in the Phase 1 report were not reproduced (section 4 has a noisy new run), and Ivy Bridge has no
`mulx`. The 16-thread Rust runs are short (400 tests, about 0.12 s after the change), so thread start-up and the static
chunking of `fermat_batch` are included. Correctness: the unit and differential tests in `crates/bigmont-cpu/tests/cpu.rs`
(oracle comparison for every odd number below 8,192, Mersenne numbers, random sizes, the Gerbicz fault drill) pass, plus new
tests that check `mul` and `sqr` against the Montgomery definition at 1 to 33 limbs on carry-stressing moduli and operands.
