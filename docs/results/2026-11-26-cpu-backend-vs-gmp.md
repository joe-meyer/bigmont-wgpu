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

## Not covered

One CPU model (Zen 4). The Xeon E5-2690 v2 figures in the Phase 1 report were not re-measured, and Ivy Bridge has no
`mulx`. The 16-thread Rust runs are short (400 tests, about 0.12 s after the change), so thread start-up and the static
chunking of `fermat_batch` are included. Correctness: the unit and differential tests in `crates/bigmont-cpu/tests/cpu.rs`
(oracle comparison for every odd number below 8,192, Mersenne numbers, random sizes, the Gerbicz fault drill) pass, plus new
tests that check `mul` and `sqr` against the Montgomery definition at 1 to 33 limbs on carry-stressing moduli and operands.
