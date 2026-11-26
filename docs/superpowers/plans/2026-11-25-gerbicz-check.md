# Gerbicz error check (opt-in) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An opt-in Gerbicz-Li check (`--gec`) for the `bigmont` driver that matches the CUDA driver's contract (verdict byte 2 on an alarm, `--inject IDX BDRY` fault drill, `gec_alarm=N` in the summary), with overhead in the range the CUDA driver measured (about 11.5 %).

**Architecture:** The existing ladder kernel is untouched (no cost, no risk when GEC is off). One new kernel, a general Montgomery multiply (`x <- x * b * R^-1`), is generated in both limb schemes. `Runner` drives the check from the host: per 40-iteration block one submission runs the ladder, then `A <- A*y_prev`, `B <- B*x`, `y_prev <- x`; after the last block `A` is squared 40 times and multiplied by `2^(sum V)` (computed with the same ladder kernel from a host-computed exponent), then compared with `B` on the host. The engine and CLI map an alarm to verdict byte 2.

**Tech Stack:** Rust 2024 (MSRV 1.90), wgpu 30 + generated WGSL, `num-bigint`.

**Spec:** GitHub issue #17 and #12 (`gh issue view 17`), and the baseline implementation `fermat_gec` in `N:\Programming\beacon\cuda\g1_fermat_v3.cu` (lines 56-64 and 113-128 for constants, 172-276 for the algorithm), its drill in `N:\Programming\beacon\tools\g2_harness.py` (lines 281-292: `--inject 1 3` on 3 candidates of 2048 bits must give verdict 2 for candidate 1 and unchanged verdicts for 0 and 2) and its gate `tools/gates_20261006/nogec_gate.sh` (GEC and non-GEC verdicts byte-identical).

## Global Constraints

- Correctness beats speed: every kernel change is checked against the CPU reference (`bigmont-oracle`) on the same inputs, including edge cases (n = 2^k - 1, n = 2^(k-1) + 1, operands near 4n).
- Both limb schemes keep working. The 13-bit kernels must not need `SHADER_INT64` (they also target browsers); the 32-bit kernels are native-only.
- `bigmont-core` must keep compiling for wasm32 (CI `wasm-check`): no blocking waits, threads or `std::time::Instant` outside `#[cfg(not(target_arch = "wasm32"))]`; real work in `async fn`s with blocking native wrappers.
- Handle device loss: GPU calls return `Err`, never panic or hang. No `unsafe`.
- GEC is **opt-in**, default off (the reference defaults it off since v3.5). With GEC off, behaviour and speed are exactly as before.
- Block size is `GEC_BLOCK = 40` ladder iterations (the reference's `w*J = 5*8`). The last `(bits - 1) % 40` iterations run unprotected, as in the reference (documented, not fixed here).
- Exit statuses: 2 = usage/input/config, 3 = failure while testing, canary mismatch 1 (unchanged).
- Clean runs must never alarm; verdicts with GEC on must be byte-identical to GEC off on clean runs.
- Never commit machine-identifying details (hostnames, usernames, local paths). Describe hardware by model only. Do not invent performance numbers; only cite what was run, naming hardware and driver.
- Do not bump versions or touch `.github/workflows/release.yml`, `LICENSE`, version/licence settings. Add user-visible changes under `## [Unreleased]` in `CHANGELOG.md`.
- Commits: Conventional Commits, small and focused, each ending with the lines
  `Assisted-by: Claude Code (Claude Sonnet 5.5)` and `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.
- Before each commit that touches Rust: `cargo fmt --all` and `cargo clippy --workspace --all-targets --all-features -- -D warnings`. GPU tests are strict: run with `cargo test --workspace --no-fail-fast -- --test-threads=1` (set `BIGMONT_ALLOW_SKIP=1` only if the machine has no GPU, and say so).

## The math (read once; every task relies on it)

The ladder state after `i` iterations is `x_i`, with `x_0 = 2` (the top bit of `e = n - 1`, which always has exactly `bits` bits for odd `n` of `bits` bits, is consumed) and iteration `i` doing `x <- x^2`, then `x <- 2x` if bit `bits - 2 - i` of `e` is set. Over a block of `s = 40` iterations starting at iteration `i`: `x_{i+s} = x_i^(2^s) * 2^V` where `V = (e >> (bits - 1 - (i + s))) & (2^s - 1)`.

With `nb = (bits - 1) / s` full blocks, `y_j = x_{j*s}` (`y_0 = x_0`) and `V_j` the block exponents: `A = prod_{j=1..nb} y_{j-1}`, `B = prod_{j=1..nb} y_j`, and the identity `A^(2^s) * 2^(sum V_j) == B (mod n)` holds iff (with overwhelming probability) every block relation held. We use one span for the whole run (no rollback exists, so shorter spans buy nothing). `sum V_j < nb * 2^40`, which fits 64 bits.

Montgomery domain: `1~ = R mod n`, `x~ = xR mod n`, `mont_mul(a~, b~) = (ab)~`. `A` and `B` start at `1~`. All kernel outputs are in `[0, 2n)` for inputs below `4n` (needs `R > 16n`, which `KernelConfig::for_bits_scheme` guarantees), so two values are equal mod n iff they are equal or differ by exactly `n`: no division is needed on the host.

## File Structure

- `crates/bigmont-oracle/src/lib.rs` (modify): `GEC_BLOCK`, `mont_mul_bits`, `gec_blocks`, `gec_window_sum`, `gec_window_sum_words`, `gerbicz_model` (the algorithm on plain modular arithmetic, the executable spec).
- `crates/bigmont-oracle/tests/oracle.rs` (modify): tests for the above.
- `crates/bigmont-core/src/unrolled.rs`, `limb32.rs` (modify): `Mode::Mul` generation.
- `crates/bigmont-core/src/kernel.rs` (modify): `KernelConfig::try_mul_wgsl_for`.
- `crates/bigmont-core/src/gpu.rs` (modify): `run_mul` / `run_mul_async` (differential-test harness for the multiply kernel).
- `crates/bigmont-core/src/runner.rs` (modify): refactor of `fermat_chunk` into reusable steps, `Verdict`, `enable_gec`, `fermat_checked`, the fault hook.
- `crates/bigmont-core/tests/mul.rs` (create), `tests/gec.rs` (create): GPU differential tests.
- `crates/bigmont-cli/src/engine.rs`, `main.rs`, `early_exit.rs` (modify), `tests/cli.rs`, `tests/engine.rs` (modify): options, flags, alarms, drill.
- `README.md`, `CHANGELOG.md`, `docs/superpowers/results/2026-11-25-gerbicz-check.md` (create): docs and measurements.

---

### Task 1: Oracle - Montgomery multiply, window sums and the executable GEC model

**Files:**
- Modify: `crates/bigmont-oracle/src/lib.rs`
- Test: `crates/bigmont-oracle/tests/oracle.rs`

**Interfaces:**
- Produces (all `pub` in `bigmont_oracle`):
  - `pub const GEC_BLOCK: u32 = 40;`
  - `pub fn mont_mul_bits(n: &BigUint, a: &BigUint, b: &BigUint, r_bits: u32) -> BigUint` (a*b*R^-1 mod n, result in `[0, n)`)
  - `pub fn gec_blocks(bits: u32, s: u32) -> u32` (`(bits - 1) / s`)
  - `pub fn gec_window_sum(e: &BigUint, bits: u32, s: u32) -> u64`
  - `pub fn gec_window_sum_words(e_words: &[u32], bits: u32, s: u32) -> u64` (same value from little-endian u32 words; the fast form the runner uses)
  - `pub struct GecModel { pub consistent: bool, pub result: BigUint }`
  - `pub fn gerbicz_model(n: &BigUint, s: u32, fault: Option<(u32, u32)>) -> GecModel` (`fault = (iteration, bit)`: after iteration `iteration` completes, bit `bit` of the state is flipped and the state reduced mod n)

- [ ] **Step 1: Write the failing tests** (append to `crates/bigmont-oracle/tests/oracle.rs`; check that file's existing imports and add what is missing)

```rust
use bigmont_oracle::*;
use num_bigint::BigUint;
use num_traits::One;

#[test]
fn mont_mul_matches_plain_modular_arithmetic() {
    let mut rng = XorShift64::new(11);
    for bits in [3u32, 64, 200] {
        let limbs = limbs_for_bits(bits);
        let r_bits = LIMB_BITS * limbs as u32;
        let r = BigUint::one() << r_bits;
        for _ in 0..20 {
            let n = random_odd_modulus(&mut rng, bits);
            let a = random_below(&mut rng, &(&n * 4u32));
            let b = random_below(&mut rng, &(&n * 4u32));
            let got = mont_mul_bits(&n, &a, &b, r_bits);
            assert!(got < n, "result must be reduced");
            assert_eq!((&got * &r) % &n, (&a * &b) % &n, "bits {bits}");
        }
    }
}

#[test]
fn window_sum_words_matches_the_bigint_form() {
    let mut rng = XorShift64::new(12);
    for (bits, s) in [(2u32, 40u32), (41, 40), (42, 40), (81, 40), (300, 40), (2029, 40), (300, 7)] {
        for _ in 0..10 {
            let n = random_odd_modulus(&mut rng, bits);
            let e = &n - 1u32;
            let words = e.to_u32_digits();
            assert_eq!(
                gec_window_sum_words(&words, bits, s),
                gec_window_sum(&e, bits, s),
                "bits {bits} s {s}"
            );
        }
    }
}

#[test]
fn window_sum_adds_the_block_exponents() {
    // e = n - 1 with n = 2^10 + 1 is 2^10: one set bit at position 10 (bits = 11, 10 iterations).
    let n = (BigUint::one() << 10u32) + 1u32;
    // s = 5: blocks cover iterations 0..5 and 5..10, i.e. exponent bits 9..5 and 4..0, all zero.
    assert_eq!(gec_window_sum(&(&n - 1u32), 11, 5), 0);
    // n = 2^10 + 2^9 + 1: e = 2^10 + 2^9, bit 9 is the first bit a block reads: V_1 = 0b10000.
    let n = (BigUint::one() << 10u32) + (BigUint::one() << 9u32) + 1u32;
    assert_eq!(gec_window_sum(&(&n - 1u32), 11, 5), 0b10000);
}

#[test]
fn the_model_agrees_with_fermat_and_a_clean_run_is_consistent() {
    let mut rng = XorShift64::new(13);
    for bits in [2u32, 3, 41, 42, 81, 130, 300] {
        for _ in 0..8 {
            let n = random_odd_modulus(&mut rng, bits);
            let m = gerbicz_model(&n, GEC_BLOCK, None);
            assert!(m.consistent, "bits {bits} n {n}");
            assert_eq!(m.result, BigUint::from(2u32).modpow(&(&n - 1u32), &n), "bits {bits}");
        }
    }
}

#[test]
fn a_fault_inside_the_blocks_is_detected_and_one_in_the_tail_is_not() {
    let mut rng = XorShift64::new(14);
    let bits = 300u32; // 299 iterations: 7 blocks (0..280) and a 19-iteration tail
    for _ in 0..6 {
        let n = random_odd_modulus(&mut rng, bits);
        for it in [0u32, 19, 20, 39, 40, 139, 279] {
            let m = gerbicz_model(&n, GEC_BLOCK, Some((it, 17)));
            assert!(!m.consistent, "fault after iteration {it} must be detected");
        }
        let m = gerbicz_model(&n, GEC_BLOCK, Some((285, 17)));
        assert!(m.consistent, "a tail fault is not covered (documented limitation)");
        assert_ne!(m.result, BigUint::from(2u32).modpow(&(&n - 1u32), &n));
    }
}
```

- [ ] **Step 2: Run, expect compile failure**

Run: `cargo test -p bigmont-oracle --test oracle`
Expected: FAIL (unresolved names `mont_mul_bits`, `gec_window_sum_words`, ...).

- [ ] **Step 3: Implement** (append to `crates/bigmont-oracle/src/lib.rs`)

```rust
/// Gerbicz block size in ladder iterations: the CUDA driver's `w * J = 5 * 8`.
pub const GEC_BLOCK: u32 = 40;

/// x <- a * b * R^-1 mod n with R = 2^r_bits; the result is in `[0, n)`. `n` must be odd.
pub fn mont_mul_bits(n: &BigUint, a: &BigUint, b: &BigUint, r_bits: u32) -> BigUint {
    assert!(n.bit(0), "modulus must be odd");
    let mask = (BigUint::one() << r_bits) - 1u32;
    let ninv = inv_mod_pow2(n, r_bits);
    let p = a * b;
    // m = -p * n^-1 mod R, so that p + m*n is divisible by R
    let m = ((&mask + 1u32) - ((&p * &ninv) & &mask)) & &mask;
    ((p + m * n) >> r_bits) % n
}

/// Full Gerbicz blocks of `s` iterations in a ladder of `bits - 1` iterations.
pub fn gec_blocks(bits: u32, s: u32) -> u32 {
    (bits - 1) / s
}

/// Sum over the full blocks `j = 1..=gec_blocks(bits, s)` of the block exponent
/// `V_j = (e >> (bits - 1 - j*s)) & (2^s - 1)`, for the base-2 ladder over `e = n - 1`.
/// Panics if the sum does not fit 64 bits (`s` up to 40 and any size this crate supports fit).
pub fn gec_window_sum(e: &BigUint, bits: u32, s: u32) -> u64 {
    assert!((1..=48).contains(&s), "block size must be 1..=48");
    let mask = (BigUint::one() << s) - 1u32;
    (1..=gec_blocks(bits, s)).fold(0u64, |acc, j| {
        let v = ((e >> (bits - 1 - j * s)) & &mask)
            .to_u64_digits()
            .first()
            .copied()
            .unwrap_or(0);
        acc.checked_add(v).expect("window sum overflows 64 bits")
    })
}

/// `s` bits of the little-endian word array starting at bit `lo` (zero beyond its end).
fn bits_at(words: &[u32], lo: u32, width: u32) -> u64 {
    let (mut v, mut got) = (0u64, 0u32);
    while got < width {
        let pos = lo + got;
        let (w, off) = ((pos / 32) as usize, pos % 32);
        let take = (32 - off).min(width - got);
        let word = u64::from(words.get(w).copied().unwrap_or(0));
        v |= ((word >> off) & ((1u64 << take) - 1)) << got;
        got += take;
    }
    v
}

/// [`gec_window_sum`] over the little-endian u32 words of `e` (no big-integer shifts).
pub fn gec_window_sum_words(e_words: &[u32], bits: u32, s: u32) -> u64 {
    assert!((1..=48).contains(&s), "block size must be 1..=48");
    (1..=gec_blocks(bits, s)).fold(0u64, |acc, j| {
        acc.checked_add(bits_at(e_words, bits - 1 - j * s, s))
            .expect("window sum overflows 64 bits")
    })
}

/// What [`gerbicz_model`] found.
pub struct GecModel {
    /// Whether `A^(2^s) * 2^(sum V) == B (mod n)` held.
    pub consistent: bool,
    /// The ladder's final value (`2^(n-1) mod n` when no fault was injected).
    pub result: BigUint,
}

/// The Gerbicz check on plain modular arithmetic: the executable specification the GPU runner is
/// tested against. Runs the base-2 ladder over `e = n - 1` (`n` odd, at least 2 bits), multiplies the
/// state at every block boundary (every `s` iterations) into `A` (the previous boundary's state) and
/// `B` (the new one), and tests the identity at the end. `fault = Some((i, b))` flips bit `b` of the
/// state after iteration `i` completes and reduces it mod n. The last `(bits - 1) % s` iterations are
/// not covered by any block.
pub fn gerbicz_model(n: &BigUint, s: u32, fault: Option<(u32, u32)>) -> GecModel {
    let bits = n.bits() as u32;
    let e = n - 1u32;
    let two = BigUint::from(2u32);
    let mut x = &two % n;
    let (mut y_prev, mut a, mut b) = (x.clone(), BigUint::one(), BigUint::one());
    for i in 0..bits - 1 {
        x = (&x * &x) % n;
        if e.bit(u64::from(bits - 2 - i)) {
            x = (&x * 2u32) % n;
        }
        if let Some((fi, fb)) = fault
            && fi == i
        {
            x = (&x ^ (BigUint::one() << fb)) % n;
        }
        if (i + 1) % s == 0 {
            a = (&a * &y_prev) % n;
            b = (&b * &x) % n;
            y_prev = x.clone();
        }
    }
    let sum = gec_window_sum(&e, bits, s);
    let lhs = (a.modpow(&(BigUint::one() << s), n) * two.modpow(&BigUint::from(sum), n)) % n;
    GecModel {
        consistent: lhs == b % n,
        result: x,
    }
}
```

- [ ] **Step 4: Run, expect pass**

Run: `cargo test -p bigmont-oracle` then `cargo clippy -p bigmont-oracle --all-targets -- -D warnings`
Expected: all pass, no warnings. (If `consistent` is false for a clean run, the bug is in the model, not a test: the identity is `B = A^(2^s) * 2^sum` with `A`, `B` as defined in "The math".)

- [ ] **Step 5: Commit**

```bash
git add crates/bigmont-oracle
git commit -m "feat(oracle): Montgomery multiply, Gerbicz window sums and an executable GEC model"
```
(with the two trailer lines from Global Constraints)

---

### Task 2: The general Montgomery multiply kernel (both limb schemes) and its differential test

**Files:**
- Modify: `crates/bigmont-core/src/unrolled.rs`, `crates/bigmont-core/src/limb32.rs`, `crates/bigmont-core/src/kernel.rs`, `crates/bigmont-core/src/gpu.rs`
- Test: `crates/bigmont-core/tests/mul.rs` (create); extend `crates/bigmont-core/tests/source_matrix.rs` / `browser_source.rs` if they enumerate kernel sources (read them first; the 13-bit multiply source must compile for `Target::Browser`, and the 32-bit one must be refused for it).

**Interfaces:**
- Produces:
  - `unrolled::Mode::Mul` (a `pub(crate)` enum variant). Bindings: `0 params: vec4<u32>` (x = instance count, y = number of multiplications, normally 1), `1 n_limbs`, `2 ninv_buf`, `3 x_limbs` (read_write, the left operand and the result), `4 b_limbs` (read, the right operand). Semantics per instance: `x <- x * b * R^-1 mod n` (`R = 2^(limb_bits * limbs)`), result in `[0, 2n)` with every limb normalised (13-bit scheme: below `2^13`), for `x, b < 4n`.
  - `KernelConfig::try_mul_wgsl_for(&self, target: Target) -> Result<String, String>` (unrolled variant only; Err for rolled; Err for `Limb32` with `Target::Browser`).
  - `gpu::run_mul(gpu: &Gpu, cfg: KernelConfig, values: &[(BigUint, BigUint, BigUint)], times: u32) -> Result<Vec<u32>, String>` and `run_mul_async` (same arguments): each value is `(n, a, b)`; returns the result limbs (`values.len() * cfg.limbs` words) after `times` multiplications `a <- a*b*R^-1`.

- [ ] **Step 1: Write the failing differential test** (`crates/bigmont-core/tests/mul.rs`)

```rust
//! The general Montgomery multiply kernel against the oracle, both limb schemes.
mod common;
use bigmont_core::gpu::run_mul;
use bigmont_core::kernel::{KernelConfig, Variant};
use bigmont_core::scheme::LimbScheme;
use bigmont_oracle::*;
use num_bigint::BigUint;
use num_traits::One;

/// n values with awkward limb patterns plus random ones; operands up to 4n - 1 (what the ladder
/// feeds the kernel after a lazy doubling).
fn cases(bits: u32, seed: u64) -> Vec<(BigUint, BigUint, BigUint)> {
    let mut rng = XorShift64::new(seed);
    let mut ns: Vec<BigUint> = (0..40).map(|_| random_odd_modulus(&mut rng, bits)).collect();
    ns.push((BigUint::one() << bits) - 1u32); // every limb all ones
    ns.push((BigUint::one() << (bits - 1)) + 1u32); // long zero runs
    let mut out = Vec::new();
    for n in ns {
        let four_n = &n * 4u32;
        let top = &four_n - 1u32;
        out.push((n.clone(), random_below(&mut rng, &four_n), random_below(&mut rng, &four_n)));
        out.push((n.clone(), top.clone(), top.clone()));
        out.push((n.clone(), BigUint::from(0u32), top.clone()));
        out.push((n.clone(), BigUint::one(), BigUint::one()));
    }
    out
}

fn check(scheme: LimbScheme, tpi: u32, wg: u32, bits: u32) {
    let cfg = KernelConfig::for_bits_scheme(scheme, bits, tpi, wg, Variant::Unrolled);
    let Some(gpu) = common::open_gpu(cfg) else {
        return;
    };
    if !gpu.caps().satisfies(cfg.requires()) {
        common::skip(format!("{cfg:?}: the adapter lacks {}", gpu.caps().missing(cfg.requires()).join(" and ")));
        return;
    }
    let values = cases(bits, 5);
    let limbs = cfg.limbs as usize;
    let r_bits = cfg.r_bits();
    let out = run_mul(&gpu, cfg, &values, 1).expect("run_mul");
    for (i, (n, a, b)) in values.iter().enumerate() {
        let got_limbs = &out[i * limbs..(i + 1) * limbs];
        assert!(scheme.normalised(got_limbs), "{cfg:?} case {i}: limbs not normalised");
        let got = scheme.unpack(got_limbs);
        assert!(got < n * 2u32, "{cfg:?} case {i}: result not below 2n");
        assert_eq!(&got % n, mont_mul_bits(n, a, b, r_bits), "{cfg:?} case {i}: wrong value (n = {n})");
    }
}

#[test]
fn limb13_multiply_matches_the_oracle() {
    for tpi in [4, 8, 16, 32] {
        check(LimbScheme::Limb13, tpi, 64, 2029);
    }
    check(LimbScheme::Limb13, 8, 64, 100);
}

#[test]
fn limb32_multiply_matches_the_oracle() {
    for tpi in [4, 8, 16, 32] {
        check(LimbScheme::Limb32, tpi, 64, 2029);
    }
    check(LimbScheme::Limb32, 8, 64, 100);
}

#[test]
fn repeated_multiplication_chains_like_the_oracle() {
    let cfg = KernelConfig::for_bits_scheme(LimbScheme::Limb13, 300, 8, 64, Variant::Unrolled);
    let Some(gpu) = common::open_gpu(cfg) else {
        return;
    };
    let values = cases(300, 6);
    let out = run_mul(&gpu, cfg, &values, 3).expect("run_mul");
    let limbs = cfg.limbs as usize;
    for (i, (n, a, b)) in values.iter().enumerate() {
        let mut x = a.clone();
        for _ in 0..3 {
            x = mont_mul_bits(n, &x, b, cfg.r_bits());
        }
        assert_eq!(cfg.scheme.unpack(&out[i * limbs..(i + 1) * limbs]) % n, x % n, "case {i}");
    }
}

#[test]
fn the_multiply_source_is_generated_for_the_right_targets() {
    use bigmont_core::kernel::Target;
    let c13 = KernelConfig::for_bits_scheme(LimbScheme::Limb13, 2029, 8, 64, Variant::Unrolled);
    assert!(c13.try_mul_wgsl_for(Target::Native).is_ok());
    assert!(c13.try_mul_wgsl_for(Target::Browser).is_ok());
    let c32 = KernelConfig::for_bits_scheme(LimbScheme::Limb32, 2029, 8, 64, Variant::Unrolled);
    assert!(c32.try_mul_wgsl_for(Target::Native).is_ok());
    assert!(c32.try_mul_wgsl_for(Target::Browser).is_err());
    let rolled = KernelConfig::new(8, 64, Variant::Rolled);
    assert!(rolled.try_mul_wgsl_for(Target::Native).is_err());
}
```

- [ ] **Step 2: Run, expect compile failure** (`run_mul`, `try_mul_wgsl_for` missing)

Run: `cargo test -p bigmont-core --test mul --no-run`

- [ ] **Step 3: Generate the 13-bit multiply** (`unrolled.rs`)

The squaring and multiply kernels differ only in where the multiplier limbs come from. In the current generator the multiplier limb of step `w` is `bcast(x{w}, owner, base)` (in `iteration`) and the multiplicand is `x` (`X0`, `XN0`, `x{u}`). For `Mode::Mul` the multiplier is `b{w}` (limbs loaded from a new read-only binding) and the multiplicand stays `x`; Montgomery multiplication is symmetric, so this gives `x*b*R^-1`.

Edits:
1. `Mode`: add `/// params.y Montgomery multiplications x <- x * b * R^-1 per dispatch (b is binding 4, read-only).` `Mul,`.
2. `header()`: for `Mode::Mul` write the params comment `// x = instance count, y = multiplications` and, after binding 3, `g.line("@group(0) @binding(4) var<storage, read> b_limbs: array<u32>;   // right operand");`.
3. `load(g, lpl)` -> `load(g, lpl, mode)`: when `mode == Mode::Mul`, after the `n{c}` loads add `for c in 0..lpl { g.line(format!("let b{c} = b_limbs[off + {c}u];")); }`.
4. `iteration(g, lpl, w)` -> `iteration(g, lpl, w, mult)` with `mult: &str` = `"b"` for `Mode::Mul`, else `"x"`; the one changed line becomes `format!("let bl = bcast({mult}{w}, owner, base) & {M24};    // next limb of the multiplier")`.
5. In `wgsl()`: pass `mode` to `load`, compute `let mult = if mode == Mode::Mul { "b" } else { "x" };` and pass it to `iteration`. `conditional_double` stays `Mode::Fermat` only. Update the module doc comment (the "Two modes" list becomes three) and add to the `wgsl` bound argument: "In multiply mode the multiplier limbs `b` are normalised (below 2^13) like `x`, so every bound below holds unchanged."

- [ ] **Step 4: Generate the 32-bit multiply** (`limb32.rs`)

Same substitution: `step(g, lpl, w)` -> `step(g, lpl, w, mult)`; the line `let bb = u64(bcast(x{w}, owner, base));` becomes `bcast({mult}{w}, owner, base)`; in `wgsl()` for `Mode::Mul` emit binding 4 (`b_limbs`, read) after binding 3, load `let b{c} = b_limbs[off + {c}u];` for each limb after the `n{c}` loads, and skip `conditional_double` (already `Mode::Fermat`-only). `ninv` etc. unchanged.

- [ ] **Step 5: `try_mul_wgsl_for`** (`kernel.rs`, next to `try_fermat_wgsl_for`)

```rust
    /// The Montgomery multiply source (unrolled variant only): bindings 0..3 as in squaring mode
    /// plus `@binding(4) b_limbs` (read-only). Per instance `x <- x * b * R^-1 mod n`, `params.y`
    /// times, with the result in `[0, 2n)` for `x, b < 4n` (needs R > 16n, see `for_bits`).
    pub fn try_mul_wgsl_for(&self, target: Target) -> Result<String, String> {
        self.check_target(target)?;
        if self.variant != Variant::Unrolled {
            return Err("the multiply kernel is generated for the unrolled variant only".to_string());
        }
        if self.scheme == LimbScheme::Limb32 {
            return crate::limb32::wgsl(self, crate::unrolled::Mode::Mul);
        }
        Ok(crate::unrolled::wgsl(self, target, crate::unrolled::Mode::Mul))
    }
```

- [ ] **Step 6: `run_mul`** (`gpu.rs`, after `run_async`; model it on `run_on_device`, which is in the same file)

```rust
/// Multiplies each `(n, a, b)` `times` times on the device (`a <- a * b * R^-1 mod n`) and returns the
/// result limbs (`values.len() * cfg.limbs` words, in `cfg`'s limb scheme). Every `a` and `b` must be
/// below `4n`. A differential-test harness for the multiply kernel the Gerbicz check uses.
#[cfg(not(target_arch = "wasm32"))]
pub fn run_mul(
    gpu: &Gpu,
    cfg: KernelConfig,
    values: &[(BigUint, BigUint, BigUint)],
    times: u32,
) -> Result<Vec<u32>, String> {
    pollster::block_on(run_mul_async(gpu, cfg, values, times))
}

/// [`run_mul`] as a future.
pub async fn run_mul_async(/* same arguments */) -> Result<Vec<u32>, String>
```
Body: validate like `run_async` (`cfg.validate()`, `cfg.check_caps`, non-empty, subgroups, `gpu.check_health()`, workgroup count `<= 65535`; every `a`, `b` `< 4n`); pack `n`, `ninv = scheme.neg_inv(n)`, `a`, `b` with `cfg.scheme.pack(.., cfg.limbs)`; create buffers `params` (uniform 16 bytes, `[count, times, 0, 0]`), `n`, `ninv`, `x` (storage, `COPY_SRC`), `b` (storage), `staging` (`MAP_READ | COPY_DST`) exactly as `run_on_device` does with `guarded(...)`; compile `cfg.try_mul_wgsl_for(Target::host())?`; bind groups entries 0..=4 in the order `params, n, ninv, x, b`; one dispatch of `count.div_ceil(cfg.instances_per_workgroup())` groups; copy `x` to `staging`; `gpu.wait_async().await?`; `gpu.read_back_async(&staging, bytes).await`. Wrap with `.map_err(|e| gpu.explain(e))` like `run_async`.

- [ ] **Step 7: Run, expect pass**

Run: `cargo test -p bigmont-core --test mul --test source_matrix --test browser_source -- --test-threads=1` and `cargo check -p bigmont-core --target wasm32-unknown-unknown --locked`
Expected: PASS (13-bit and 32-bit at TPI 4/8/16/32; 32-bit skipped loudly only if the adapter lacks int64 and `BIGMONT_ALLOW_SKIP=1`). If a WGSL compile error mentions `b_limbs` unused or `y` uniformity, fix the generator, not the test.

- [ ] **Step 8: fmt, clippy, commit**

```bash
cargo fmt --all && cargo clippy --workspace --all-targets --all-features -- -D warnings
git add crates/bigmont-core
git commit -m "feat(core): general Montgomery multiply kernel for both limb schemes"
```

---

### Task 3: `Runner` - checked Fermat runs (`enable_gec`, `fermat_checked`, fault hook)

**Files:**
- Modify: `crates/bigmont-core/src/runner.rs`
- Test: `crates/bigmont-core/tests/gec.rs` (create)

**Interfaces:**
- Consumes (Task 1/2): `bigmont_oracle::{GEC_BLOCK, gec_blocks, gec_window_sum_words}`, `KernelConfig::try_mul_wgsl_for`, the multiply kernel's bindings (`0 params`, `1 n`, `2 ninv`, `3 x` rw, `4 b` read).
- Produces:
  - `pub enum Verdict { Composite, Prp, Alarm }` with `pub fn byte(self) -> u8` (0, 1, 2), `Clone, Copy, Debug, PartialEq, Eq`.
  - `Runner::enable_gec(&mut self, gpu: &Gpu) -> Result<(), String>` (native) and `pub async fn enable_gec_async(&mut self, gpu: &Gpu) -> Result<(), String>`; idempotent; allocates three extra numbers per candidate (A, B, y_prev) plus small buffers.
  - `Runner::fermat_checked(&mut self, gpu: &Gpu, ns: &[BigUint]) -> Result<Vec<Verdict>, String>` (native) and `fermat_checked_async`. `Err` if `enable_gec` was not called.
  - `#[doc(hidden)] pub struct Fault { pub index: usize, pub block: u32 }` and `#[doc(hidden)] pub fault: Option<Fault>` on `Runner`: in the next `fermat_checked` call, candidate `index` (into that call's `ns`) has bit 17 of its ladder state flipped after 20 iterations of block `block` (0-based; the reference flips after 4 of its 8 chunks). A block number `>= gec_blocks(bits, 40)` flips nothing. Cleared at the end of the call.
  - `Runner::fermat` is unchanged in behaviour.

**Design notes the implementer must follow**
- Refactor `fermat_chunk` into reusable private steps without changing the non-GEC path: `validate_chunk(ns, base)`, `upload_chunk(gpu, ns) -> Vec<Prep>` (host prep + the four `write_buffer`s), `ladder_range(gpu, count, from, to)` (the existing adaptive segment loop, with `total` replaced by `to` and `done` starting at `from`), and `read_x(gpu, count) -> Vec<u32>` (copy `x_buf` to `staging`, read back, check `normalised`). `fermat_chunk` becomes `validate; upload; ladder_range(0, bits-1); read_x; verdicts`. Run the existing tests after this refactor alone (commit it separately).
- Per-candidate extra state lives in `struct Gec` (a private field `gec: Option<Gec>`): `mul_pipeline`, `mul_params` (uniform, 16 bytes, `[count, 1, 0, 0]`), buffers `a_buf`, `b_buf`, `y_buf` (storage, `COPY_DST`, `a_buf`/`y_buf` also `COPY_SRC` not needed; `x_buf -> y_buf` is a copy so `y_buf` needs `COPY_DST`), `zero_e` (`cap * 2` words, never written, so all zero), `sumv` (`cap * 2` words), `staging2` (`MAP_READ | COPY_DST`, `2 * limb_bytes`; `a_buf` and `b_buf` are copied into it, so both need `COPY_SRC`), and bind groups `bg_mul_a` (binding 3 = `a_buf`, 4 = `y_buf`), `bg_mul_b` (3 = `b_buf`, 4 = `x_buf`), `bg_lad_a` (the ladder layout `self.pipeline.get_bind_group_layout(0)` with 3 = `a_buf`, 4 = `zero_e`), `bg_lad_y` (3 = `y_buf`, 4 = `sumv`); bindings 0, 1, 2, 5 of the ladder groups are the runner's existing `params`, `n_buf`, `ninv_buf`, `fp`; bindings 0, 1, 2 of the multiply groups are `mul_params`, `n_buf`, `ninv_buf`. Build them with `guarded(...)` like `build` does; errors go through `gpu.explain` at the call site.
- The check, per chunk of `count` candidates (`nb = gec_blocks(bits, 40)`; when `nb == 0` skip the blocks and the check, the run is unprotected, as in the reference):
  1. Upload: `n, x0, ninv, e` as today; `a_buf` and `b_buf` = packed `p.r` (`R mod n` = `1~`) for every candidate; `y_buf` = the packed `x0`s; `sumv` = per candidate `[lo, hi]` words of `gec_window_sum_words(&p.e, bits, 40)`.
  2. For `blk in 0..nb`: `block_step(first = blk*40, iters = 40, boundary = true)`. If the fault applies to this block (`fault.block == blk` and the candidate is in this chunk), do instead `block_step(first, 20, boundary=false)`, then corrupt, then `block_step(first + 20, 20, boundary=true)`.
  3. `block_step(count, first, iters, boundary)`: `queue.write_buffer(params, [count, iters, 0, 0])`, `queue.write_buffer(fp, [first, bits, words, 0])`; one encoder: a compute pass with the ladder pipeline + `bind_group`; if `boundary`: a pass with `mul_pipeline` + `bg_mul_a` (`A <- A * y_prev`), a pass with `mul_pipeline` + `bg_mul_b` (`B <- B * x`), then `encoder.copy_buffer_to_buffer(x_buf, 0, y_buf, 0, count*limbs*4)`. Submit, `gpu.wait_async().await?`, add elapsed to `stats.gpu_secs`. (The order inside the encoder is the contract: the multiplies use the OLD `y_prev`, the copy comes last.)
  4. Tail: `ladder_range(gpu, count, nb*40, bits-1)`.
  5. Check, only when `nb > 0`:
     - a. squarings of A: `params = [count, 40, 0, 0]`, `fp = [0, 42, 2, 0]`, ladder pipeline with `bg_lad_a`. (`gi = fp.y - 2 - (fp.x + s)` runs 40..1 and reads the all-zero `zero_e`, so no doubling: 40 pure squarings.) Submit and wait.
     - b. `2^(sum V)`: `queue.write_buffer(y_buf, packed 1~ for every candidate)` (the old `y_prev` is dead), then `L = 40 + (32 - nb.leading_zeros())` (an upper bound of the bit length of `sum V`, at most 64), `params = [count, L, 0, 0]`, `fp = [0, L + 1, 2, 0]`, ladder with `bg_lad_y` (starting from `1~` and reading the exponent bits `L-1..0` of `sumv` gives `(2^sum V)~`). Submit and wait.
     - c. `A <- A * y_prev` with `mul_pipeline` + `bg_mul_a` (`mul_params`). Submit and wait.
     - d. copy `a_buf` to `staging2[0..bytes)` and `b_buf` to `staging2[bytes..2*bytes)`, read back, `scheme.normalised` check (else `Err("kernel returned an unnormalised limb")`). Candidate `i` raises an alarm unless `a == b` or `|a - b| == n` (both are in `[0, 2n)`).
  6. `read_x`, then verdict `Alarm` if alarmed, else `Prp` if `unpack(x) % n == p.r`, else `Composite`. `stats.tests += count`.
- Fault corruption (`corrupt(gpu, idx)`): the limb holding bit 17 is `17 / limb_bits` at bit `17 % limb_bits`; copy that one word of `x_buf` (offset `(idx * limbs + limb) * 4`) into `staging`, `read_back_async(&staging, 4)`, XOR the bit, `queue.write_buffer(x_buf, offset, ..)`.
- `fermat_checked_async` mirrors `fermat_async` (health check, chunks of `cap`, `gpu.explain` on errors), computing the per-chunk fault `(index - base, block)` when `base <= index < base + chunk.len()`, and clearing `self.fault` at the end (also on error).
- `#[cfg(not(target_arch = "wasm32"))]` wrappers use `pollster::block_on`, like the existing ones. Use `web_time::Instant` as the file already does.

- [ ] **Step 1: Refactor `fermat_chunk` into the steps above (no behaviour change)**

Run: `cargo test -p bigmont-core --test fermat -- --test-threads=1`
Expected: PASS before and after. Commit: `git commit -m "refactor(core): split the runner's chunk into upload, ladder and readback steps"`.

- [ ] **Step 2: Write the failing tests** (`crates/bigmont-core/tests/gec.rs`)

```rust
//! The Gerbicz check in `Runner::fermat_checked`: clean runs agree with the oracle and never alarm,
//! injected faults alarm exactly the injected candidate.
mod common;
use bigmont_core::gpu::Gpu;
use bigmont_core::kernel::{KernelConfig, Variant};
use bigmont_core::runner::{Fault, Runner, Verdict};
use bigmont_core::scheme::LimbScheme;
use bigmont_oracle::*;
use num_bigint::BigUint;

fn open() -> Option<Gpu> {
    match Gpu::new() {
        Ok(g) => Some(g),
        Err(e) => {
            common::skip(format!("gec: no adapter: {e}"));
            None
        }
    }
}

fn runner(gpu: &Gpu, scheme: LimbScheme, tpi: u32, bits: u32, cap: u32) -> Option<Runner> {
    let cfg = KernelConfig::for_bits_scheme(scheme, bits, tpi, 64, Variant::Unrolled);
    if !gpu.caps().satisfies(cfg.requires()) {
        common::skip(format!("{cfg:?}: the adapter lacks {}", gpu.caps().missing(cfg.requires()).join(" and ")));
        return None;
    }
    if let Err(e) = gpu.check_subgroups(tpi) {
        common::skip(format!("{cfg:?}: {e}"));
        return None;
    }
    let mut r = Runner::new(gpu, cfg, bits, cap).expect("runner");
    r.enable_gec(gpu).expect("enable_gec");
    Some(r)
}

/// Random odd numbers plus a few probable primes, so both verdicts occur.
fn mixed(bits: u32, random: u32, prps: u32, seed: u64) -> Vec<BigUint> {
    let mut rng = XorShift64::new(seed);
    let mut v: Vec<BigUint> = (0..random).map(|_| random_odd_modulus(&mut rng, bits)).collect();
    for _ in 0..prps {
        v.push(random_prp(&mut rng, bits));
    }
    v
}

fn truth(ns: &[BigUint]) -> Vec<Verdict> {
    ns.iter()
        .map(|n| if fermat_base2(n) { Verdict::Prp } else { Verdict::Composite })
        .collect()
}

#[test]
fn a_clean_checked_run_equals_the_oracle_and_never_alarms() {
    let Some(gpu) = open() else { return };
    for scheme in LimbScheme::ALL.iter().copied() {
        // 3 bits: no full block (unprotected); 41/42: one block with a 0/1 iteration tail; then
        // several blocks; 2029 is the production size.
        for bits in [3u32, 41, 42, 81, 130, 300, 2029] {
            let Some(mut r) = runner(&gpu, scheme, 8, bits, 64) else { break };
            let ns = mixed(bits, 12, 2, 40 + u64::from(bits));
            let got = r.fermat_checked(&gpu, &ns).expect("fermat_checked");
            assert_eq!(got, truth(&ns), "{scheme:?} {bits} bits");
        }
    }
}

#[test]
fn checked_and_unchecked_runs_agree_and_chunking_keeps_the_order() {
    let Some(gpu) = open() else { return };
    let Some(mut r) = runner(&gpu, LimbScheme::Limb13, 8, 300, 5) else { return };
    let ns = mixed(300, 17, 3, 9); // 20 candidates over chunks of 5
    let checked = r.fermat_checked(&gpu, &ns).expect("checked");
    let plain = r.fermat(&gpu, &ns).expect("plain");
    let from_plain: Vec<Verdict> = plain
        .iter()
        .map(|&p| if p { Verdict::Prp } else { Verdict::Composite })
        .collect();
    assert_eq!(checked, from_plain);
}

#[test]
fn an_injected_fault_alarms_only_its_candidate() {
    let Some(gpu) = open() else { return };
    for scheme in LimbScheme::ALL.iter().copied() {
        let bits = 300u32; // 299 iterations: blocks 0..=6 and a 19-iteration tail
        let Some(mut r) = runner(&gpu, scheme, 8, bits, 5) else { break };
        let ns = mixed(bits, 9, 3, 77); // 12 candidates, chunks of 5: index 7 is in the second chunk
        let want = truth(&ns);
        for (index, block) in [(0usize, 0u32), (7, 3), (11, 6), (3, 1)] {
            r.fault = Some(Fault { index, block });
            let got = r.fermat_checked(&gpu, &ns).expect("fermat_checked");
            for (i, v) in got.iter().enumerate() {
                if i == index {
                    assert_eq!(*v, Verdict::Alarm, "{scheme:?} index {index} block {block}");
                } else {
                    assert_eq!(*v, want[i], "{scheme:?} candidate {i} unaffected by a fault in {index}");
                }
            }
        }
        // The fault was consumed: a following clean run is clean.
        assert_eq!(r.fermat_checked(&gpu, &ns).expect("clean"), want);
        // A block that does not exist flips nothing.
        r.fault = Some(Fault { index: 2, block: 7 });
        assert_eq!(r.fermat_checked(&gpu, &ns).expect("no such block"), want);
    }
}

#[test]
fn a_fault_in_a_probable_prime_alarms_instead_of_passing_it() {
    let Some(gpu) = open() else { return };
    let Some(mut r) = runner(&gpu, LimbScheme::Limb13, 8, 300, 16) else { return };
    let ns = mixed(300, 2, 2, 5); // the last two are probable primes
    assert_eq!(truth(&ns)[3], Verdict::Prp);
    r.fault = Some(Fault { index: 3, block: 2 });
    assert_eq!(r.fermat_checked(&gpu, &ns).expect("run")[3], Verdict::Alarm);
}

#[test]
fn checking_needs_enable_gec() {
    let Some(gpu) = open() else { return };
    let cfg = KernelConfig::for_bits_scheme(LimbScheme::Limb13, 300, 8, 64, Variant::Unrolled);
    let mut r = Runner::new(&gpu, cfg, 300, 8).expect("runner");
    let err = r.fermat_checked(&gpu, &mixed(300, 2, 0, 1)).unwrap_err();
    assert!(err.contains("enable_gec"), "{err}");
}
```

- [ ] **Step 3: Run, expect compile failure**, then implement per the design notes above.

Run: `cargo test -p bigmont-core --test gec --no-run`

- [ ] **Step 4: Implement `Verdict`, `Fault`, `Gec`, `enable_gec(_async)`, `fermat_checked(_async)`, `block_step`, `corrupt`** as described. Keep every doc comment accurate (public items need docs; `Fault` is `#[doc(hidden)]` like the engine's other test hooks). Do not change `Runner::new`/`new_async` signatures.

- [ ] **Step 5: Run, expect pass**

Run: `cargo test -p bigmont-core -- --test-threads=1` and `cargo check -p bigmont-core -p bigmont-wasm --target wasm32-unknown-unknown --locked`
Expected: PASS everywhere. If a clean run alarms, suspect (in this order) the operand order of the boundary multiplies, the missing `1~` upload into `y_buf` before step 5b, and `fp.y`/`fp.z` of the check dispatches; use `gerbicz_model` from Task 1 as the reference for intermediate values.

- [ ] **Step 6: fmt, clippy, commit**

```bash
cargo fmt --all && cargo clippy --workspace --all-targets --all-features -- -D warnings
git add crates/bigmont-core
git commit -m "feat(core): Gerbicz-checked Fermat runs with a fault-injection hook"
```

---

### Task 4: Engine, CLI (`--gec`, `--inject IDX BDRY`), summary, docs

**Files:**
- Modify: `crates/bigmont-cli/src/engine.rs`, `crates/bigmont-cli/src/main.rs`, `README.md`, `CHANGELOG.md`
- Test: `crates/bigmont-cli/tests/engine.rs`, `crates/bigmont-cli/tests/cli.rs`

**Interfaces:**
- Consumes: `bigmont_core::runner::{Runner, Verdict, Fault}` (`enable_gec`, `fermat_checked`).
- Produces:
  - `Options::gec: bool` (default `false`): GPU buckets run checked; an alarm gives verdict byte 2.
  - `Engine::inject_fault(&mut self, index: usize, block: u32)` (`#[doc(hidden)]`, test and drill hook): applies to the next `verdicts` call; `index` is into that call's `ns`; cleared at the end of the call.
  - `EngineReport::gec_alarms: u64`.
  - `Engine::verdicts` may now return `2` (only with `Options::gec`). Update the module doc ("The engine never returns 2") and the `verdicts` doc.
  - CLI: `--gec` (boolean flag), `--inject IDX BDRY` (two values; replaces the `UNSUPPORTED` entry, remove that const and its check if it becomes empty).

**Behaviour**
- `try_gpu`: after building a `Runner`, when `opts.gec`, call `runner.enable_gec(gpu)` inside the same `and_then` chain as `Runner::new` so a failure is handled like any other build error (device failure -> `DeviceFailure`; otherwise a message and CPU fallback). Call `runner.fermat_checked` when `opts.gec`, else `runner.fermat` mapped to `Verdict`. Set `runner.fault` from the pending engine fault (mapped from the call's input index to the bucket position) just before the call.
- Integrity pass: `Verdict::Prp` is re-checked on the CPU as today (disagreement -> CPU verdict, counters as today); sampled composites as today; `Verdict::Alarm` is counted in `report.gec_alarms`, logged once per alarm (`bigmont: GEC alarm on a {bits}-bit candidate`), never re-checked, and stays `Alarm` (the caller, like Beacon's feeders, retests or arbitrates). Buckets answered by the CPU (small buckets, fallbacks) are unchecked and cannot alarm: document this in the `Options::gec` doc.
- CLI: `--inject` requires `--gec` ("--inject needs --gec"), refuses `--ee` ("--inject applies to plain runs only"), refuses `--backend cpu`, requires `IDX < candidate count` (exit 2 otherwise), and sets `min_gpu_batch = 1` (the reference drill runs 3 candidates; the default of 64 would answer them on the CPU and inject nothing). `run_engine` takes `inject: Option<(usize, u32)>` and calls `engine.inject_fault(idx - chunk_start, block)` for the chunk that contains `idx`. `canary` passes `None`. The plain-mode summary line gains `GEC alarms {n}`; EE mode already prints `gec_alarm={out.alarms}` and keeps working (verdict 2 retires the k). Update `USAGE` (flags, exit semantics unchanged).

- [ ] **Step 1: Write the failing tests**

`crates/bigmont-cli/tests/engine.rs` (append; `common::usable_gpu()` is the file's strict-GPU guard):

```rust
fn gec_opts() -> Options {
    Options {
        backend: Backend::Gpu,
        gec: true,
        min_gpu_batch: 1,
        capacity: 32,
        verify_rate: 0,
        ..Options::default()
    }
}

#[test]
fn gec_verdicts_equal_plain_verdicts_on_clean_input_and_never_alarm() {
    if !common::usable_gpu() {
        return;
    }
    let mut rng = XorShift64::new(21);
    let mut ns: Vec<BigUint> = (0..40)
        .map(|_| random_odd_modulus(&mut rng, 1019))
        .collect();
    ns.extend((0..2).map(|_| random_prp(&mut rng, 1019)));
    let want: Vec<u8> = ns.iter().map(|n| u8::from(fermat_base2(n))).collect();
    assert!(want.contains(&1), "the batch must contain a probable prime");
    let mut checked = Engine::open(gec_opts());
    assert_eq!(checked.verdicts(&ns).unwrap(), want);
    assert_eq!(checked.report().gec_alarms, 0);
    let mut plain = Engine::open(Options {
        gec: false,
        ..gec_opts()
    });
    assert_eq!(plain.verdicts(&ns).unwrap(), want);
}

#[test]
fn an_injected_fault_gives_verdict_2_for_that_candidate_only() {
    if !common::usable_gpu() {
        return;
    }
    // The Beacon harness drill: 3 candidates of 2029 bits, fault in candidate 1 at block 3.
    let mut rng = XorShift64::new(22);
    let ns: Vec<BigUint> = (0..3)
        .map(|_| random_odd_modulus(&mut rng, 2029))
        .collect();
    let want: Vec<u8> = ns.iter().map(|n| u8::from(fermat_base2(n))).collect();
    let mut e = Engine::open(gec_opts());
    e.inject_fault(1, 3);
    assert_eq!(e.verdicts(&ns).unwrap(), vec![want[0], 2, want[2]]);
    assert_eq!(e.report().gec_alarms, 1);
    // The hook is one-shot: the next call is clean.
    assert_eq!(e.verdicts(&ns).unwrap(), want);
    assert_eq!(e.report().gec_alarms, 1);
}
```

`crates/bigmont-cli/tests/cli.rs`: first replace the old test `inject_is_still_refused_and_the_ee_flags_are_rejected_by_other_subcommands` by `the_ee_flags_are_rejected_by_other_subcommands` (keep its `for sub in ["canary", "tune"]` loop, delete the `--inject ... not supported` assertions at its top), then append:

```rust
/// Three random 2029-bit candidates (the Beacon harness drill input) in a 2048-bit file.
fn drill_candidates(dir: &Path) -> (PathBuf, Vec<BigUint>) {
    let mut rng = XorShift64::new(31);
    let ns: Vec<BigUint> = (0..3)
        .map(|_| random_odd_modulus(&mut rng, 2029))
        .collect();
    let p = dir.join("drill.bin");
    formats::write_candidates(&p, 2048, &ns).unwrap();
    (p, ns)
}

#[test]
fn a_gec_run_writes_the_same_verdicts_as_a_plain_run() {
    if !gpu_available() {
        return;
    }
    let dir = tmp_dir("gec-clean");
    let cand = candidates(&dir);
    let cache = dir.join("tune.cache");
    let mut files = Vec::new();
    for (name, extra) in [("plain.bin", None), ("gec.bin", Some("--gec"))] {
        let verdict = dir.join(name);
        let mut args = vec![
            cand.to_str().unwrap(),
            verdict.to_str().unwrap(),
            "--backend",
            "gpu",
            "--cache",
            cache.to_str().unwrap(),
        ];
        args.extend(extra);
        let out = run(bigmont(&args));
        assert_eq!(code(&out), Some(0), "{}", String::from_utf8_lossy(&out.stderr));
        if extra.is_some() {
            assert!(
                String::from_utf8_lossy(&out.stderr).contains("GEC alarms 0"),
                "{}",
                String::from_utf8_lossy(&out.stderr)
            );
        }
        files.push(formats::read_verdicts(&verdict).unwrap().1);
    }
    assert_eq!(files[0], files[1]);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn the_inject_drill_gives_verdict_2_for_the_injected_candidate() {
    if !gpu_available() {
        return;
    }
    let dir = tmp_dir("gec-drill");
    let (cand, ns) = drill_candidates(&dir);
    let verdict = dir.join("v.bin");
    let cache = dir.join("tune.cache");
    let out = run(bigmont(&[
        cand.to_str().unwrap(),
        verdict.to_str().unwrap(),
        "--backend",
        "gpu",
        "--cache",
        cache.to_str().unwrap(),
        "--gec",
        "--inject",
        "1",
        "3",
    ]));
    assert_eq!(code(&out), Some(0), "{}", String::from_utf8_lossy(&out.stderr));
    let want: Vec<u8> = ns.iter().map(|n| u8::from(bigmont_cpu::fermat_base2(n))).collect();
    assert_eq!(
        formats::read_verdicts(&verdict).unwrap().1,
        vec![want[0], 2, want[2]]
    );
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn inject_usage_errors_exit_2_before_any_work() {
    let dir = tmp_dir("gec-usage");
    let (cand, _) = drill_candidates(&dir);
    let (c, v) = (cand.to_str().unwrap(), dir.join("v.bin"));
    let v = v.to_str().unwrap();
    let cases: [(&[&str], &str); 5] = [
        (&[c, v, "--inject", "1", "3"], "--inject needs --gec"),
        (&[c, v, "--gec", "--inject", "5", "3"], "5"),
        (&[c, v, "--gec", "--inject", "0", "0", "--ee", "s.bin"], "plain runs only"),
        (&[c, v, "--gec", "--inject", "0", "0", "--backend", "cpu"], "GPU"),
        (&[c, v, "--gec", "--inject=1"], "--inject"),
    ];
    for (args, needle) in cases {
        let out = run(bigmont(args));
        assert_eq!(code(&out), Some(2), "{args:?}");
        let err = String::from_utf8_lossy(&out.stderr);
        assert!(err.contains(needle), "{args:?}: {err}");
        assert!(!Path::new(v).exists(), "{args:?} must not write a verdict file");
    }
    let _ = std::fs::remove_dir_all(dir);
}
```
Argument errors that need no file (`--ee`, `--backend cpu`, missing `--gec`, `--inject=1`) are raised in `parse_args`; the index check (`5` of 3) is raised in `cmd_verdicts` right after the candidate file is read and before the engine is opened.

- [ ] **Step 2: Run, expect failure**

Run: `cargo test -p bigmont-cli --test engine --test cli -- --test-threads=1`

- [ ] **Step 3: Implement** engine and CLI per "Behaviour". Add unit tests for the argument parsing where the file has them (`--inject` consumes exactly two values; `--inject=1` is a usage error because it needs two values).

- [ ] **Step 4: Docs** - `README.md`: a short "Error checking (Gerbicz)" section: what `--gec` does, that it is off by default (the CUDA driver's production default too), the verdict-2 contract, `--inject IDX BDRY` as the fault drill, the unprotected tail of `(bits - 1) % 40` iterations, that CPU-answered buckets are unchecked, and the measured overhead from Task 5 once known (leave the number out until then; do not invent it). `CHANGELOG.md` `[Unreleased]` / Added: one entry covering `--gec`, `--inject`, verdict 2, `gec_alarm`, and "closes #17 and #12".

- [ ] **Step 5: Run everything**

Run: `cargo fmt --all --check && cargo clippy --workspace --all-targets --all-features -- -D warnings && cargo test --workspace --no-fail-fast -- --test-threads=1 && cargo doc --workspace --no-deps --all-features`
Expected: all green (RUSTDOCFLAGS `-D warnings` for the doc build as CI does).

- [ ] **Step 6: Commit**

```bash
git add -A crates README.md CHANGELOG.md
git commit -m "feat(cli): opt-in Gerbicz check (--gec) and the --inject fault drill"
```

---

### Task 5: Gate and measurement

**Files:**
- Create: `docs/superpowers/results/2026-11-25-gerbicz-check.md`
- Modify: `README.md` (the overhead sentence), `CHANGELOG.md` only if the numbers change a claim.

The reference's gate (`nogec_gate.sh`): GEC and non-GEC verdicts byte-identical on a known-answer set and on random candidates, then throughput with and without. Reproduce it with `bigmont`.

- [ ] **Step 1: Build release and generate candidates**

```bash
cargo build --release -p bigmont-cli
cargo run --release -p bigmont-cli --example make_candidates -- 20000 1 /tmp/gec_rand.bin
```
(`make_candidates` appends 3 probable primes, so both verdicts appear.)

- [ ] **Step 2: Identity** - run `bigmont /tmp/gec_rand.bin /tmp/v_plain.bin` and `bigmont --gec /tmp/gec_rand.bin /tmp/v_gec.bin` on the same GPU and `--backend gpu`; `cmp` the files (must be identical) and confirm the summary shows `GEC alarms 0`. Run `bigmont canary` (see `bigmont --help`) with and without `--gec` if the canary files are available; otherwise say they were not. Run the drill: `bigmont --gec --inject 1 3 <3 candidates> out.bin` and show the bytes (candidate 1 is 2).

- [ ] **Step 3: Throughput** - on the same idle GPU run each of plain and `--gec` three times alternately on `/tmp/gec_rand.bin`, reading "tests/s excluding tuning" from the summary line (discard the first run of each, which tunes unless the tuning cache is warm: use the same cache for both). Compute the mean per mode and the throughput cost `1 - gec/plain`.

- [ ] **Step 4: Write the results file** with: hardware model, driver/backend, OS, limb scheme and TPI the tuner picked, commands, the identity result, the drill bytes, the six throughput numbers, the cost, and the CUDA driver's figure for comparison (514,850 -> 581,795 tests/s on an RTX 5090 = about 11.5 % cost; cite `KIT_v8.55_NOTE.md` of the Beacon repo by name only, no paths). State plainly which vendors were not tested.

- [ ] **Step 5: Decide** - if the cost is at most about 15 %, finish. If it is higher, record the profile (`stats.gpu_secs` vs wall time per chunk from `Runner::stats`) in the results file and note the follow-up (several blocks per submission via a pool of `fp` bind groups, or fusing the boundary multiplies into the ladder kernel) without implementing it here.

- [ ] **Step 6: Update the README overhead sentence with the measured number, commit**

```bash
git add docs README.md
git commit -m "docs: Gerbicz check gate and measured overhead"
```

---

## PR

Title `Opt-in Gerbicz error check and the --inject fault drill (closes #17, closes #12)`. Fill the "Tested on" section of the PR template from the results file (never claim a vendor that was not run; the six rows NVIDIA, AMD, Intel, Apple, Software, Browser each get a "tested" or "not tested" entry; the 13-bit multiply compiles for wasm32 but was not run in a browser unless it was). End the description with the Claude Code attribution line.

## Self-review notes

- Spec coverage: general multiply mode in both schemes (Task 2); per-candidate state, block bookkeeping, span check (Task 3); choosing block size and measuring overhead (Tasks 3, 5; single span argued in "The math"); tests with injected faults, clean runs, verdicts unchanged (Tasks 1, 3, 4, 5); opt-in default and the CPU re-check of PRPs retained (Task 4); #12 `--inject` folded in because the reference harness's drill depends on it.
- Known limitation kept from the reference: the last `(bits - 1) % 40` iterations are unprotected (tested by the model in Task 1 so it is documented behaviour, not an accident).
- One submission and wait per 40-iteration block is the first implementation; Task 5 decides whether that costs too much.
