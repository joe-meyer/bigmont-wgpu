# bigmont-wgpu Phase 0 (speed spike) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure how fast a 2,048-bit Montgomery squaring runs through wgpu on the RX 6700 XT, in a one-thread-per-number kernel and in subgroup-cooperative kernels, with every result checked against a CPU oracle first.

**Architecture:** A Cargo workspace with two crates. `bigmont-oracle` is a pure-CPU reference (big-integer Montgomery chain plus 13-bit limb packing). `bigmont-core` holds one parametrised WGSL kernel (`TPI` lanes per number; `TPI = 1` is the one-thread-per-number baseline), the wgpu runner, validation tests and a benchmark example. Numbers are 160 limbs of 13 bits (R = 2^2080) with deferred carries, the same scheme gapsieve uses at small sizes.

**Tech Stack:** Rust 1.100 (edition 2024), wgpu 30.0.2, pollster 1.0.3, bytemuck 1.25.2, num-bigint 0.4. WGSL with `enable subgroups;` for the cooperative kernels.

**Spec:** `docs/superpowers/specs/2026-11-24-bigmont-wgpu-design.md` (sections 3, 4 and 6 are the ones this plan implements). Reference code for the wgpu 30 API and the 13-bit-limb scheme: `N:\Programming\gapsieve\examples\mul_bench.rs`, `N:\Programming\gapsieve\src\gpu.rs` (readback), `N:\Programming\gapsieve\src\limb13.rs` (limb notes).

## Global Constraints

- Rust edition 2024; wgpu exactly `30.0.2`; pollster `1.0.3`; bytemuck `1.25.2` with `derive`. These are already in the local cargo cache from gapsieve.
- Kernels use only WGSL features that browsers implement. The only extension allowed is `enable subgroups;` (wgpu `Features::SUBGROUP`). No `SHADER_INT64`, no push constants, no u64 anywhere in WGSL.
- Number format: `LIMBS = 160` limbs of `LIMB_BITS = 13`, little-endian, one limb per `u32`. Montgomery radix `R = 2^(13*160) = 2^2080`. Test moduli are odd, exactly 2029 bits (Beacon's size class), so `R > 8n`.
- Correctness before speed: no throughput number is reported for a kernel configuration until it passes the oracle test.
- Apache-2.0 license on all new crates. Files carry no copied Beacon or CGBN code.
- Windows TDR: keep every single GPU dispatch under about 0.5 s. The benchmark sizes dispatches adaptively for this reason.
- Do not download anything large (no CUDA toolkit), post anywhere, or contact anyone. Phase 0 runs only on this machine's GPU(s). The RTX 4070 laptop run is a separate step the user decides on.
- Commit messages end with the line `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.
- Work in `N:\Programming\bigmont-wgpu` (a git repo, branch `main`). Use absolute paths in tools; the shell's working directory may be elsewhere.

## File Structure

```
bigmont-wgpu/
  Cargo.toml                         workspace
  .gitignore                         target/
  crates/bigmont-oracle/
    Cargo.toml
    src/lib.rs                       limbs, RNG, Montgomery chain (CPU)
    tests/oracle.rs                  oracle self-tests
  crates/bigmont-core/
    Cargo.toml
    src/lib.rs                       re-exports
    src/mont.wgsl                    the kernel template
    src/kernel.rs                    KernelConfig, WGSL generation
    src/gpu.rs                       Gpu (adapter/device), Batch, run()
    tests/chain.rs                   GPU vs oracle validation
    examples/spike.rs                benchmark + validation sweep
  docs/superpowers/results/          written in Task 5
```

---

### Task 1: Workspace and CPU oracle

**Files:**
- Create: `N:\Programming\bigmont-wgpu\Cargo.toml`
- Create: `N:\Programming\bigmont-wgpu\.gitignore`
- Create: `N:\Programming\bigmont-wgpu\crates\bigmont-oracle\Cargo.toml`
- Create: `N:\Programming\bigmont-wgpu\crates\bigmont-oracle\src\lib.rs`
- Test: `N:\Programming\bigmont-wgpu\crates\bigmont-oracle\tests\oracle.rs`

**Interfaces:**
- Consumes: nothing.
- Produces (all `pub` in `bigmont_oracle`):
  - `const LIMB_BITS: u32 = 13`, `const LIMB_MASK: u32 = 0x1fff`, `const LIMBS: usize = 160`, `const R_BITS: u32 = 2080`
  - `struct XorShift64` with `fn new(seed: u64) -> Self`, `fn next_u64(&mut self) -> u64`
  - `fn random_bits(rng: &mut XorShift64, bits: u32) -> BigUint` (uniform below `2^bits`)
  - `fn random_odd_modulus(rng: &mut XorShift64, bits: u32) -> BigUint` (exactly `bits` bits, odd)
  - `fn random_below(rng: &mut XorShift64, n: &BigUint) -> BigUint`
  - `fn pack_limbs(x: &BigUint) -> Vec<u32>` (exactly `LIMBS` entries, each below 2^13; panics if `x >= 2^2080`)
  - `fn unpack_limbs(limbs: &[u32]) -> BigUint` (accepts limbs above 2^13; sums with carries)
  - `fn neg_inv_limb(n: &BigUint) -> u32` (`-n^-1 mod 2^13`)
  - `fn inv_mod_pow2(n: &BigUint, bits: u32) -> BigUint`
  - `fn mont_square_chain(n: &BigUint, x0: &BigUint, squarings: u32) -> BigUint` (`x <- x^2 * R^-1 mod n`, repeated; result in `[0, n)`)

- [ ] **Step 1: Create the workspace files**

`N:\Programming\bigmont-wgpu\Cargo.toml`:

```toml
[workspace]
resolver = "3"
members = ["crates/bigmont-oracle", "crates/bigmont-core"]

[workspace.package]
edition = "2024"
license = "Apache-2.0"
version = "0.1.0"

[profile.test]
opt-level = 3
```

`N:\Programming\bigmont-wgpu\.gitignore`:

```
target/
.superpowers/
```

`N:\Programming\bigmont-wgpu\crates\bigmont-oracle\Cargo.toml`:

```toml
[package]
name = "bigmont-oracle"
description = "CPU reference for the 13-bit-limb Montgomery arithmetic used by the bigmont GPU kernels"
version.workspace = true
edition.workspace = true
license.workspace = true

[dependencies]
num-bigint = "0.4"
num-traits = "0.2"
```

Also create a placeholder `crates/bigmont-core` so the workspace resolves: `N:\Programming\bigmont-wgpu\crates\bigmont-core\Cargo.toml` with

```toml
[package]
name = "bigmont-core"
version.workspace = true
edition.workspace = true
license.workspace = true

[dependencies]
bigmont-oracle = { path = "../bigmont-oracle" }
```

and an empty `N:\Programming\bigmont-wgpu\crates\bigmont-core\src\lib.rs`.

- [ ] **Step 2: Write the failing tests**

`N:\Programming\bigmont-wgpu\crates\bigmont-oracle\tests\oracle.rs`:

```rust
use bigmont_oracle::*;
use num_bigint::BigUint;
use num_traits::One;

fn modpow_u128(mut b: u128, mut e: u128, m: u128) -> u128 {
    let mut r = 1 % m;
    b %= m;
    while e > 0 {
        if e & 1 == 1 {
            r = r * b % m;
        }
        b = b * b % m;
        e >>= 1;
    }
    r
}

/// Inverse of a modulo m (m small), by the extended Euclidean algorithm.
fn inv_u128(a: u128, m: u128) -> u128 {
    let (mut old_r, mut r) = (a as i128, m as i128);
    let (mut old_s, mut s) = (1i128, 0i128);
    while r != 0 {
        let q = old_r / r;
        (old_r, r) = (r, old_r - q * r);
        (old_s, s) = (s, old_s - q * s);
    }
    assert_eq!(old_r, 1, "not invertible");
    old_s.rem_euclid(m as i128) as u128
}

#[test]
fn pack_unpack_roundtrip() {
    let mut rng = XorShift64::new(1);
    for bits in [1, 13, 14, 31, 32, 33, 1000, 2029, 2080] {
        let x = random_bits(&mut rng, bits);
        let limbs = pack_limbs(&x);
        assert_eq!(limbs.len(), LIMBS);
        assert!(limbs.iter().all(|&l| l <= LIMB_MASK));
        assert_eq!(unpack_limbs(&limbs), x, "bits={bits}");
    }
}

#[test]
fn unpack_accepts_unnormalized_limbs() {
    // 2 * 2^13 + 5 written as limbs [5 + 2^13 * 2, 0] must equal limbs [5, 2].
    assert_eq!(unpack_limbs(&[5 + (2 << 13), 0]), unpack_limbs(&[5, 2]));
    assert_eq!(unpack_limbs(&[0xffff_ffff, 0xffff_ffff]), (BigUint::from(0xffff_ffffu32) * ((BigUint::one() << 13) + 1u32)));
}

#[test]
fn random_modulus_has_exact_size_and_is_odd() {
    let mut rng = XorShift64::new(2);
    for _ in 0..50 {
        let n = random_odd_modulus(&mut rng, 2029);
        assert_eq!(n.bits(), 2029);
        assert_eq!(n.to_u32_digits()[0] & 1, 1);
    }
}

#[test]
fn neg_inv_limb_property() {
    let mut rng = XorShift64::new(3);
    for _ in 0..50 {
        let n = random_odd_modulus(&mut rng, 2029);
        let n0 = n.to_u32_digits()[0] & LIMB_MASK;
        let ninv = neg_inv_limb(&n);
        assert!(ninv <= LIMB_MASK);
        assert_eq!((n0 * ninv + 1) & LIMB_MASK, 0, "n * ninv must be -1 mod 2^13");
    }
}

#[test]
fn inv_mod_pow2_property() {
    let mut rng = XorShift64::new(4);
    for bits in [1, 13, 100, 2080] {
        let n = random_odd_modulus(&mut rng, 2029);
        let inv = inv_mod_pow2(&n, bits);
        let mask = (BigUint::one() << bits) - 1u32;
        assert_eq!((&n * &inv) & &mask, BigUint::one() & &mask, "bits={bits}");
    }
}

#[test]
fn chain_matches_independent_small_case() {
    // n and x0 small enough for u128 arithmetic; R = 2^2080.
    let (n, x0) = (1_000_003u128, 123_456u128);
    let r_mod_n = modpow_u128(2, u128::from(R_BITS), n);
    let rinv = inv_u128(r_mod_n, n);
    let one_step = x0 * x0 % n * rinv % n;
    let two_steps = one_step * one_step % n * rinv % n;
    let (nb, xb) = (BigUint::from(n), BigUint::from(x0));
    assert_eq!(mont_square_chain(&nb, &xb, 1), BigUint::from(one_step));
    assert_eq!(mont_square_chain(&nb, &xb, 2), BigUint::from(two_steps));
    assert_eq!(mont_square_chain(&nb, &xb, 0), xb);
}

#[test]
fn chain_composes_at_full_size() {
    let mut rng = XorShift64::new(5);
    let n = random_odd_modulus(&mut rng, 2029);
    let x0 = random_below(&mut rng, &n);
    let three = mont_square_chain(&n, &x0, 3);
    let one_then_two = mont_square_chain(&n, &mont_square_chain(&n, &x0, 1), 2);
    assert_eq!(three, one_then_two);
    assert!(three < n);
}
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `cargo test -p bigmont-oracle --manifest-path N:\Programming\bigmont-wgpu\Cargo.toml`
Expected: compile FAIL (items not found in `bigmont_oracle`).

- [ ] **Step 4: Write the implementation**

`N:\Programming\bigmont-wgpu\crates\bigmont-oracle\src\lib.rs`:

```rust
//! CPU reference ("oracle") for the 13-bit-limb Montgomery arithmetic used by the GPU kernels.
//! Everything here is plain big-integer arithmetic, deliberately unrelated to the GPU code's
//! tricks, so that the two can be compared.

use num_bigint::BigUint;
use num_traits::One;

pub const LIMB_BITS: u32 = 13;
pub const LIMB_MASK: u32 = (1 << LIMB_BITS) - 1;
/// Limbs per number: 160 * 13 = 2080 bits.
pub const LIMBS: usize = 160;
/// Montgomery radix is R = 2^R_BITS.
pub const R_BITS: u32 = LIMB_BITS * LIMBS as u32;

/// Small deterministic RNG (xorshift64); test data only.
#[derive(Clone)]
pub struct XorShift64(u64);

impl XorShift64 {
    pub fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

/// Uniform random number below 2^bits.
pub fn random_bits(rng: &mut XorShift64, bits: u32) -> BigUint {
    let nbytes = bits.div_ceil(8) as usize;
    let mut bytes = Vec::with_capacity(nbytes + 8);
    while bytes.len() < nbytes {
        bytes.extend_from_slice(&rng.next_u64().to_le_bytes());
    }
    bytes.truncate(nbytes);
    BigUint::from_bytes_le(&bytes) & ((BigUint::one() << bits) - 1u32)
}

/// Odd number of exactly `bits` bits.
pub fn random_odd_modulus(rng: &mut XorShift64, bits: u32) -> BigUint {
    random_bits(rng, bits) | (BigUint::one() << (bits - 1)) | BigUint::one()
}

/// Random number in [0, n).
pub fn random_below(rng: &mut XorShift64, n: &BigUint) -> BigUint {
    random_bits(rng, n.bits() as u32 + 64) % n
}

/// Split x into LIMBS limbs of 13 bits (little-endian), one per u32.
pub fn pack_limbs(x: &BigUint) -> Vec<u32> {
    assert!(x.bits() <= u64::from(R_BITS), "value does not fit in {LIMBS} limbs");
    let d = x.to_u32_digits();
    (0..LIMBS)
        .map(|k| {
            let o = LIMB_BITS as usize * k;
            let (w, s) = (o / 32, o % 32);
            let lo = u64::from(d.get(w).copied().unwrap_or(0));
            let hi = u64::from(d.get(w + 1).copied().unwrap_or(0));
            (((lo | (hi << 32)) >> s) as u32) & LIMB_MASK
        })
        .collect()
}

/// Inverse of `pack_limbs`; limbs may exceed 13 bits (carries are summed in).
pub fn unpack_limbs(limbs: &[u32]) -> BigUint {
    let mut digits = vec![0u32; (limbs.len() * LIMB_BITS as usize).div_ceil(32) + 3];
    for (k, &v) in limbs.iter().enumerate() {
        let o = LIMB_BITS as usize * k;
        let (w, s) = (o / 32, o % 32);
        let mut carry = u64::from(v) << s;
        let mut i = w;
        while carry != 0 {
            let sum = u64::from(digits[i]) + (carry & 0xffff_ffff);
            digits[i] = sum as u32;
            carry = (carry >> 32) + (sum >> 32);
            i += 1;
        }
    }
    BigUint::new(digits)
}

/// -n^-1 mod 2^13 for odd n (Newton iteration on the low word).
pub fn neg_inv_limb(n: &BigUint) -> u32 {
    let n0 = n.to_u32_digits().first().copied().unwrap_or(0);
    assert!(n0 & 1 == 1, "modulus must be odd");
    let mut inv = n0; // correct to 3 bits
    for _ in 0..4 {
        // 3 -> 6 -> 12 -> 24 -> 48 correct bits
        inv = inv.wrapping_mul(2u32.wrapping_sub(n0.wrapping_mul(inv)));
    }
    inv.wrapping_neg() & LIMB_MASK
}

/// n^-1 mod 2^bits for odd n (Newton / Hensel lifting).
pub fn inv_mod_pow2(n: &BigUint, bits: u32) -> BigUint {
    let mut inv = BigUint::one(); // n * 1 = 1 mod 2 for odd n
    let mut p = 1u32;
    while p < bits {
        p = (p * 2).min(bits);
        let modulus = BigUint::one() << p;
        let t = (n * &inv) % &modulus; // n * inv mod 2^p
        let two_minus_t = (BigUint::from(2u32) + &modulus - t) % &modulus;
        inv = (inv * two_minus_t) % &modulus;
    }
    inv
}

/// x <- x^2 * R^-1 mod n, `squarings` times, with R = 2^R_BITS. Result in [0, n).
pub fn mont_square_chain(n: &BigUint, x0: &BigUint, squarings: u32) -> BigUint {
    let mask = (BigUint::one() << R_BITS) - 1u32;
    let ninv = inv_mod_pow2(n, R_BITS);
    let mut x = x0.clone();
    for _ in 0..squarings {
        let x2 = &x * &x;
        // m = -x2 * n^-1 mod R, so that x2 + m*n is divisible by R
        let m = ((&mask + 1u32) - ((&x2 * &ninv) & &mask)) & &mask;
        x = ((x2 + m * n) >> R_BITS) % n;
    }
    x
}
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cargo test -p bigmont-oracle --manifest-path N:\Programming\bigmont-wgpu\Cargo.toml`
Expected: 7 tests PASS. If `unpack_accepts_unnormalized_limbs`'s second assertion fails because of the arithmetic in the expected value, recompute it by hand: limbs `[0xffffffff, 0xffffffff]` are `0xffffffff + 0xffffffff * 2^13`; fix the expected expression in the test, not the implementation.

- [ ] **Step 6: Commit**

```bash
cd /n/Programming/bigmont-wgpu && git add Cargo.toml .gitignore crates && git commit -m "Add workspace and CPU oracle for 13-bit-limb Montgomery arithmetic" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 2: GPU runner and the one-thread-per-number kernel (TPI = 1)

**Files:**
- Modify: `N:\Programming\bigmont-wgpu\crates\bigmont-core\Cargo.toml`
- Create: `N:\Programming\bigmont-wgpu\crates\bigmont-core\src\lib.rs` (replace the empty file)
- Create: `N:\Programming\bigmont-wgpu\crates\bigmont-core\src\mont.wgsl`
- Create: `N:\Programming\bigmont-wgpu\crates\bigmont-core\src\kernel.rs`
- Create: `N:\Programming\bigmont-wgpu\crates\bigmont-core\src\gpu.rs`
- Test: `N:\Programming\bigmont-wgpu\crates\bigmont-core\tests\chain.rs`

**Interfaces:**
- Consumes: everything listed under Task 1 "Produces".
- Produces (in `bigmont_core`):
  - `kernel::KernelConfig { pub tpi: u32, pub wg: u32 }` (`Clone, Copy, Debug, PartialEq, Eq`) with `fn validate(&self) -> Result<(), String>`, `fn needs_subgroups(&self) -> bool`, `fn instances_per_workgroup(&self) -> u32`, `fn limbs_per_lane(&self) -> u32`, `fn wgsl(&self) -> String`
  - `gpu::Gpu { pub info: wgpu::AdapterInfo, pub has_subgroups: bool, pub device: wgpu::Device, pub queue: wgpu::Queue }` with `fn new() -> Result<Gpu, String>`, `fn list_adapters() -> Vec<String>`, `fn check_subgroups(&self, tpi: u32) -> Result<(), String>`
  - `gpu::Batch { pub count: u32, pub n: Vec<u32>, pub ninv: Vec<u32>, pub x: Vec<u32> }` with `fn from_values(values: &[(BigUint, BigUint)]) -> Batch` and `fn random_values(count: u32, bits: u32, seed: u64) -> Vec<(BigUint, BigUint)>`
  - `gpu::Run { pub x: Vec<u32>, pub best_secs: f64 }` and `fn run(gpu: &Gpu, cfg: KernelConfig, batch: &Batch, squarings: u32, repeats: u32) -> Result<Run, String>`. Semantics: each of the `repeats` dispatches performs `squarings` Montgomery squarings on the buffer as left by the previous dispatch; `best_secs` is the fastest dispatch (submit to completion); `x` is read back after the last one.
- Kernel contract: for each instance, `x <- x^2 * R^-1 mod n` (result congruent mod n, in `[0, 2n)`, limbs fully normalised to below 2^13), repeated `squarings` times. Requires input `x < 2n` and limbs below 2^13.

**Algorithm (what the WGSL implements).** Instance = `TPI` lanes; lane `j` owns limbs `j*LPL .. j*LPL+LPL-1` (`LPL = 160 / TPI`) of `x`, `n` and the accumulator `t`. For `i` in `0..160`: broadcast limb `b = x[i]` from its owner lane; `t[c] += x[c] * b`; broadcast `t[0]` of lane 0, `m = (t0 * ninv) & 0x1fff`; `t[c] += n[c] * m`; shift the accumulator down one limb (lane `j`'s top limb takes lane `j+1`'s limb 0; lane 0 adds `carry0 = (new t0) >> 13` into its limb 0 after the shift). Every 16 iterations run a "light pass" (each limb keeps its low 13 bits, the carry moves one limb up; the carry out of a lane's top limb goes to the next lane's limb 0). After the loop, repeat light passes until no limb is at or above 2^13. Why 16: products are below 2^26, each iteration adds at most two per column, and a light pass leaves limbs below 2^19 + 2^13, so 16 iterations stay below 2^32 (about 2.16e9). The bound on the result is `< (4n^2 + R*n)/R < 1.5n` because `R > 8n`, so `x < 2n` is preserved with no compare.

- [ ] **Step 1: Cargo manifest**

`N:\Programming\bigmont-wgpu\crates\bigmont-core\Cargo.toml`:

```toml
[package]
name = "bigmont-core"
description = "Portable (wgpu) 2048-bit Montgomery arithmetic kernels"
version.workspace = true
edition.workspace = true
license.workspace = true

[dependencies]
bigmont-oracle = { path = "../bigmont-oracle" }
bytemuck = { version = "1.25.2", features = ["derive"] }
num-bigint = "0.4"
num-traits = "0.2"
pollster = "1.0.3"
wgpu = "30.0.2"
```

- [ ] **Step 2: Write the failing test**

`N:\Programming\bigmont-wgpu\crates\bigmont-core\tests\chain.rs`:

```rust
use bigmont_core::gpu::{Batch, Gpu, run};
use bigmont_core::kernel::KernelConfig;
use bigmont_oracle::*;
use num_bigint::BigUint;
use num_traits::One;

/// Runs `cfg` on the first adapter and compares every instance with the oracle.
/// Skips (loudly) only when there is no adapter or the adapter cannot run this configuration.
fn check(cfg: KernelConfig) {
    let gpu = match Gpu::new() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("SKIP {cfg:?}: no adapter: {e}");
            return;
        }
    };
    if cfg.needs_subgroups() {
        if let Err(e) = gpu.check_subgroups(cfg.tpi) {
            eprintln!("SKIP {cfg:?}: {e}");
            return;
        }
    }
    eprintln!("adapter: {} ({:?}), config {cfg:?}", gpu.info.name, gpu.info.backend);

    let mut values = Batch::random_values(61, 2029, 7);
    // Carry-ripple stress: n = 2^2029 - 1 (every limb all ones) and n = 2^2028 + 1 (long zero runs).
    let all_ones = (BigUint::one() << 2029) - 1u32;
    values.push((all_ones.clone(), &all_ones - 1u32));
    let sparse = (BigUint::one() << 2028) + 1u32;
    values.push((sparse.clone(), &sparse - 1u32));

    let batch = Batch::from_values(&values);
    let squarings = 3;
    let result = run(&gpu, cfg, &batch, squarings, 1).expect("run failed");
    for (i, (n, x0)) in values.iter().enumerate() {
        let limbs = &result.x[i * LIMBS..(i + 1) * LIMBS];
        assert!(limbs.iter().all(|&l| l <= LIMB_MASK), "instance {i}: limbs not normalised");
        let got = unpack_limbs(limbs);
        assert!(got < n * 2u32, "instance {i}: result not below 2n");
        assert_eq!(&got % n, mont_square_chain(n, x0, squarings), "instance {i}: wrong value");
    }
}

#[test]
fn tpi1_matches_oracle() {
    check(KernelConfig { tpi: 1, wg: 64 });
}
```

- [ ] **Step 3: Run it to verify it fails**

Run: `cargo test -p bigmont-core --manifest-path N:\Programming\bigmont-wgpu\Cargo.toml --test chain`
Expected: compile FAIL (`bigmont_core::gpu` / `kernel` not found).

- [ ] **Step 4: Write the kernel template**

`N:\Programming\bigmont-wgpu\crates\bigmont-core\src\mont.wgsl`:

```wgsl
__ENABLE__
const TPI: u32 = __TPI__;     // lanes per number
const LPL: u32 = __LPL__;     // limbs per lane; TPI * LPL = 160
const L: u32 = TPI * LPL;
const WG: u32 = __WG__;
const B: u32 = 13u;
const MASK: u32 = 0x1fffu;
const PERIOD: u32 = 16u;      // iterations between light carry passes (see plan: bound < 2^32)

@group(0) @binding(0) var<uniform> params: vec4<u32>;          // x = instance count, y = squarings
@group(0) @binding(1) var<storage, read> n_limbs: array<u32>;
@group(0) @binding(2) var<storage, read> ninv_buf: array<u32>;
@group(0) @binding(3) var<storage, read_write> x_limbs: array<u32>;

var<private> xx: array<u32, LPL>;
var<private> nn: array<u32, LPL>;
var<private> tt: array<u32, LPL>;

__SHUFFLES__

// Each limb keeps its low 13 bits; the carry moves one limb up (across lanes for the top limb).
fn light_pass(lane: u32) {
    var c_in = up1(tt[LPL - 1u] >> B, lane);   // read before any lane modifies its limbs
    for (var c = 0u; c < LPL; c++) {
        let v = tt[c];
        tt[c] = (v & MASK) + c_in;
        c_in = v >> B;
    }
}

// Repeat light passes until every limb of every instance in the subgroup is below 2^13.
fn full_normalize(lane: u32) {
    for (var round = 0u; round < L + 2u; round++) {
        var big = 0u;
        for (var c = 0u; c < LPL; c++) { big |= tt[c] >> B; }
        if (!any_nonzero(big)) { break; }
        light_pass(lane);
    }
}

// tt <- xx * xx * R^-1 (mod n), tt normalised, value below 2n.
fn mont_sqr(lane: u32, base: u32, ninv: u32) {
    for (var c = 0u; c < LPL; c++) { tt[c] = 0u; }
    for (var i = 0u; i < L; i++) {
        let bi = bcast(xx[i % LPL], i / LPL, base);
        for (var c = 0u; c < LPL; c++) {
            tt[c] += (xx[c] & 0xffffffu) * (bi & 0xffffffu);
        }
        let m = (bcast(tt[0], 0u, base) * ninv) & MASK;
        for (var c = 0u; c < LPL; c++) {
            tt[c] += (nn[c] & 0xffffffu) * m;
        }
        let carry0 = bcast(tt[0], 0u, base) >> B;   // limb 0 is now divisible by 2^13
        let next_first = down1(tt[0], lane);        // limb 0 of the next lane becomes our new top limb
        for (var c = 0u; c + 1u < LPL; c++) { tt[c] = tt[c + 1u]; }
        tt[LPL - 1u] = next_first;
        if (lane == 0u) { tt[0] += carry0; }
        if ((i % PERIOD) == PERIOD - 1u) { light_pass(lane); }
    }
    full_normalize(lane);
}

@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32 __SGPARAM__) {
    __SGLET__
    let lane = sg % TPI;
    let base = sg - lane;
    let inst = wid.x * (WG / TPI) + li / TPI;
    let live = inst < params.x;
    let ic = min(inst, params.x - 1u);     // out-of-range lanes compute on a valid instance; their store is skipped
    let off = ic * L + lane * LPL;
    for (var c = 0u; c < LPL; c++) {
        xx[c] = x_limbs[off + c];
        nn[c] = n_limbs[off + c];
    }
    let ninv = ninv_buf[ic];
    for (var s = 0u; s < params.y; s++) {
        mont_sqr(lane, base, ninv);
        for (var c = 0u; c < LPL; c++) { xx[c] = tt[c]; }
    }
    if (live) {
        for (var c = 0u; c < LPL; c++) { x_limbs[off + c] = xx[c]; }
    }
}
```

- [ ] **Step 5: Write `kernel.rs`**

`N:\Programming\bigmont-wgpu\crates\bigmont-core\src\kernel.rs`:

```rust
//! Kernel configuration and WGSL generation.

use bigmont_oracle::LIMBS;

const TEMPLATE: &str = include_str!("mont.wgsl");

/// Scalar helpers for TPI = 1: no cross-lane traffic at all.
const SCALAR_HELPERS: &str = "
fn bcast(v: u32, src_lane: u32, base: u32) -> u32 { return v; }
fn down1(v: u32, lane: u32) -> u32 { return 0u; }
fn up1(v: u32, lane: u32) -> u32 { return 0u; }
fn any_nonzero(v: u32) -> bool { return v != 0u; }
";

/// Subgroup helpers for TPI > 1; `base` is the subgroup lane id of this instance's lane 0.
const SUBGROUP_HELPERS: &str = "
fn bcast(v: u32, src_lane: u32, base: u32) -> u32 { return subgroupShuffle(v, base + src_lane); }
fn down1(v: u32, lane: u32) -> u32 { return select(0u, subgroupShuffleDown(v, 1u), lane != TPI - 1u); }
fn up1(v: u32, lane: u32) -> u32 { return select(0u, subgroupShuffleUp(v, 1u), lane != 0u); }
fn any_nonzero(v: u32) -> bool { return subgroupAny(v != 0u); }
";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KernelConfig {
    /// Lanes per number: 1 (one thread per number) or a cooperative width.
    pub tpi: u32,
    /// Workgroup size (invocations).
    pub wg: u32,
}

impl KernelConfig {
    pub fn validate(&self) -> Result<(), String> {
        if !matches!(self.tpi, 1 | 8 | 16 | 32) {
            return Err(format!("tpi {} not supported (1, 8, 16, 32)", self.tpi));
        }
        if LIMBS as u32 % self.tpi != 0 {
            return Err(format!("{LIMBS} limbs not divisible by tpi {}", self.tpi));
        }
        if self.wg == 0 || self.wg % self.tpi != 0 || self.wg > 256 {
            return Err(format!("workgroup size {} must be a multiple of tpi {} and at most 256", self.wg, self.tpi));
        }
        Ok(())
    }

    pub fn needs_subgroups(&self) -> bool {
        self.tpi > 1
    }

    pub fn instances_per_workgroup(&self) -> u32 {
        self.wg / self.tpi
    }

    pub fn limbs_per_lane(&self) -> u32 {
        LIMBS as u32 / self.tpi
    }

    /// The complete WGSL source for this configuration.
    pub fn wgsl(&self) -> String {
        let (enable, sgparam, sglet, helpers) = if self.needs_subgroups() {
            ("enable subgroups;", ", @builtin(subgroup_invocation_id) sg_in: u32", "let sg = sg_in;", SUBGROUP_HELPERS)
        } else {
            ("", "", "let sg = 0u;", SCALAR_HELPERS)
        };
        TEMPLATE
            .replace("__ENABLE__", enable)
            .replace("__TPI__", &format!("{}u", self.tpi))
            .replace("__LPL__", &format!("{}u", self.limbs_per_lane()))
            .replace("__WG__", &format!("{}u", self.wg))
            .replace("__SGPARAM__", sgparam)
            .replace("__SGLET__", sglet)
            .replace("__SHUFFLES__", helpers)
    }
}
```

- [ ] **Step 6: Write `gpu.rs` and `lib.rs`**

`N:\Programming\bigmont-wgpu\crates\bigmont-core\src\lib.rs`:

```rust
//! Portable (wgpu) 2048-bit Montgomery arithmetic kernels. Phase 0: speed spike.

pub mod gpu;
pub mod kernel;
```

`N:\Programming\bigmont-wgpu\crates\bigmont-core\src\gpu.rs`:

```rust
//! wgpu device setup, input batches, and the kernel runner.

use crate::kernel::KernelConfig;
use bigmont_oracle::{LIMBS, XorShift64, neg_inv_limb, pack_limbs, random_below, random_odd_modulus};
use num_bigint::BigUint;
use std::time::Instant;

pub struct Gpu {
    pub info: wgpu::AdapterInfo,
    pub has_subgroups: bool,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

fn describe(info: &wgpu::AdapterInfo) -> String {
    format!("{} [{:?}, {:?}] subgroup {}..{}", info.name, info.backend, info.device_type, info.subgroup_min_size, info.subgroup_max_size)
}

impl Gpu {
    /// Picks the adapter named by env `BIGMONT_ADAPTER` (substring of the adapter name) and
    /// `BIGMONT_BACKEND` (substring of the backend name, e.g. "Vulkan", "Dx12"); otherwise the
    /// high-performance default. Requests `Features::SUBGROUP` when the adapter has it.
    pub fn new() -> Result<Gpu, String> {
        pollster::block_on(async {
            let instance = wgpu::Instance::default();
            let name = std::env::var("BIGMONT_ADAPTER").ok().map(|s| s.to_lowercase());
            let backend = std::env::var("BIGMONT_BACKEND").ok().map(|s| s.to_lowercase());
            let adapter = if name.is_some() || backend.is_some() {
                instance
                    .enumerate_adapters(wgpu::Backends::all())
                    .await
                    .into_iter()
                    .find(|a| {
                        let i = a.get_info();
                        name.as_ref().is_none_or(|w| i.name.to_lowercase().contains(w))
                            && backend.as_ref().is_none_or(|w| format!("{:?}", i.backend).to_lowercase().contains(w))
                    })
                    .ok_or("no adapter matches BIGMONT_ADAPTER / BIGMONT_BACKEND")?
            } else {
                instance
                    .request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, ..Default::default() })
                    .await
                    .map_err(|e| e.to_string())?
            };
            let info = adapter.get_info();
            let has_subgroups = adapter.features().contains(wgpu::Features::SUBGROUP);
            let mut features = wgpu::Features::empty();
            if has_subgroups {
                features |= wgpu::Features::SUBGROUP;
            }
            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor { required_features: features, ..Default::default() })
                .await
                .map_err(|e| e.to_string())?;
            Ok(Gpu { info, has_subgroups, device, queue })
        })
    }

    pub fn list_adapters() -> Vec<String> {
        pollster::block_on(async {
            let instance = wgpu::Instance::default();
            instance
                .enumerate_adapters(wgpu::Backends::all())
                .await
                .iter()
                .map(|a| {
                    let i = a.get_info();
                    format!("{} subgroups={}", describe(&i), a.features().contains(wgpu::Features::SUBGROUP))
                })
                .collect()
        })
    }

    pub fn describe(&self) -> String {
        describe(&self.info)
    }

    /// Whether `tpi` lanes per number fit this adapter's subgroups.
    pub fn check_subgroups(&self, tpi: u32) -> Result<(), String> {
        if !self.has_subgroups {
            return Err("adapter lacks wgpu::Features::SUBGROUP".into());
        }
        let (lo, hi) = (self.info.subgroup_min_size, self.info.subgroup_max_size);
        if lo < tpi || lo % tpi != 0 || hi % tpi != 0 {
            return Err(format!("subgroup size {lo}..{hi} is not a multiple of tpi {tpi}"));
        }
        Ok(())
    }
}

/// Input arrays, one entry per instance: moduli, -n^-1 mod 2^13, and starting values, as limbs.
pub struct Batch {
    pub count: u32,
    pub n: Vec<u32>,
    pub ninv: Vec<u32>,
    pub x: Vec<u32>,
}

impl Batch {
    pub fn from_values(values: &[(BigUint, BigUint)]) -> Batch {
        let mut b = Batch { count: values.len() as u32, n: Vec::new(), ninv: Vec::new(), x: Vec::new() };
        for (n, x) in values {
            assert!(x < &(n * 2u32), "x must be below 2n");
            b.n.extend(pack_limbs(n));
            b.ninv.push(neg_inv_limb(n));
            b.x.extend(pack_limbs(x));
        }
        b
    }

    /// `count` random (odd modulus of exactly `bits` bits, x below modulus) pairs.
    pub fn random_values(count: u32, bits: u32, seed: u64) -> Vec<(BigUint, BigUint)> {
        let mut rng = XorShift64::new(seed);
        (0..count)
            .map(|_| {
                let n = random_odd_modulus(&mut rng, bits);
                let x = random_below(&mut rng, &n);
                (n, x)
            })
            .collect()
    }
}

pub struct Run {
    pub x: Vec<u32>,
    pub best_secs: f64,
}

/// Uploads `batch`, runs `repeats` dispatches of `squarings` Montgomery squarings each (every
/// dispatch continues from the previous one's output), and reads the result back.
pub fn run(gpu: &Gpu, cfg: KernelConfig, batch: &Batch, squarings: u32, repeats: u32) -> Result<Run, String> {
    cfg.validate()?;
    if cfg.needs_subgroups() {
        gpu.check_subgroups(cfg.tpi)?;
    }
    let (device, queue) = (&gpu.device, &gpu.queue);
    let words = (batch.count as usize * LIMBS) as u64;
    let storage = |label: &str, size: u64, extra: wgpu::BufferUsages| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | extra,
            mapped_at_creation: false,
        })
    };
    let params = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("params"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let n_buf = storage("n", words * 4, wgpu::BufferUsages::empty());
    let ninv_buf = storage("ninv", u64::from(batch.count) * 4, wgpu::BufferUsages::empty());
    let x_buf = storage("x", words * 4, wgpu::BufferUsages::COPY_SRC);
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size: words * 4,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&params, 0, bytemuck::cast_slice(&[batch.count, squarings, 0u32, 0u32]));
    queue.write_buffer(&n_buf, 0, bytemuck::cast_slice(&batch.n));
    queue.write_buffer(&ninv_buf, 0, bytemuck::cast_slice(&batch.ninv));
    queue.write_buffer(&x_buf, 0, bytemuck::cast_slice(&batch.x));

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("mont"), source: wgpu::ShaderSource::Wgsl(cfg.wgsl().into()) });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("mont"),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: n_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: ninv_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: x_buf.as_entire_binding() },
        ],
    });
    let groups = batch.count.div_ceil(cfg.instances_per_workgroup());
    let mut best = f64::MAX;
    for _ in 0..repeats.max(1) {
        let mut enc = device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        let t = Instant::now();
        queue.submit([enc.finish()]);
        device.poll(wgpu::PollType::wait_indefinitely()).map_err(|e| format!("GPU poll failed: {e}"))?;
        best = best.min(t.elapsed().as_secs_f64());
    }
    let mut enc = device.create_command_encoder(&Default::default());
    enc.copy_buffer_to_buffer(&x_buf, 0, &staging, 0, words * 4);
    queue.submit([enc.finish()]);
    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("mapping readback buffer failed"));
    device.poll(wgpu::PollType::wait_indefinitely()).map_err(|e| format!("GPU poll failed: {e}"))?;
    let x = {
        let view = slice.get_mapped_range().expect("readback buffer not mapped");
        bytemuck::cast_slice::<u8, u32>(&view).to_vec()
    };
    staging.unmap();
    Ok(Run { x, best_secs: best })
}
```

API note: these signatures follow wgpu 30.0.2 as used in `N:\Programming\gapsieve\examples\mul_bench.rs` and `src\gpu.rs`. If the compiler disagrees on a detail (for example `enumerate_adapters` returning a plain `Vec`, or `get_mapped_range` returning a view directly), follow the compiler and keep the semantics.

- [ ] **Step 7: Run the test to verify it passes**

Run: `cargo test -p bigmont-core --manifest-path N:\Programming\bigmont-wgpu\Cargo.toml --test chain -- --nocapture`
Expected: `tpi1_matches_oracle ... ok`, and the output names the adapter. If it prints `SKIP ... no adapter`, stop and report that, because that means the test did not exercise the kernel.

If the kernel is wrong, debug in this order: (1) one instance, `squarings = 0` should return the input (add a temporary assertion), (2) `squarings = 1` with a tiny `n`-sized `x`, (3) compare limb-by-limb against the oracle's `mont_square_chain` before `% n` (the GPU value is congruent but may be up to `2n`, so compare modulo n).

- [ ] **Step 8: Commit**

```bash
cd /n/Programming/bigmont-wgpu && git add crates && git commit -m "Add wgpu runner and one-thread-per-number Montgomery squaring kernel" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Subgroup-cooperative kernels (TPI = 8, 16, 32)

**Files:**
- Modify: `N:\Programming\bigmont-wgpu\crates\bigmont-core\tests\chain.rs`
- Modify (only if the WGSL needs fixes): `N:\Programming\bigmont-wgpu\crates\bigmont-core\src\mont.wgsl`, `N:\Programming\bigmont-wgpu\crates\bigmont-core\src\kernel.rs`

**Interfaces:**
- Consumes: `KernelConfig`, `Gpu::check_subgroups`, `run`, `Batch` from Task 2.
- Produces: validated cooperative configurations `KernelConfig { tpi: 8 | 16 | 32, wg: 64 }`. No new public API.

Most of the cooperative code already exists in the template and `SUBGROUP_HELPERS` from Task 2; this task proves it, and fixes what fails.

- [ ] **Step 1: Add the failing tests**

Append to `N:\Programming\bigmont-wgpu\crates\bigmont-core\tests\chain.rs`:

```rust
#[test]
fn tpi8_matches_oracle() {
    check(KernelConfig { tpi: 8, wg: 64 });
}

#[test]
fn tpi16_matches_oracle() {
    check(KernelConfig { tpi: 16, wg: 64 });
}

#[test]
fn tpi32_matches_oracle() {
    check(KernelConfig { tpi: 32, wg: 64 });
}

#[test]
fn tpi32_other_workgroup_sizes_match_oracle() {
    check(KernelConfig { tpi: 32, wg: 128 });
    check(KernelConfig { tpi: 32, wg: 256 });
}
```

- [ ] **Step 2: Run them**

Run: `cargo test -p bigmont-core --manifest-path N:\Programming\bigmont-wgpu\Cargo.toml --test chain -- --nocapture`
Expected on an adapter with subgroups whose size is a multiple of the TPI: all PASS. Expected where the adapter cannot run a config: a `SKIP` line naming the reason (that is acceptable, but record which configs skipped).

- [ ] **Step 3: If a cooperative test fails, fix the kernel, not the test**

Likely causes, in order:
1. **Shader rejected.** If naga reports a uniformity error on `subgroupShuffle*` or `subgroupAny` inside loops, first confirm the loop conditions really are subgroup-uniform (they are: `i`, `c`, `round` are loop counters; `break` depends on `subgroupAny`). If the diagnostic is still raised, add `diagnostic(off, subgroup_uniformity);` as a module-level directive to the template (inside the `__ENABLE__` replacement for the cooperative case) and note it in the commit message.
2. **Wrong lane mapping.** The kernel assumes the subgroup's lanes are consecutive `local_invocation_index` values. If results are wrong only for instances that are not first in their workgroup, print `sg_in` against `li` for one workgroup (write both into `x_limbs` from a throwaway kernel variant) and confirm `li % subgroup_size == sg_in`. If not, report it as a finding; do not paper over it.
3. **Cross-lane carry bug.** Run `squarings = 1` on `n = 2^2029 - 1`. If only that case fails, the bug is in `light_pass` or `full_normalize` handling of a long carry chain across lanes: check that `full_normalize`'s round cap `L + 2` is reached or that the `any_nonzero` vote covers the whole subgroup.
4. **TPI-1 passes and TPI > 1 fails on every instance.** Compare the accumulator after one iteration against a CPU model of the same algorithm; the shift step (`down1`, `carry0`) is the most likely culprit.

- [ ] **Step 4: Run the whole test suite**

Run: `cargo test --workspace --manifest-path N:\Programming\bigmont-wgpu\Cargo.toml -- --nocapture`
Expected: all PASS (skips allowed only as described). Record in the commit message which configurations passed and which skipped, with the adapter name.

- [ ] **Step 5: Commit**

```bash
cd /n/Programming/bigmont-wgpu && git add crates && git commit -m "Validate subgroup-cooperative Montgomery kernels (TPI 8, 16, 32) against the oracle" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Benchmark sweep example

**Files:**
- Create: `N:\Programming\bigmont-wgpu\crates\bigmont-core\examples\spike.rs`

**Interfaces:**
- Consumes: `Gpu::{new, list_adapters, describe, check_subgroups}`, `Batch::{random_values, from_values}`, `run`, `KernelConfig`, and the oracle (`mont_square_chain`, `unpack_limbs`, `LIMBS`, `LIMB_MASK`).
- Produces: the executable `spike` that prints a validation line and a throughput row per configuration.

Output columns per row: `tpi`, `wg`, `instances`, `squarings/dispatch`, `best ms`, `squarings/s`, `useful Gmad/s` (= squarings/s × 2 × 160² / 1e9; each squaring does 160 iterations of two 160-limb multiply-adds), `projected Fermat tests/s` (= squarings/s / 2029; assumes one squaring per exponent bit with the base-2 doubling folded into the reduction, as in gapsieve's scheme).

- [ ] **Step 1: Write the example**

`N:\Programming\bigmont-wgpu\crates\bigmont-core\examples\spike.rs`:

```rust
//! Phase 0 speed spike: validates each kernel configuration against the oracle, then measures
//! Montgomery squarings per second.
//!
//!   cargo run --release -p bigmont-core --example spike
//!   cargo run --release -p bigmont-core --example spike -- --list
//!   BIGMONT_ADAPTER=6700 BIGMONT_BACKEND=vulkan cargo run --release -p bigmont-core --example spike
//!   cargo run --release -p bigmont-core --example spike -- --quick     (fewer configurations)

use bigmont_core::gpu::{Batch, Gpu, run};
use bigmont_core::kernel::KernelConfig;
use bigmont_oracle::{LIMB_MASK, LIMBS, mont_square_chain, unpack_limbs};

const BITS: u32 = 2029;
const MAD_PER_SQUARING: f64 = 2.0 * (LIMBS * LIMBS) as f64;

/// Validates one configuration on 50 random instances (3 squarings). Returns Err with details.
fn validate(gpu: &Gpu, cfg: KernelConfig) -> Result<(), String> {
    let values = Batch::random_values(50, BITS, 11);
    let batch = Batch::from_values(&values);
    let out = run(gpu, cfg, &batch, 3, 1)?;
    for (i, (n, x0)) in values.iter().enumerate() {
        let limbs = &out.x[i * LIMBS..(i + 1) * LIMBS];
        if limbs.iter().any(|&l| l > LIMB_MASK) {
            return Err(format!("instance {i}: limbs not normalised"));
        }
        let got = unpack_limbs(limbs);
        if &got % n != mont_square_chain(n, x0, 3) {
            return Err(format!("instance {i}: wrong value"));
        }
    }
    Ok(())
}

struct Row {
    cfg: KernelConfig,
    count: u32,
    squarings: u32,
    best: f64,
}

/// Grows the batch until one dispatch takes at least 50 ms (staying far below the ~2 s Windows
/// GPU timeout), then takes the best of 5 dispatches.
fn bench(gpu: &Gpu, cfg: KernelConfig) -> Result<Row, String> {
    let squarings = 2;
    let per_wg = cfg.instances_per_workgroup();
    let mut count = (1024 / per_wg).max(1) * per_wg;
    loop {
        let batch = Batch::from_values(&Batch::random_values(count, BITS, 99));
        let probe = run(gpu, cfg, &batch, squarings, 1)?;
        if probe.best_secs >= 0.05 || count >= 65536 {
            let timed = run(gpu, cfg, &batch, squarings, 5)?;
            return Ok(Row { cfg, count, squarings, best: timed.best_secs });
        }
        count *= 2;
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--list") {
        for a in Gpu::list_adapters() {
            println!("{a}");
        }
        return;
    }
    let quick = args.iter().any(|a| a == "--quick");
    let gpu = Gpu::new().expect("no usable adapter");
    println!("adapter: {}", gpu.describe());
    println!("subgroups feature: {}", gpu.has_subgroups);
    println!();

    let tpis = [1u32, 8, 16, 32];
    let wgs: &[u32] = if quick { &[64] } else { &[64, 128, 256] };
    println!("{:>4} {:>4} {:>9} {:>5} {:>9} {:>12} {:>10} {:>12}  status", "tpi", "wg", "instances", "sq/d", "best ms", "squarings/s", "Gmad/s", "tests/s*");
    for &tpi in &tpis {
        for &wg in wgs {
            let cfg = KernelConfig { tpi, wg };
            if let Err(e) = cfg.validate() {
                println!("{tpi:>4} {wg:>4}  skipped: {e}");
                continue;
            }
            if cfg.needs_subgroups() {
                if let Err(e) = gpu.check_subgroups(tpi) {
                    println!("{tpi:>4} {wg:>4}  skipped: {e}");
                    continue;
                }
            }
            if let Err(e) = validate(&gpu, cfg) {
                println!("{tpi:>4} {wg:>4}  VALIDATION FAILED: {e} (no speed reported)");
                continue;
            }
            match bench(&gpu, cfg) {
                Ok(r) => {
                    let sps = f64::from(r.count) * f64::from(r.squarings) / r.best;
                    println!(
                        "{:>4} {:>4} {:>9} {:>5} {:>9.2} {:>12.0} {:>10.1} {:>12.0}  ok",
                        r.cfg.tpi,
                        r.cfg.wg,
                        r.count,
                        r.squarings,
                        r.best * 1e3,
                        sps,
                        sps * MAD_PER_SQUARING / 1e9,
                        sps / f64::from(BITS)
                    );
                }
                Err(e) => println!("{tpi:>4} {wg:>4}  bench failed: {e}"),
            }
        }
    }
    println!();
    println!("* tests/s = squarings/s / {BITS}: one squaring per exponent bit of a base-2 Fermat test (doubling folded into the reduction). A projection, not a measurement of a full test.");
}
```

- [ ] **Step 2: Build and smoke-test**

Run: `cargo run --release -p bigmont-core --example spike --manifest-path N:\Programming\bigmont-wgpu\Cargo.toml -- --quick`
Expected: the adapter line, then one row per `tpi` with status `ok` (or a `skipped:` reason for cooperative widths the adapter cannot run), no `VALIDATION FAILED`. If a dispatch looks like it approaches 2 s, or the device is lost, reduce the starting `count` or `squarings` and re-run; do not change the kernel.

- [ ] **Step 3: Commit**

```bash
cd /n/Programming/bigmont-wgpu && git add crates && git commit -m "Add Phase 0 benchmark sweep example" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Run the spike and write the results

**Files:**
- Create: `N:\Programming\bigmont-wgpu\docs\superpowers\results\2026-11-24-phase0-results.md`

**Interfaces:**
- Consumes: the `spike` example from Task 4.
- Produces: a results document the user can read to set the go/no-go threshold. No code changes.

- [ ] **Step 1: List adapters**

Run: `cargo run --release -p bigmont-core --example spike --manifest-path N:\Programming\bigmont-wgpu\Cargo.toml -- --list`
Record the output. Identify the RX 6700 XT entries (expect one per graphics API, for example Vulkan and Dx12).

- [ ] **Step 2: Run the full sweep on every API the 6700 XT appears under**

For each backend found, run (PowerShell; set the variables, run, clear them):

```powershell
$env:BIGMONT_ADAPTER = "6700"; $env:BIGMONT_BACKEND = "vulkan"
cargo run --release -p bigmont-core --example spike --manifest-path N:\Programming\bigmont-wgpu\Cargo.toml 2>&1 | Tee-Object -FilePath N:\Programming\bigmont-wgpu\docs\superpowers\results\raw-6700-vulkan.txt
$env:BIGMONT_BACKEND = "dx12"
cargo run --release -p bigmont-core --example spike --manifest-path N:\Programming\bigmont-wgpu\Cargo.toml 2>&1 | Tee-Object -FilePath N:\Programming\bigmont-wgpu\docs\superpowers\results\raw-6700-dx12.txt
Remove-Item Env:BIGMONT_ADAPTER, Env:BIGMONT_BACKEND
```

If an adapter named differently (for example the iGPU) is the default and the 6700 XT does not appear, say so in the results. Do not run on other machines.

- [ ] **Step 3: Gather comparison figures (read-only)**

- Peak 24-bit multiply-add rate for this card: run `cargo run --release --example mul_bench` from `N:\Programming\gapsieve` (it needs `SHADER_INT64`; if the 6700 XT cannot provide it, skip and say so). Report the `mad_u32_u24` line as the ceiling to compare "useful Gmad/s" against.
- Beacon's published figure, from its README: 516-523k tests/s per RTX 5090 at the 2048-bit class (the README says "measured 516–523k tests/s/card at the 2048-bit class" for its Fermat kernel). Note in the results that it is a different card and that I have not confirmed whether it counts full tests or early-exit tests. Do not draw a conclusion from the ratio beyond "order of magnitude".

- [ ] **Step 4: Write the results document**

`N:\Programming\bigmont-wgpu\docs\superpowers\results\2026-11-24-phase0-results.md` must contain:
1. Machine and adapter(s): names, backends, subgroup min/max sizes, driver strings if shown.
2. The two result tables (one per backend), copied from the raw output, with validation status for every row.
3. Which configurations were skipped and why.
4. Efficiency: best useful Gmad/s as a fraction of the `mad_u32_u24` ceiling, if measured.
5. Best configuration per backend, and the cooperative-versus-one-thread ratio.
6. The comparison with Beacon's figure, with the caveats above.
7. What was not measured: the RTX 4070 laptop, CGBN on any card, a full Fermat test (the "tests/s" column is a projection), subgroup behaviour in a browser.
8. A proposed go/no-go threshold for the user to accept or change, stated as a number with its derivation. Do not decide it; mark it "proposed".

- [ ] **Step 5: Commit**

```bash
cd /n/Programming/bigmont-wgpu && git add docs && git commit -m "Record Phase 0 speed spike results on the RX 6700 XT" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 6: Hand back to the user**

Report: the tables, the proposed threshold, and the two things only the user can decide: whether the results clear the gate, and whether to run the same sweep on the RTX 4070 laptop (copy the repo there, `cargo run --release -p bigmont-core --example spike`; no CUDA needed for that run).
