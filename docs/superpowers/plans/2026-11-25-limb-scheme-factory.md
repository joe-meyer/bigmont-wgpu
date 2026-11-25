# Limb-scheme factory (13-bit and 32-bit limbs, chosen by measurement) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the limb width (13-bit limbs in u32, or 32-bit limbs with u64 products) a first-class kernel parameter that the autotuner measures per adapter, so the engine runs the faster scheme on each GPU, with the Fermat ladder, the runner, the tuner, the cache and the CLI all working for both.

**Architecture:** A `LimbScheme` enum in `bigmont-core` owns everything that depends on the radix (limbs for a bit length, R, host-side pack/unpack, `-n^-1`, normalisation check, the oracle chain). `KernelConfig` gains a `scheme` field and a `requires() -> Caps` method; its source generators dispatch on the scheme (`unrolled.rs` for 13-bit, a new `limb32.rs` for 32-bit). `Caps` (subgroups, int64) describes both what a device offers and what a kernel needs, and a browser target offers no int64. The tuner enumerates (scheme, TPI, workgroup) candidates the adapter can run, self-tests and times each, and caches the winner with its scheme. No vendor is named anywhere: which scheme wins is a measurement.

**Tech Stack:** Rust (edition of the workspace), wgpu 30.0.2 / naga 30, generated WGSL, `num-bigint`, the existing `bigmont-oracle` and `bigmont-cpu` reference crates.

**Spec:** There is no separate spec. The binding material is the measurements and design reasoning in `docs/superpowers/results/2026-11-24-limb-scheme-experiment.md` (one-thread comparison, fair-size addendum) and the cooperative result recorded in `docs/superpowers/results/phase1/schemes/coop_bench-4070.txt` and `coop_bench-6700xt.txt` (RTX 4070 Laptop: 32-bit limbs 1.79x the 13-bit kernel at 2,029 bits; RX 6700 XT: 0.62x). The reference implementation of the 32-bit squaring kernel is `crates/bigmont-core/src/coop32.rs` at commit `c6922af` on branch `worktree-agent-a190d857c66461b4e` (read it with `git show c6922af:crates/bigmont-core/src/coop32.rs`; it is NOT on this branch). Its tests are `git show c6922af:crates/bigmont-core/tests/coop32.rs`.

## Global Constraints

- Work on branch `limb-factory` (cut from `phase1` at 481f4b7). Do not push. Do not touch `N:\Programming\gapsieve`. Nothing under `.superpowers/` or `.claude/` is ever committed.
- Commit messages end with the line `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.
- The 13-bit path must keep behaving exactly as before: same generated WGSL for `LimbScheme::Limb13` (the existing `browser_source` tests compare generated source), same cache lines for 13-bit winners (`unrolled <tpi> <wg>|<rate>`), same CLI behaviour when `--limbs` is not given.
- The browser target offers no `SHADER_INT64`. A 32-bit kernel must never be generated for `Target::Browser`; asking for one is an `Err`, and the autotuner only offers schemes whose `requires()` the device satisfies.
- No vendor, device name or vendor id may decide which scheme runs. Scheme choice comes from `Caps` (can it run) and measurement (is it faster).
- `R = 2^(limb_bits * limbs)` must exceed `16 n` for every supported `n` (the Fermat ladder doubles lazily): `limbs_for_bits(bits) = ceil((bits + 4) / limb_bits)`, then rounded up to a multiple of the lane count.
- A kernel that fails its oracle self-test on the actual adapter is skipped, never used.
- Tests that need a GPU are strict by default (a missing adapter or capability fails the test); `BIGMONT_ALLOW_SKIP=1` turns that into a loud skip. One `Gpu` per test, not one per check (opening many devices in one process exhausted the Linux NVIDIA driver's static TLS).
- Every test run on a device with a bounded timeout. Run the workspace suite as `cargo test --release --workspace --no-fail-fast -- --test-threads=1`. The implementer's machine has an AMD GPU (RX 6700 XT, 32-bit kernels run there). Use `BIGMONT_BACKEND=Vulkan` for Vulkan-forced runs. The controller runs the laptop (RTX 4070) suite.
- Scrub machine-specific user paths (home directories, usernames) from any new committed log or doc.
- Existing tests are the safety net for Task 1: they must pass unchanged (they may be edited only where a signature the plan names changes).

## File Structure

- Create `crates/bigmont-core/src/scheme.rs`: `LimbScheme`, `Caps` and the radix-dependent host helpers.
- Create `crates/bigmont-core/src/limb32.rs`: WGSL generator for the 32-bit scheme (squaring and Fermat modes), ported from `coop32.rs`.
- Modify `crates/bigmont-core/src/kernel.rs`: `KernelConfig.scheme`, constructors, `requires`, `retarget`, fallible source generators.
- Modify `crates/bigmont-core/src/gpu.rs`: `Gpu.has_int64`, `Gpu::caps`, int64 feature request, `Batch::from_values_cfg`, `run` for either scheme.
- Modify `crates/bigmont-core/src/runner.rs`: scheme-generic host preparation and verdict.
- Modify `crates/bigmont-core/src/lib.rs`: module declarations.
- Modify `crates/bigmont-cli/src/tune.rs`, `engine.rs`, `main.rs`: scheme-aware candidates, self-test, cache, engine option and the `--limbs` flag.
- Create tests: `crates/bigmont-core/tests/scheme.rs` (pure), `tests/source_matrix.rs` (naga), `tests/limb32.rs` (GPU squarings), additions to `tests/fermat.rs`, `tests/sizes.rs`, `crates/bigmont-cli/tests/{tune,engine,cli}.rs`.
- Create `crates/bigmont-core/examples/scheme_compare.rs` (the cooperative benchmark, via `KernelConfig`).

---

### Task 1: LimbScheme and Caps, with the 13-bit path moved behind them

Behaviour-neutral refactor. After it, no radix literal remains in the runner or the tuner, and `LimbScheme` has exactly one variant (Task 2 adds the second, and the compiler then points at every `match` that needs updating).

**Files:**
- Create: `crates/bigmont-core/src/scheme.rs`
- Modify: `crates/bigmont-core/src/{lib.rs, kernel.rs, gpu.rs, runner.rs}`, `crates/bigmont-cli/src/tune.rs`
- Test: `crates/bigmont-core/tests/scheme.rs`

**Interfaces:**
- Consumes: oracle `LIMB_BITS` (13), `LIMB_MASK`, `limbs_for_bits`, `pack_limbs_n`, `unpack_limbs`, `neg_inv_limb`, `mont_square_chain_bits(n, x0, squarings, r_bits)`; existing `KernelConfig`, `Gpu`, `Batch`.
- Produces (exact signatures later tasks rely on):

```rust
// bigmont-core/src/scheme.rs
use num_bigint::BigUint;

/// What a device offers, or what a kernel needs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Caps { pub subgroups: bool, pub int64: bool }
impl Caps {
    /// What a browser (WebGPU) target offers: subgroups, no 64-bit integers.
    pub const BROWSER: Caps = Caps { subgroups: true, int64: false };
    /// Capabilities of `self` that `need` requires and `self` lacks, as words ("subgroups", "int64").
    pub fn missing(self, need: Caps) -> Vec<&'static str>;
    pub fn satisfies(self, need: Caps) -> bool;      // missing(need).is_empty()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LimbScheme { Limb13 }
impl LimbScheme {
    pub const ALL: &'static [LimbScheme];
    pub fn limb_bits(self) -> u32;
    pub fn name(self) -> &'static str;                       // "13"
    pub fn parse(s: &str) -> Option<LimbScheme>;             // "13"
    pub fn requires(self) -> Caps;                           // scheme-only needs (no subgroups)
    pub fn limbs_for_bits(self, bits: u32) -> u32;           // ceil((bits + 4) / limb_bits)
    pub fn r_bits(self, limbs: u32) -> u32;                  // limb_bits * limbs
    pub fn pack(self, x: &BigUint, limbs: usize) -> Vec<u32>;
    pub fn unpack(self, limbs: &[u32]) -> BigUint;
    pub fn neg_inv(self, n: &BigUint) -> u32;                // -n^-1 mod 2^limb_bits
    pub fn normalised(self, limbs: &[u32]) -> bool;          // every limb below 2^limb_bits
    pub fn chain(self, n: &BigUint, x0: &BigUint, squarings: u32, limbs: usize) -> BigUint; // oracle chain at this radix
}
// kernel.rs
impl KernelConfig {
    pub scheme: LimbScheme,                                   // new public field, default Limb13
    pub fn for_bits_scheme(scheme: LimbScheme, bits: u32, tpi: u32, wg: u32, variant: Variant) -> Self;
    pub fn requires(&self) -> Caps;                           // scheme.requires() plus subgroups when tpi > 1
    pub fn retarget(&self, bits: u32) -> KernelConfig;        // same scheme/tpi/wg/variant, limbs for `bits`
    pub fn r_bits(&self) -> u32;
    pub fn try_wgsl_for(&self, target: Target) -> Result<String, String>;
    pub fn try_fermat_wgsl_for(&self, target: Target) -> Result<String, String>;
}
// gpu.rs
impl Gpu { pub has_int64: bool; pub fn caps(&self) -> Caps; }
impl Batch { pub fn from_values_cfg(values: &[(BigUint, BigUint)], cfg: &KernelConfig) -> Batch; }
```

`KernelConfig::for_bits(bits, tpi, wg, variant)` keeps its signature and means `for_bits_scheme(LimbScheme::Limb13, ...)`. `wgsl_for` and `fermat_wgsl_for` keep their signatures (they call the `try_` versions and `.expect` with the error text).

- [ ] **Step 1: Write the failing pure tests**

Create `crates/bigmont-core/tests/scheme.rs`:

```rust
use bigmont_core::kernel::{KernelConfig, Target, Variant};
use bigmont_core::scheme::{Caps, LimbScheme};
use bigmont_oracle::*;

#[test]
fn limb13_matches_the_oracle_helpers() {
    let s = LimbScheme::Limb13;
    assert_eq!(s.limb_bits(), 13);
    for bits in [2u32, 3, 9, 48, 64, 130, 254, 1019, 2029, 4096] {
        assert_eq!(s.limbs_for_bits(bits) as usize, limbs_for_bits(bits), "{bits} bits");
        let l = s.limbs_for_bits(bits);
        assert!(s.r_bits(l) >= bits + 4, "R must exceed 16n at {bits} bits");
    }
    let mut rng = XorShift64::new(7);
    for bits in [64u32, 2029] {
        let n = random_odd_modulus(&mut rng, bits);
        let limbs = s.limbs_for_bits(bits) as usize;
        let packed = s.pack(&n, limbs);
        assert_eq!(packed, pack_limbs_n(&n, limbs));
        assert_eq!(s.unpack(&packed), n);
        assert!(s.normalised(&packed));
        assert_eq!(s.neg_inv(&n), neg_inv_limb(&n));
    }
    assert!(!s.normalised(&[1 << 13]));
}

#[test]
fn caps_missing_and_satisfies() {
    let none = Caps::default();
    let both = Caps { subgroups: true, int64: true };
    assert!(both.satisfies(none));
    assert!(both.satisfies(both));
    assert!(!none.satisfies(Caps { subgroups: true, int64: false }));
    assert_eq!(none.missing(both), vec!["subgroups", "int64"]);
    assert!(Caps::BROWSER.subgroups && !Caps::BROWSER.int64);
}

#[test]
fn kernel_config_carries_the_scheme_and_retargets() {
    let cfg = KernelConfig::for_bits(2029, 8, 64, Variant::Unrolled);
    assert_eq!(cfg.scheme, LimbScheme::Limb13);
    assert_eq!(cfg, KernelConfig::for_bits_scheme(LimbScheme::Limb13, 2029, 8, 64, Variant::Unrolled));
    assert_eq!(cfg.requires(), Caps { subgroups: true, int64: false });
    assert_eq!(cfg.r_bits(), 13 * cfg.limbs);
    let small = cfg.retarget(100);
    assert_eq!((small.scheme, small.tpi, small.wg, small.variant), (cfg.scheme, 8, 64, Variant::Unrolled));
    assert_eq!(small, KernelConfig::for_bits(100, 8, 64, Variant::Unrolled));
    assert_eq!(cfg.try_wgsl_for(Target::Native).unwrap(), cfg.wgsl());
    assert_eq!(cfg.try_fermat_wgsl_for(Target::Native).unwrap(), cfg.fermat_wgsl_for(Target::Native));
}

#[test]
fn batch_from_values_cfg_matches_the_13_bit_batch() {
    use bigmont_core::gpu::Batch;
    let values = Batch::random_values(5, 130, 3);
    let cfg = KernelConfig::for_bits(130, 4, 64, Variant::Unrolled);
    let a = Batch::from_values_cfg(&values, &cfg);
    let b = Batch::from_values_n(&values, cfg.limbs as usize);
    assert_eq!((a.count, a.limbs, &a.n, &a.ninv, &a.x), (b.count, b.limbs, &b.n, &b.ninv, &b.x));
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test --release -p bigmont-core --test scheme`
Expected: FAIL to compile (`bigmont_core::scheme` does not exist).

- [ ] **Step 3: Implement `scheme.rs` and register it**

Create `crates/bigmont-core/src/scheme.rs` with `Caps` and `LimbScheme` as specified above. `Limb13` bodies delegate to the oracle (`LIMB_BITS`, `limbs_for_bits`, `pack_limbs_n`, `unpack_limbs`, `neg_inv_limb`, `LIMB_MASK`, `mont_square_chain_bits(n, x0, squarings, 13 * limbs as u32)`). `missing` returns words in the order subgroups, int64. `ALL` is `&[LimbScheme::Limb13]`. `requires` is `Caps::default()` for Limb13. `parse("13")` returns `Some(Limb13)`. Add `pub mod scheme;` to `lib.rs` with a one-line doc comment.

- [ ] **Step 4: Make `KernelConfig` scheme-aware (kernel.rs)**

Add `pub scheme: LimbScheme` to `KernelConfig`; `new` sets `LimbScheme::Limb13`; `for_bits` calls `for_bits_scheme(Limb13, ...)`; `for_bits_scheme` computes `need = scheme.limbs_for_bits(bits)` and rounds up to a multiple of `tpi`. `requires()` is `Caps { subgroups: self.needs_subgroups(), int64: scheme.requires().int64 }`. `retarget(bits)` is `Self::for_bits_scheme(self.scheme, bits, self.tpi, self.wg, self.variant)`. `r_bits()` is `scheme.r_bits(self.limbs)`. Move the body of `wgsl_for` and `fermat_wgsl_for` into `try_wgsl_for` / `try_fermat_wgsl_for` returning `Ok(...)` for Limb13 (the existing assert on variant for the Fermat ladder becomes an `Err`); the old names call them with `.unwrap_or_else(|e| panic!("{e}"))`. `KernelConfig::validate` is unchanged for now.

- [ ] **Step 5: gpu.rs, runner.rs, tune.rs use the scheme**

- `gpu.rs`: add `pub has_int64: bool` to `Gpu`; set it in `from_parts` from `device.features().contains(wgpu::Features::SHADER_INT64)` (no signature change); in `Gpu::open` request `SHADER_INT64` when the adapter has it, next to the existing subgroup request; add `pub fn caps(&self) -> Caps { Caps { subgroups: self.has_subgroups, int64: self.has_int64 } }`; add `Batch::from_values_cfg` using `cfg.scheme.pack/neg_inv` (assert `x < 2n` as `from_values_n` does) and make `from_values_n` call it with a Limb13 config's scheme so there is one packing path (keep `from_values_n`'s signature). Fix the `Batch` doc comment ("-n^-1 mod 2^limb_bits").
- `runner.rs`: replace `13 * cfg.limbs` with `cfg.r_bits()`; `LIMB_MASK` check with `!cfg.scheme.normalised(&x)`; `prepare(ns, limbs, words)` takes `scheme: LimbScheme` and uses `scheme.r_bits(limbs)`, `scheme.pack`, `scheme.neg_inv`; the verdict uses `cfg.scheme.unpack`. Replace the `gpu.check_subgroups(cfg.tpi)` precondition with: `let missing = gpu.caps().missing(cfg.requires()); if !missing.is_empty() { return Err(...) }` followed by the existing `gpu.check_subgroups(cfg.tpi)?` for the width test.
- `tune.rs` (cli): `self_test_inner` and `measure` use `cfg.scheme` (`r_bits`, `normalised`, `unpack`, `chain`) instead of `13`, `LIMB_MASK`, `unpack_limbs`, `mont_square_chain_n`, and build batches with `Batch::from_values_cfg`. The squaring-mode test size becomes `cfg.r_bits() - 3`. `limb_class` and `class_bits` stay defined on the 13-bit class (that is the tuning class for every scheme; Task 4 documents why).

- [ ] **Step 6: Run the whole workspace suite**

Run: `cargo test --release --workspace --no-fail-fast -- --test-threads=1`
Expected: every existing test passes (120 before this task) plus the 4 new ones in `tests/scheme.rs`. Also run `cargo build --release --workspace --examples`; examples must still compile (edit them only if a signature actually changed).

- [ ] **Step 7: Commit**

```bash
git add crates docs
git commit -m "Introduce LimbScheme and Caps; route the radix-dependent host code through them

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 2: The 32-bit limb scheme: squaring kernel

Adds `LimbScheme::Limb32`, the generator for its squaring mode (a port of `coop32.rs`), capability gating, and the tests that prove it against the oracle and keep it out of the browser build.

**Files:**
- Create: `crates/bigmont-core/src/limb32.rs`, `crates/bigmont-core/tests/source_matrix.rs`, `crates/bigmont-core/tests/limb32.rs`, `crates/bigmont-core/examples/scheme_compare.rs`
- Modify: `crates/bigmont-core/src/{lib.rs, scheme.rs, kernel.rs, gpu.rs}`, `crates/bigmont-core/Cargo.toml` (naga dev-dependency)

**Interfaces:**
- Consumes: Task 1's `LimbScheme`, `Caps`, `KernelConfig::{for_bits_scheme, requires, try_wgsl_for}`, `Batch::from_values_cfg`, `gpu::run`; `kernel::SUBGROUP_HELPERS`; oracle `pack_u32_limbs`, `unpack_u32_limbs`, `neg_inv_u32`, `mont_square_chain_bits`.
- Produces: `LimbScheme::Limb32` (name `"32"`, `limb_bits` 32, `requires()` = `Caps { subgroups: false, int64: true }`, `normalised` always true, `limbs_for_bits = ceil((bits + 4) / 32)`); `limb32::wgsl(cfg: &KernelConfig, mode: Mode) -> String` where `Mode` is the existing `unrolled::Mode` (make it `pub(crate)` shared; Limb32 `Mode::Fermat` returns `Err` until Task 3); `KernelConfig::validate` accepts Limb32 only with `Variant::Unrolled` and `tpi` in 4, 8, 16, 32.

The kernel design is exactly the one in the module documentation of `coop32.rs`: two interleaved u64 chains per limb (`x*b + t + c1`, `n*m + lo + c2`), the carry out of a lane's last limb deferred one step into `pend` and added to the same lane's last limb at the next step with wrap detection (`s2 < q`), and a final resolve loop that adds each lane's pending carry into the next lane's limb 0 until a uniform `subgroupAny` says none remain. Do not redesign it; port it.

- [ ] **Step 1: Write the failing tests**

`tests/source_matrix.rs` (pure, no GPU; add `naga = { version = "30", features = ["wgsl-in"] }` to `[dev-dependencies]` after checking `cargo tree -p wgpu` for the naga major version actually in use; `git show worktree-agent-a190d857c66461b4e:crates/bigmont-core/Cargo.toml` shows what the experiment used):

```rust
use bigmont_core::kernel::{KernelConfig, Target, Variant};
use bigmont_core::scheme::{Caps, LimbScheme};

const PRELUDE: &str = "enable subgroups;\ndiagnostic(off, subgroup_uniformity);\n";

fn validate(src: &str, caps: Caps) -> Result<(), String> {
    let src = src.strip_prefix(PRELUDE).unwrap_or(src); // naga 30 rejects the browser-only directives
    let module = naga::front::wgsl::parse_str(src).map_err(|e| e.emit_to_string(src))?;
    let mut c = naga::valid::Capabilities::empty();
    if caps.subgroups {
        c |= naga::valid::Capabilities::SUBGROUP;
    }
    if caps.int64 {
        c |= naga::valid::Capabilities::SHADER_INT64;
    }
    naga::valid::Validator::new(naga::valid::ValidationFlags::all(), c).validate(&module).map(|_| ()).map_err(|e| format!("{e:?}"))
}

#[test]
fn every_scheme_and_target_that_generates_validates_with_exactly_that_targets_capabilities() {
    let native = Caps { subgroups: true, int64: true };
    let mut generated = 0;
    for &scheme in LimbScheme::ALL {
        for tpi in [4u32, 8, 16, 32] {
            for bits in [64u32, 254, 1019, 2029] {
                let cfg = KernelConfig::for_bits_scheme(scheme, bits, tpi, 64, Variant::Unrolled);
                for (target, caps) in [(Target::Native, native), (Target::Browser, Caps::BROWSER)] {
                    match cfg.try_wgsl_for(target) {
                        Ok(src) => {
                            validate(&src, caps).unwrap_or_else(|e| panic!("{cfg:?} {target:?}: {e}"));
                            generated += 1;
                        }
                        Err(_) => assert!(!caps.satisfies(cfg.requires()), "{cfg:?} {target:?} refused although the target can run it"),
                    }
                }
            }
        }
    }
    assert!(generated > 0);
}

#[test]
fn a_32_bit_kernel_is_never_generated_for_the_browser() {
    let cfg = KernelConfig::for_bits_scheme(LimbScheme::Limb32, 2029, 8, 64, Variant::Unrolled);
    let e = cfg.try_wgsl_for(Target::Browser).unwrap_err();
    assert!(e.contains("int64"), "{e}");
    assert!(cfg.try_wgsl_for(Target::Native).unwrap().contains("u64"));
    // and its source does need the int64 capability: validating it without that capability fails.
    let src = cfg.try_wgsl_for(Target::Native).unwrap();
    assert!(validate(&src, Caps { subgroups: true, int64: false }).is_err());
    // the 13-bit source never mentions u64.
    let c13 = KernelConfig::for_bits(2029, 8, 64, Variant::Unrolled).try_wgsl_for(Target::Browser).unwrap();
    assert!(!c13.contains("u64"));
}

#[test]
fn limb32_geometry() {
    let s = LimbScheme::Limb32;
    assert_eq!((s.limb_bits(), s.name()), (32, "32"));
    assert_eq!(s.limbs_for_bits(2029), 64);
    assert_eq!(s.limbs_for_bits(254), 9); // (254 + 4) / 32 = 8.06: R > 16n needs a ninth limb
    assert_eq!(LimbScheme::parse("32"), Some(s));
    assert_eq!(s.requires(), Caps { subgroups: false, int64: true });
    assert!(s.normalised(&[u32::MAX]));
    assert_eq!(KernelConfig::for_bits_scheme(s, 2029, 8, 64, Variant::Unrolled).limbs, 64);
    assert_eq!(KernelConfig::for_bits_scheme(s, 254, 8, 64, Variant::Unrolled).limbs, 16); // 9 rounded up to a multiple of 8
}
```

`tests/limb32.rs` (GPU, strict): port `git show c6922af:crates/bigmont-core/tests/coop32.rs` to the factory API: `KernelConfig::for_bits_scheme(LimbScheme::Limb32, bits, tpi, wg, Variant::Unrolled)`, `Batch::from_values_cfg(&vals, &cfg)`, `gpu::run(&gpu, cfg, &batch, squarings, repeats)`, decode with `cfg.scheme.unpack`, expected value `cfg.scheme.chain(n, x0, total, limbs)`, assertion `got < 2n` and `got % n == chain`. Keep the three tests (small sizes over TPI 4, 8, 16, 32; the 2,029-bit class; 510/1022-bit mixed workgroups) and the one-`Gpu`-per-test structure, skipping through `common::skip` when `!gpu.caps().satisfies(cfg.requires())`. Drop TPI 2 (not supported by the factory).

- [ ] **Step 2: Run to verify they fail**

Run: `cargo test --release -p bigmont-core --test source_matrix --test limb32`
Expected: FAIL to compile (`LimbScheme::Limb32` does not exist).

- [ ] **Step 3: Add the variant everywhere the compiler points**

Add `Limb32` to `LimbScheme` (`ALL`, `limb_bits` 32, `name` "32", `parse`, `requires`, `limbs_for_bits`, `r_bits`, `pack` = `pack_u32_limbs`, `unpack` = `unpack_u32_limbs`, `neg_inv` = `neg_inv_u32`, `normalised` = true, `chain` = `mont_square_chain_bits(.., 32 * limbs)`). Fix every `match` the compiler flags. In `KernelConfig::validate`, for Limb32 require `Variant::Unrolled` and `tpi` in {4, 8, 16, 32} with the same limbs/wg multiple rules as before. In `try_wgsl_for` / `try_fermat_wgsl_for`, for a scheme whose `requires()` the target does not offer (`Target::Browser` offers `Caps::BROWSER`; `Target::Native` offers everything) return `Err(format!("{scheme:?} needs {missing:?}, which the {target:?} target does not offer"))` with the word `int64` in it.

- [ ] **Step 4: Port the generator**

Create `limb32.rs` from `coop32.rs`'s `Coop32::wgsl`, `step` and `resolve` (everything except `Coop32`'s own struct, `Session` and `run`), parameterised by `&KernelConfig` (`tpi`, `wg`, `limbs_per_lane()`), with the same bindings as the 13-bit unrolled kernel (`params`, `n_limbs`, `ninv_buf`, `x_limbs`). `try_wgsl_for` for Limb32 calls it with `Mode::Squarings`; `Mode::Fermat` returns `Err("the 32-bit Fermat ladder is added in the next task")` (removed in Task 3). The Browser target never reaches the generator. Keep the module documentation of `coop32.rs` (the carry argument) in `limb32.rs`.

- [ ] **Step 5: Run the tests**

Run: `cargo test --release -p bigmont-core --test source_matrix --test limb32 --test scheme -- --test-threads=1`
Expected: PASS.

- [ ] **Step 6: Mutation check (record the result in the commit message)**

Temporarily delete ` + (a >> 32u)` from the `pend = ...` line in `limb32.rs` and re-run `--test limb32`: it must FAIL on many instances. Restore the line and confirm the tests pass again. Do the same for ` + wrap`: this one only triggers on rare wraps, so it may not fail; say so in the report instead of claiming it does.

- [ ] **Step 7: Port the benchmark and run the whole suite**

Create `examples/scheme_compare.rs` from `git show c6922af:crates/bigmont-core/examples/coop_bench.rs`, replacing `Coop32`/`Session` with `KernelConfig::for_bits_scheme(scheme, ...)` and `gpu::run` for both schemes (it needs the same slope method; keep the `--bits --count --runs --tpis --wgs` flags and add `--schemes 13,32`). Then `cargo test --release --workspace --no-fail-fast -- --test-threads=1` must be all green, and `cargo build --release --workspace --examples` must compile.

- [ ] **Step 8: Commit**

```bash
git add crates
git commit -m "Add the 32-bit limb scheme (u64 products): squaring kernel, capability gating, naga source matrix

<mutation-check result here>

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Fermat ladder and runner for the 32-bit scheme

**Files:**
- Modify: `crates/bigmont-core/src/{limb32.rs, runner.rs}`, `crates/bigmont-core/tests/{fermat.rs, sizes.rs}`

**Interfaces:**
- Consumes: Tasks 1-2.
- Produces: `try_fermat_wgsl_for` works for Limb32 (native only); `Runner::new(gpu, cfg, bits, capacity)` accepts a Limb32 config and `Runner::fermat` returns correct base-2 Fermat verdicts for it.

The Fermat mode has the same bindings as the 13-bit one (`@binding(4) e_words`, `@binding(5) fp`: `fp.x` first iteration, `fp.y` nbits, `fp.z` words per instance; `params.y` = iterations in this dispatch), reads exponent bit `gi = fp.y - 2 - (fp.x + s)` of `e_words[ic * fp.z + (gi >> 5)]`, and after every squaring does the branch-free conditional doubling. With canonical 32-bit limbs the doubling needs no carry ripple: limb `c` of `2x` is `(x_c << 1) | (x_{c-1} >> 31)`, and limb 0 of a lane takes its low bit from the previous lane's last limb. R > 16n keeps `2x < 4n < R`, so nothing leaves the top limb. All lanes execute the shuffle (uniform control flow); the bit only selects the result.

```wgsl
// in the squaring loop, after `x{c} = t{c}`:
let gi = fp.y - 2u - (fp.x + s);
let ebit = (e_words[ic * fp.z + (gi >> 5u)] >> (gi & 31u)) & 1u;
var prev = up1(x{LPL-1} >> 31u, lane);      // bit shifted in from the lane below (0 for lane 0)
let d0 = (x0 << 1u) | prev;  prev = x0 >> 31u;
let d1 = (x1 << 1u) | prev;  prev = x1 >> 31u;
// ... one pair per limb ...
x0 = select(x0, d0, ebit == 1u);
// ... one select per limb ...
```

- [ ] **Step 1: Write the failing tests**

Parameterise `tests/fermat.rs` over the scheme without copy-pasting bodies: `verify`/`verify_on` take a `LimbScheme`, build the config with `KernelConfig::for_bits_scheme(scheme, bits, tpi, wg, Variant::Unrolled)`, and every existing `#[test]` is run for each `scheme in LimbScheme::ALL` (loop inside the test, one `Gpu` per test, skipping a scheme whose `requires()` the device lacks through `common::skip`). Add Limb32-specific expectations the loop already covers: the Mersenne table (including the 2^2029-1 pseudoprime), exhaustive odd `n` in 3..4095 at TPI 4 and 8, the edge lengths 2, 3, 31, 32, 33, 62, 63, 64, 65 bits and the limb-class boundary lengths `32*k - 4`, `32*k - 3` for k = 2, 3, 8, 64. In `tests/sizes.rs` add a Limb32 variant of the existing sizes test that runs the squaring chain at limb counts 4, 5, 8, 9, 16, 33, 64 (use `bits = r_bits - 3`).

- [ ] **Step 2: Run to verify they fail**

Run: `cargo test --release -p bigmont-core --test fermat --test sizes -- --test-threads=1`
Expected: FAIL with the "added in the next task" error for Limb32.

- [ ] **Step 3: Implement the Fermat mode**

In `limb32.rs` implement `Mode::Fermat`: add the two bindings in the header, the `gi`/`ebit` lines and the doubling above after the end-of-squaring copy; remove the staging `Err`. In `runner.rs` nothing radix-specific should remain after Task 1; fix whatever Task 1 left (the runner's Variant check, error texts that say "13"). `Runner::new` for Limb32 uses `limbs = cfg.limbs` (already a multiple of tpi).

- [ ] **Step 4: Run the tests, then the mutation check**

Run: `cargo test --release -p bigmont-core --test fermat --test sizes -- --test-threads=1` (PASS). Then delete the `up1(...)` term (use `0u` instead) from the doubling and confirm `--test fermat` FAILS for Limb32 at TPI >= 4 (the carry between lanes is exercised whenever a doubling happens at a lane boundary); restore. Report both results.

- [ ] **Step 5: Whole suite and commit**

Run `cargo test --release --workspace --no-fail-fast -- --test-threads=1` (all green). Commit with message `Add the 32-bit Fermat ladder and run it through the runner` plus the trailer.

---

### Task 4: Tuner, cache, engine and CLI choose the scheme by measurement

**Files:**
- Modify: `crates/bigmont-cli/src/{tune.rs, engine.rs, main.rs}`
- Test: `crates/bigmont-cli/tests/{tune.rs, engine.rs, cli.rs}`

**Interfaces:**
- Consumes: Tasks 1-3.
- Produces:

```rust
// tune.rs
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SchemeChoice { #[default] Auto, Only(LimbScheme) }
impl SchemeChoice {
    pub fn allows(self, s: LimbScheme) -> bool;
    pub fn parse(s: &str) -> Result<SchemeChoice, String>;   // "auto" | "13" | "32"
}
pub fn candidates(gpu: &Gpu, bits: u32, choice: SchemeChoice) -> Vec<KernelConfig>;   // adds the scheme dimension
pub fn autotune(gpu: &Gpu, bits: u32, cache: Option<&Path>, retune: bool, choice: SchemeChoice) -> Result<Tuned, String>;
// engine.rs: EngineOptions gains `pub scheme: tune::SchemeChoice` (default Auto)
// main.rs: `--limbs auto|13|32` on `run`, `canary` and `tune`
```

Rules fixed by this plan:
- `candidates` offers, for every scheme `choice` allows, every (tpi, wg) the adapter supports and for which `gpu.caps().satisfies(cfg.requires())`. Auto on a device without `SHADER_INT64` therefore offers only 13-bit kernels, with no vendor test anywhere.
- The tuning class stays `limb_class(bits)` (13-bit limb count, at least 4) for every scheme, and `class_bits(limbs)` the representative length; the winning configuration is stored as a shape (scheme, tpi, wg) and the engine applies it to each exact bit length with `KernelConfig::retarget(bits)`. Unlike 13-bit, a 32-bit kernel's limb count can differ between lengths of one class (for example 244..256 bits needs 8 limbs up to 252 bits and 9 from 253), so the self-test at the class's representative length validates the kernel shape, and `tests/sizes.rs` (Task 3) is what validates the generator at other limb counts.
- Cache value text: a 13-bit winner is still `unrolled <tpi> <wg>|<rate>` (existing caches stay valid and parse as Limb13); a 32-bit winner is `unrolled32 <tpi> <wg>|<rate>`. An unknown first word makes the line a miss. Cache key: unchanged for `Auto`; for `Only(s)` the token `only-<name>` is inserted before the class field (`name|backend|driver|only-13|<class>`), so a forced run never overwrites or deletes an Auto entry (the old key `...|<class>` is not a prefix of the new one).
- A cached shape is used only if `candidates(gpu, bits, choice)` still contains it and it passes its self-test, as today.
- `tune` subcommand output gains a `scheme` column.

- [ ] **Step 1: Write the failing tests**

Pure tests in `crates/bigmont-cli/tests/tune.rs` (no GPU): `SchemeChoice::parse` accepts `auto`, `13`, `32` and rejects `64`; `allows`; cache value round trip for both schemes through `write_cache`/the parser the tests can reach (expose `pub fn parse_value(v: &str) -> Option<(LimbScheme, u32, u32, f64)>` and `pub fn cache_value(...)`/`cache_key(...)` for testing, `#[doc(hidden)]` is fine); an old-format line `unrolled 8 64|123.5` parses as `(Limb13, 8, 64, 123.5)`; `unrolled32 8 64|9.5` as Limb32; `unrolled64 ...` and `rolled ...` do not parse; the forced key differs from the Auto key and `write_cache` of the Auto key keeps a forced entry (use a temp file). GPU tests: `candidates(Auto)` contains a Limb32 config iff `gpu.has_int64`; `candidates(Only(Limb13))` contains none; `autotune(..., Only(Limb32))` returns a Limb32 config whose self-test passed (skip loudly when the device lacks int64); `autotune(Auto)` returns a config that appears in `candidates(Auto)` and caches it with its scheme (a second call reports `from_cache` and the same scheme). In `tests/engine.rs`: run the engine with `scheme: Only(Limb13)`, `Only(Limb32)` and `Auto` over mixed bit lengths (40, 100, 254, 255, 300, 2029) and compare every verdict with the oracle, the same way the existing `gpu_backend_matches_the_oracle_and_buckets_by_bit_length` test does; assert `gpu_tests` is non-zero for each. In `tests/cli.rs`: `--limbs 32` runs a small candidate file and writes the expected verdict file; `--limbs 7` exits non-zero with a message naming the allowed values.

- [ ] **Step 2: Run to verify they fail**

Run: `cargo test --release -p bigmont-cli --no-fail-fast -- --test-threads=1`
Expected: FAIL to compile (new signatures).

- [ ] **Step 3: Implement**

Implement `SchemeChoice`; thread `choice` through `candidates`, `autotune`, `cache_key`, `cache_value`/`parse_value`, `self_test` (already scheme-generic from Task 1: it reads `cfg.scheme`); add `EngineOptions.scheme`; in `engine.rs` replace `KernelConfig::for_bits(bits, tuned.tpi, tuned.wg, Variant::Unrolled)` with `tuned.retarget(bits)` and pass `self.opts.scheme` to `autotune`; add the CLI flag to `run`, `canary` and `tune` (default `auto`) and print the scheme in the `tune` table and in the engine's `tune ...` stderr lines (the config's `Debug` already includes `scheme`). Update the existing tests only where a signature changed (`autotune(..., SchemeChoice::Auto)`).

- [ ] **Step 4: Run the CLI tests and the whole suite**

Run: `cargo test --release -p bigmont-cli --no-fail-fast -- --test-threads=1` then the workspace suite. All green.

- [ ] **Step 5: Commit**

Message: `Tune the limb scheme per adapter: candidates, cache, engine option and --limbs flag` plus the trailer.

---

### Task 5: Measurement and record (controller, no subagent)

**Files:**
- Create: `docs/superpowers/results/2026-11-25-limb-factory-validation.md` and raw logs under `docs/superpowers/results/phase1/factory/`
- Modify: `docs/superpowers/phase1-backlog.md`

- [ ] **Step 1: Full suites.** `cargo test --release --workspace --no-fail-fast -- --test-threads=1` on the desktop (RX 6700 XT, Vulkan and default backends) and on the laptop (`BIGMONT_ADAPTER=4070 BIGMONT_BACKEND=Vulkan`; also the 780M and llvmpipe adapters for the tuner tests). Save the logs.
- [ ] **Step 2: What the tuner picks.** On each adapter run `bigmont tune --bits 2029` (fresh cache) and record the winning scheme, TPI and workgroup with all candidates' rates. Expected from the cooperative benchmark: 32-bit on the 4070, 13-bit on the 6700 XT. If the tuner picks otherwise, say so and look at why before writing anything down.
- [ ] **Step 3: Engine throughput.** With `make_candidates --bits 2029` files, run `bigmont run` with `--limbs 13`, `--limbs 32` and `--limbs auto`, five repeats each, on the 4070 and the 6700 XT; compare with the 36,053 tests/s (4070) and 33,842 tests/s (6700 XT) recorded in the Phase 1 validation document, and with Beacon's 132,096 tests/s on the 4070.
- [ ] **Step 4: Record.** Write the validation document (what was measured, what was not, the limitations in the Global Constraints that remain) and update the backlog (Intel and Apple untested; Tint/Dawn compile of the 13-bit source untested; WGSL has no hardware carry; AMD instruction-count follow-up).
- [ ] **Step 5: Commit** (message `Record the limb-factory validation on the RTX 4070 and RX 6700 XT` plus the trailer).

---

## Self-review

**Spec coverage:** limb width as a tuned parameter (Tasks 1, 4); both squaring and ladder kernels for 32-bit limbs (Tasks 2, 3); native versus web capability separation and its compile-time checks (Task 1 `Caps`, Task 2 naga matrix and the browser refusal); no vendor logic (Global Constraints, Task 4 `candidates`); cache compatibility (Task 4); measurement against the baselines (Task 5).

**Placeholders:** Task 2 Step 4 and Task 3 Step 3 refer to `coop32.rs` at commit `c6922af` for the generator body instead of repeating about 150 lines, because that file is the tested reference and is read-only input for the implementer; the stub `Err` in Task 2 for the Fermat mode is a staging message removed in Task 3.

**Type consistency:** `LimbScheme`, `Caps`, `KernelConfig::{for_bits_scheme, requires, retarget, r_bits, try_wgsl_for, try_fermat_wgsl_for}`, `Gpu::{has_int64, caps}`, `Batch::from_values_cfg`, `SchemeChoice`, `candidates(.., choice)` and `autotune(.., choice)` are spelled the same in every task that names them.
