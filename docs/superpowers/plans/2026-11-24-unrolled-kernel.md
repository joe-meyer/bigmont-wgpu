# Unrolled cooperative kernel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a second code path for the cooperative kernels (TPI 8, 16, 32) that generates fully unrolled straight-line WGSL with named scalar variables, with the one-limb shift folded into the accumulate, and measure whether it is faster than the current rolled kernel.

**Architecture:** A new `Variant` (`Rolled` = today's template, `Unrolled` = new generator in `unrolled.rs`) selected through `KernelConfig`. Same bindings, same entry point `main`, same contract as the rolled kernel, so `run()`, the oracle tests and the sweep work unchanged. The motivation is evidence from the AMD driver (`examples/pipeline_stats.rs`): the rolled kernel's arithmetic loops are not unrolled (two multiply-add instructions in the whole program, about 20 indirect register moves, 57 to 61 branches).

**Tech Stack:** Rust 1.100, wgpu 30.0.2, WGSL (subgroup built-ins without an `enable` line, as today), naga 30.

**Spec / background:** `docs/superpowers/specs/2026-11-24-bigmont-wgpu-design.md`; ideas and measurements in `docs/superpowers/results/2026-11-24-cgbn-lessons.md`; existing rolled kernel in `crates/bigmont-core/src/mont.wgsl` and `kernel.rs`.

## Global Constraints

- Kernels use only browser-implementable WGSL features: 32-bit integers only, no `SHADER_INT64`, no push constants. The subgroup built-ins are allowed exactly as the rolled kernel uses them (no `enable subgroups;` line is emitted because naga 30 rejects it; keep the existing `TODO(browser)` note valid).
- Number format unchanged: `LIMBS = 160` limbs of 13 bits, little-endian, one limb per `u32`; Montgomery radix R = 2^2080; test moduli odd, 2029 bits.
- Correctness before speed: no throughput number is reported for a configuration that has not passed the oracle comparison; assertions are never weakened to make a kernel pass.
- The tests are strict by default (a skip panics unless `BIGMONT_ALLOW_SKIP=1`).
- Every GPU dispatch stays under about 0.5 s (Windows GPU timeout).
- The rolled kernel and all its existing tests and sweep behaviour must keep working unchanged (it is the comparison baseline).
- No code is copied from CGBN or Beacon; the generator is written from the algorithm below.
- Do not download anything beyond cargo's normal crate fetches. Do not touch `N:\Programming\gapsieve`.
- Commit messages end with the line `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.
- Work in `N:\Programming\bigmont-wgpu` (git, branch `main`); use absolute paths and forward-slash manifest paths for cargo.

## The algorithm the unrolled kernel must implement

Same arithmetic contract as the rolled kernel: per instance, `x <- x^2 * R^-1 mod n` (result in [0, 2n), limbs fully normalised below 2^13), repeated `params.y` times. Notation: `TPI` lanes per number, `LPL = 160 / TPI` limbs per lane; lane `j` holds limbs `j*LPL .. j*LPL+LPL-1` of `x`, `n` and the accumulator `t`, as named scalars `x0..x{LPL-1}`, `n0..`, `t0..` (no arrays, no loops over limbs). `MASK = 0x1fff`, `B = 13`, `PERIOD = 24`. `base`, `lane`, `bcast`, `down1`, `up1`, `any_nonzero` are exactly as in the rolled kernel (reuse `SUBGROUP_HELPERS` from `kernel.rs`).

Hoisted once per instance (outside the squaring loop): `ninv`, `is0 = select(0u, 0xffffffffu, lane == 0u)`, `N0 = bcast(n0, 0, base)` (lane 0's limb 0 of `n`), `NN0 = down1(n0, lane)` (the next lane's limb 0 of `n`; 0 for the last lane).

Per squaring: `X0 = bcast(x0, 0, base)`, `XN0 = down1(x0, lane)`, all `t_c = 0`. Then:

```
for owner in 0 .. TPI            // rolled loop, uniform
    // body unrolled textually for w in 0 .. LPL (limb x_w of the owning lane):
    b     = bcast(x_w, owner, base)               // the next limb of the multiplier
    T0    = bcast(t0, 0, base)                    // accumulator limb 0 of lane 0
    TN    = down1(t0, lane)                       // next lane's accumulator limb 0 (0 for the last lane)
    p     = T0 + (X0 & 0xffffff) * (b & 0xffffff)             // lane 0's limb 0 after adding x*b
    m     = (p * ninv) & MASK                                 // Montgomery factor
    carry0 = (p + (N0 & 0xffffff) * m) >> B                   // exact: low 13 bits are zero
    // new accumulator = old accumulator shifted down one limb, plus x*b and n*m shifted the same way,
    // written in place from c = 0 upward so each line reads the old value above it:
    t0 = t1 + (x1 & 0xffffff)*(b & 0xffffff) + (n1 & 0xffffff)*m + (carry0 & is0)
    t1 = t2 + (x2 & ...)*(b & ...)           + (n2 & ...)*m
    ...
    t{LPL-2} = t{LPL-1} + (x{LPL-1} & ...)*(b & ...) + (n{LPL-1} & ...)*m
    t{LPL-1} = TN + (XN0 & 0xffffff)*(b & 0xffffff) + (NN0 & 0xffffff)*m
    if (owner % K) == K - 1 then light_pass     // K = max(1, PERIOD / LPL), so K*LPL <= 24 iterations between passes
full_normalize                                   // loop over rounds: light passes until no limb is >= 2^13
x_c = t_c                                        // for the next squaring
```

`light_pass` (unrolled scalars): `c_in = up1(t{LPL-1} >> B, lane)` computed first (before any limb is modified); then for `c = 0..LPL-1` in order: `v = t_c; t_c = (v & MASK) + c_in; c_in = v >> B`. `full_normalize`: for up to `L + 2` rounds: `big = (t0 >> B) | (t1 >> B) | ...`; break if `!any_nonzero(big)`; else `light_pass`. The `(v & 0xffffff)` style masks on multiply operands are deliberate (they steer AMD's compiler to the 24-bit multiply-add; keep them).

Why the bounds hold (the implementer should keep this reasoning in a comment): limbs entering an iteration are below 2^13 (x, n) so every product is below 2^26; each accumulator word gains at most two products (below 2^27) per iteration plus `carry0` (below 2^19) on lane 0's limb 0; a light pass leaves every limb below 2^19 + 2^13; so after at most 24 iterations without a pass a limb is below about 2^19 + 2^13 + 24 * (2^27 + 2^19) ≈ 3.23e9, under 2^32. `p` and `p + N0*m` are limb-0 values after one and two products and are bounded the same way. The value stays below 2n because R > 4n (as in the rolled kernel), so the top lane's discarded carry is zero.

The generator is only valid for `TPI` in {8, 16, 32} (`LPL` 20, 10, 5). `TPI = 1` stays rolled-only.

---

### Task 1: Unrolled generator, variant plumbing, oracle tests

**Files:**
- Modify: `crates/bigmont-core/src/kernel.rs` (add `Variant`, a `variant` field on `KernelConfig`, dispatch in `wgsl()`, validation)
- Create: `crates/bigmont-core/src/unrolled.rs` (the generator)
- Modify: `crates/bigmont-core/src/lib.rs` (add the module)
- Modify: `crates/bigmont-core/tests/chain.rs` (new unrolled tests; existing tests set `variant: Variant::Rolled`)
- Modify: `crates/bigmont-core/examples/spike.rs`, `crates/bigmont-core/examples/pipeline_stats.rs` (only as needed so they still compile: add `variant: Variant::Rolled` to their `KernelConfig` literals; the flags come in Task 2)

**Interfaces:**
- Consumes: `bigmont_oracle::{LIMBS, LIMB_BITS, LIMB_MASK}`, the existing `SUBGROUP_HELPERS` and template placeholders in `kernel.rs`, and the binding layout of `mont.wgsl` (binding 0 `var<uniform> params: vec4<u32>` with x = instance count, y = squarings; binding 1 `n_limbs`, 2 `ninv_buf`, 3 `x_limbs` read_write; entry point `main`, `@workgroup_size(WG)`, instance mapping `inst = wid.x * (WG / TPI) + li / TPI`, dead lanes clamp to `ic = min(inst, params.x - 1)` and skip only the final store).
- Produces: `pub enum Variant { Rolled, Unrolled }` (`Clone, Copy, Debug, PartialEq, Eq`); `KernelConfig { pub tpi: u32, pub wg: u32, pub variant: Variant }`; `KernelConfig::validate()` rejects `Unrolled` with `tpi == 1`; `KernelConfig::wgsl()` returns the unrolled source for `Variant::Unrolled`; `pub(crate) fn unrolled::wgsl(cfg: &KernelConfig) -> String`.

- [ ] **Step 1: Write the failing tests.** In `tests/chain.rs`, make the shared `check` helper unchanged in assertions, update the existing `KernelConfig { tpi, wg }` literals with `variant: Variant::Rolled`, and add `tpi8_unrolled_matches_oracle`, `tpi16_unrolled_matches_oracle`, `tpi32_unrolled_matches_oracle` (all wg 64) and `tpi32_unrolled_other_workgroup_sizes_match_oracle` (wg 128 and 256), each calling `check(KernelConfig { tpi, wg, variant: Variant::Unrolled })`. The batch (133 instances plus the two carry-stress moduli, 3 chained squarings) stays as is.
- [ ] **Step 2: Run them to verify they fail (compile failure on `Variant`).** `cargo test -p bigmont-core --manifest-path N:/Programming/bigmont-wgpu/Cargo.toml --test chain --no-run`.
- [ ] **Step 3: Implement `Variant`, the field, validation, dispatch and `unrolled.rs`** following the algorithm above. The generator builds the WGSL as a `String` (a `Vec<String>` of lines joined is fine); it must emit the same `const`s (`TPI`, `LPL`, `L`, `WG`, `B`, `MASK`), the same four bindings and the same `main` signature as `mont.wgsl` (the per-config text substitution for `__SGPARAM__`/`__SGLET__` works the same way), and the subgroup helper functions from `SUBGROUP_HELPERS`. Add a doc comment on `unrolled::wgsl` that records the bound argument above.
- [ ] **Step 4: Run the tests on the real GPU, strict, on Vulkan.** `BIGMONT_ADAPTER=6700 BIGMONT_BACKEND=vulkan cargo test --release -p bigmont-core --manifest-path N:/Programming/bigmont-wgpu/Cargo.toml -- --nocapture --test-threads=1`. All rolled and unrolled tests must pass with no skips. If an unrolled test fails, debug in this order: (1) one squaring, one instance; (2) compare the accumulator against a CPU model of the same algorithm; (3) check the carry/shift bookkeeping (lane 0's `carry0` goes into limb 0 only; the last lane's top limb reads zeros from `down1`). Do not weaken assertions.
- [ ] **Step 5: Run the oracle crate tests** (`cargo test --release -p bigmont-oracle ...`) to make sure nothing regressed, then commit: `Add unrolled cooperative kernel generator (TPI 8/16/32) and variant plumbing`.

---

### Task 2: Measure it

**Files:**
- Modify: `crates/bigmont-core/examples/spike.rs` (add `--variant rolled|unrolled`, default `rolled`; the variant is printed in the adapter header and in each row so raw files are self-describing)
- Modify: `crates/bigmont-core/examples/pipeline_stats.rs` (optional fourth argument `unrolled`; default `rolled`)
- Create: `docs/superpowers/results/2026-11-24-unrolled-kernel-results.md`
- Create: raw outputs under `docs/superpowers/results/other-adapters/unrolled/`

**Interfaces:**
- Consumes: Task 1's `Variant` and `KernelConfig`.
- Produces: the flags above and the results document.

- [ ] **Step 1: Add the flags** (`--variant` in `spike.rs`; the `unrolled` argument in `pipeline_stats.rs`, which must skip `tpi 1` for it). `--variant unrolled` with `--tpi 1` is skipped with a printed reason, not an error. Build both examples.
- [ ] **Step 2: Compiler evidence on this machine.** `pipeline_stats` for TPI 8, 16, 32 (wg 64) with `unrolled`, device substring `6700`: report VGPRs, scratch bytes, and from the saved ISA the counts of `v_mad_u32_u24`, `v_movrel`, `ds_bpermute`/`v_readlane`/`ds_swizzle` and `s_cbranch`. Compare with the rolled numbers in `docs/superpowers/results/2026-11-24-cgbn-lessons.md` section 2 (rolled: 2 `v_mad_u32_u24`, about 20 `v_movrel`, 57 to 61 `s_cbranch`).
- [ ] **Step 3: Sweeps on this machine (RX 6700 XT).** Run `spike --variant unrolled --max-count 131070` on Vulkan (`BIGMONT_ADAPTER=6700 BIGMONT_BACKEND=vulkan`) and on Dx12 (`BIGMONT_BACKEND=dx12` with `C:\Program Files (x86)\Windows Kits\10\Redist\D3D\x64` prepended to `PATH` for that run only). Save raw outputs to `docs/superpowers/results/other-adapters/unrolled/raw-desktop-vulkan.txt` and `raw-desktop-dx12-dxc.txt`. Any `VALIDATION FAILED` or `REJECTED` row is a finding to report, not to hide.
- [ ] **Step 4: Write the results document.** Table of unrolled versus rolled for each TPI (best wg row, squarings/s at long dispatch, projected tests/s = squarings/s / 2029) using the rolled figures from `docs/superpowers/results/other-adapters/amortised/raw-desktop-vulkan.txt` and `raw-desktop-dx12-dxc.txt`; the ISA comparison from Step 2; an honest statement of what is measured and what is not (laptop and host2 GPUs are not run by this task; the controller does those). No go/no-go verdict.
- [ ] **Step 5: Run the whole test suite once more (strict, Vulkan) and commit** the examples, the results document and the raw files: `Measure the unrolled kernel against the rolled kernel on the RX 6700 XT`.

---

### Task 3: Verify the timed output and reconcile the ISA story (from the skeptical review of Task 2)

**Files:**
- Modify: `crates/bigmont-core/examples/spike.rs` (verify the timed dispatch's output against the oracle)
- Modify: `crates/bigmont-core/examples/pipeline_stats.rs` (match wgpu's subgroup-size mode)
- Modify: `docs/superpowers/results/2026-11-24-unrolled-kernel-results.md` (corrections below)
- Create: `docs/superpowers/results/other-adapters/unrolled/isa/` (the saved ISA text files, committed)
- Create: new raw sweep outputs under `docs/superpowers/results/other-adapters/unrolled/` (names ending `-verified`)

**Interfaces:**
- Consumes: Task 1 and Task 2 code (`Variant`, `KernelConfig`, `run`, `Batch::random_values`, `Batch::from_values`, `mont_square_chain`, `unpack_limbs`).
- Produces: a sweep that never prints a speed for a row whose timed output failed the oracle comparison.

- [ ] **Step 1: Oracle-verify the exact timed dispatches.** In `bench()` the batch values come from `Batch::random_values(count, BITS, 99)`; keep those values. `run(gpu, cfg, &batch, s, r)` uploads the original batch fresh and chains `r` dispatches of `s` squarings each, so after `run(.., s, r)` every instance has had `s * r` squarings. For the FINAL timed long run (the one whose time is reported as `long ms`), compare the returned `x` with `mont_square_chain(n, x0, s * r)` for a sample of at least 24 instances: the first 4, the last 4 (live ones, i.e. index below `count`), and 16 spread across the batch, including indices in the first and last workgroups. Compare modulo n like `validate` does (`unpack_limbs(limbs) % n`), and also assert every limb is below 2^13 and the value is below 2n. A mismatch returns the `Failure::Wrong` path (no speed printed, exit status nonzero). Run it for both variants (rolled too) so the baseline is verified the same way. Print a per-row `verified` marker (e.g. `ok, verified 24`).
- [ ] **Step 2: Which compiled code is actually timed?** Find out from the wgpu-hal Vulkan source (`~/.cargo/registry/src/*/wgpu-hal-30.0.2/src/vulkan/`) what wgpu does for compute pipelines with subgroup operations: does it set a required subgroup size, or the `REQUIRE_FULL_SUBGROUPS` / `ALLOW_VARYING_SUBGROUP_SIZE` stage flags (VK_EXT_subgroup_size_control)? Make `pipeline_stats.rs` mirror it (enable the extension and set the same flags and, if wgpu pins a size, the same required size) so the saved ISA is for the same compilation mode. Then redo the ISA counts for unrolled TPI 8, 16, 32 (wg 64) on the RX 6700 XT: `v_mad_u32_u24`, `v_readlane`, `v_readfirstlane`, `s_cbranch`, VGPRs, scratch, and the wave size the compiler reports, and count how many instructions one loop iteration of the owner loop contains. Say plainly whether the earlier "readlane waterfall" reading still holds in the matching mode. Commit the ISA text files you analyse under `docs/superpowers/results/other-adapters/unrolled/isa/` (text only, named by variant, TPI, wg, mode).
- [ ] **Step 3: Re-run the headline sweeps with verification** on this machine: Vulkan and Dx12 (with `C:\Program Files (x86)\Windows Kits\10\Redist\D3D\x64` prepended to PATH for that run only), `--variant unrolled --max-count 131070` and `--variant rolled --max-count 131070`, saved as `raw-desktop-vulkan-verified.txt`, `raw-desktop-dx12-dxc-verified.txt` (unrolled) and `raw-desktop-vulkan-rolled-verified.txt` (rolled). If any row fails verification that is the most important finding of the whole experiment: stop and report it with the details before doing anything else.
- [ ] **Step 4: Corrections to the results document** (do not change measured numbers; fix interpretation): (a) the ISA/peak interpretation is "unreconciled with the measured rate": state the reviewer's cycle-budget argument (about 50 straight-line vector ops per iteration plus two waterfall loops per iteration, against 82.1 M squarings/s x 160 iterations x 8 lanes) and what Step 2 found; remove or soften "60% of peak" and "limited headroom" accordingly; the nominal peak is an unverified spec-sheet estimate. (b) add one sentence that the old `mul_bench` figure of 2,861 G chain-steps/s was never a hardware ceiling (even its add-only kernel reached only about 2.4 T/s) and is superseded; the new kernel's 4.2 T mad/s exceeds it. (c) footnote that each lane also recomputes `p` and `carry0`, so the issued multiply-adds are about 53,760 per squaring at TPI 8 against the 51,200 counted. (d) the rolled baseline is fair: its slope column is only 2 to 6% above its long-dispatch rate and it did not improve at 16 squarings per dispatch. (e) say that timings are wall-clock around submit and a blocking poll, best of 3, no GPU timestamps, single run per row. (f) point to the committed ISA files.
- [ ] **Step 5: Run the full strict test suite on Vulkan once more and commit** (`Verify the timed unrolled output against the oracle and reconcile the ISA reading`). Trailer exactly `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`; nothing under N:\Programming\gapsieve; no downloads.
