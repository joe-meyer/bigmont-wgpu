# Tuning pass: cheaper Montgomery factor and TPI 4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Two browser-compatible (32-bit WGSL, subgroups) improvements to the unrolled cooperative kernel: (a) compute the Montgomery factor with operands that fit a 24-bit multiply instead of a full 32-bit low multiply (quarter rate on AMD); (b) support 4 lanes per number (40 limbs per lane).

**Architecture:** Changes confined to `crates/bigmont-core/src/unrolled.rs` (generator) and the validation list in `kernel.rs`. The rolled kernel (`mont.wgsl`) is the unchanged baseline except that TPI 4 is also allowed there if it works unchanged.

**Spec / background:** `docs/superpowers/plans/2026-11-24-unrolled-kernel.md` (algorithm, bounds, constraints; read everything above its "### Task 1:"), `docs/superpowers/results/2026-11-24-unrolled-cross-vendor.md` (baseline numbers), `docs/superpowers/results/2026-11-24-browser-test.md` (browser constraints).

## Global Constraints

- All constraints of `docs/superpowers/plans/2026-11-24-unrolled-kernel.md` apply: 32-bit integers only in WGSL, no `SHADER_INT64`, no push constants; subgroup built-ins as today; 160 limbs of 13 bits, R = 2^2080; strict tests; never weaken assertions; no CGBN/Beacon code; trailer `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`; nothing under `N:\Programming\gapsieve`; no downloads.
- The kernel must stay valid for browsers: do not use any native-only feature. (The browser page prepends `enable subgroups;` and `diagnostic(off, subgroup_uniformity);`; keep the generated source valid with those two lines in front.)
- The carry-bound argument must still hold: at most 24 iterations between light passes (a limb is below 2^19 + 2^13 after a pass; each iteration adds below 2^27 plus a carry below 2^19 to limb 0).
- Dispatches under 0.5 s planned. Use `N:/Programming/bigmont-wgpu/Cargo.toml` with cargo.

---

### Task 1: Cheaper Montgomery factor and TPI 4

**Files:**
- Modify: `crates/bigmont-core/src/unrolled.rs`, `crates/bigmont-core/src/kernel.rs`
- Modify: `crates/bigmont-core/tests/chain.rs` (new tests), `crates/bigmont-core/examples/spike.rs` (default width list includes 4)
- Create: `docs/superpowers/results/2026-11-24-tuning-pass-results.md`, raw outputs under `docs/superpowers/results/other-adapters/tuning/`

**Interfaces:**
- Consumes: the existing generator and `KernelConfig { tpi, wg, variant }`.
- Produces: `KernelConfig::validate()` accepts `tpi` 4 for both variants; `unrolled::wgsl` supports `tpi` in {4, 8, 16, 32}; the unrolled kernel computes `m = ((p & 0x1fff) * ninv) & MASK` (both factors below 2^13, so the product fits 24 bits).

- [ ] **Step 1 (a): Montgomery factor.** In `unrolled.rs` change the factor computation from `m = (p * ninv) & MASK` to `m = ((p & MASK) * ninv) & MASK`. Justification to put in the comment: only the low 13 bits of `p * ninv` matter, and `(p & MASK) * ninv` is below 2^26, so it is exact and can use the 24-bit multiply. Run the strict tests; expect no change in results.
- [ ] **Step 2 (b): TPI 4.** `LPL` becomes 40 at TPI 4, so one owner iteration has 40 steps, more than 24 iterations between light passes. Change the pass placement so a pass is emitted INSIDE the unrolled body every 20 steps (after step 19 and after step 39 for LPL 40; after step 19 for LPL 20; for LPL 10 and 5 keep the existing runtime `(owner % K) == K - 1` placement with K = 2 and 4, so passes still come every 20 iterations). Keep 20 iterations between passes for every width. Update the generator's runtime asserts and the doc comment (the bound argument), `KernelConfig::validate()` (allow 4), and `needs_subgroups`/`check_subgroups` logic if it hard-codes widths. Add oracle tests `tpi4_unrolled_matches_oracle` (wg 64; also wg 128 and 256 in one test) and a rolled `tpi4_matches_oracle` only if the rolled kernel works at TPI 4 unchanged (if it does not, say so and leave rolled TPI 4 rejected by `validate`, with a test). Include TPI 4 in the long-chain test. The `spike` default width list becomes `[1, 4, 8, 16, 32]`.
- [ ] **Step 3: strict tests.** `BIGMONT_ADAPTER=6700 BIGMONT_BACKEND=vulkan cargo test --release -p bigmont-core --manifest-path N:/Programming/bigmont-wgpu/Cargo.toml -- --nocapture --test-threads=1` and the oracle crate tests; all pass, no skips. Fix the kernel, never the tests.
- [ ] **Step 4: measure on this machine (RX 6700 XT).** `spike --variant unrolled --tpi 4,8,16,32 --max-count 131070` on Vulkan (`BIGMONT_ADAPTER=6700 BIGMONT_BACKEND=vulkan`) and on Dx12 (prepend `C:\Program Files (x86)\Windows Kits\10\Redist\D3D\x64` to PATH for that run only). The sweep verifies the timed output against the oracle; any `FAILED` row is the most important finding: stop and report. Save as `raw-desktop-vulkan.txt` and `raw-desktop-dx12-dxc.txt` under `docs/superpowers/results/other-adapters/tuning/`. Also run `pipeline_stats` (wgpu mode) for unrolled TPI 4, 8 (wg 64) and report VGPRs, scratch bytes and the counts of `v_mad_u32_u24`, `v_mul_lo_u32`, `v_mul_u32_u24`.
- [ ] **Step 5: results document.** Compare TPI 8/16/32 after (a) against the previous verified rows in `docs/superpowers/results/other-adapters/unrolled/raw-desktop-vulkan-verified.txt` and `raw-desktop-dx12-dxc-verified.txt` (same machine, same sweep settings), and add TPI 4. State plainly if (a) or TPI 4 is slower or within noise (run-to-run differences in earlier repeats were about 1%). Measured vs projected labelled; no verdict. Laptop and host2 GPUs are run by the controller afterwards.
- [ ] **Step 6: commit** (`Cheaper Montgomery factor and TPI 4 in the unrolled kernel`), with the exact trailer.
