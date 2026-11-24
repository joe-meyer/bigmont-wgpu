# Phase 0 leftovers and Phase 1 backlog (2026-11-24)

Collected from the per-task reviews, the whole-branch review and the remote runs. Nothing here blocks reading the Phase 0 results; items are grouped by when they matter.

## Fix before relying on the numbers (measurement)
- Time with many squarings per dispatch (32 to 256, not 2) so fixed per-dispatch cost vanishes, then redo every sweep (desktop and laptop). See `results/2026-11-24-phase0-other-adapters.md` section 4.
- Spot-check the timed outputs of the benchmark against the oracle (about 16 instances of the large batches); today only the 50-instance validation batch is checked.
- Run the RX 6700 XT under Linux to separate "AMD Windows driver" from "RDNA2 hardware" (the 780M under Mesa printed faster).
- Investigate the one-thread kernel being about 4x faster on Dx12 without dxcompiler (presumably FXC) than under Vulkan or Dx12/DXC.

## Known small defects (deferred minors)
- `spike.rs` footer still says "(doubling folded into the reduction)", which contradicts the corrected projection wording in the results doc.
- Results doc: "TPI 16 led at all three workgroup sizes" is slightly overstated (TPI 32 at wg 64 was skipped at cap 131072, so wg 64 compares TPI 16 with TPI 8 only); Dx12 lead is 3.8% (written "about 4%").
- Oracle: `cargo test` result (8 tests) is cited in the doc but has no raw file; the raw test files carry no command or environment header.
- A "5 passed" line in the suite counts allowed skips as passes (only with `BIGMONT_ALLOW_SKIP=1`).
- Error scopes now cover Validation, Internal and OutOfMemory around creation calls; the exit-nonzero and REJECTED paths of `spike` were not exercised with a deliberately broken shader.
- Process-global `INIT_LOCK` in `Gpu::new` is a mitigation for intermittent adapter misses under parallel tests; root cause not found.
- `num-bigint` / `num-traits` are normal dependencies of core only for the test helper `Batch::random_values`.
- PERIOD (iterations between light carry passes) could be raised to about 24 to 31; profile the 8% efficiency.

## Portability and browser (before any web work)
- naga 30 rejects `enable subgroups;`, so the emitted cooperative WGSL is not browser-valid. A web build must emit the directive; check Tint's `subgroup_uniformity` diagnostic (may need `diagnostic(off, subgroup_uniformity)`).
- Kernel assumes each subgroup is a contiguous TPI-aligned range of `local_invocation_index`; add a per-adapter startup self-validation (canary) before real work.
- Dx12 shuffles use non-uniform lane indices; verified only on AMD with DXC.
- Wave64 never observed; TPI 32 with two instances per subgroup untested. Intel (subgroup 8 to 32) and Apple GPUs untested. `llvmpipe` has subgroup size 8, so only TPI 1 and 8 run there.
- Workgroup-memory cooperative backend (spec backend 2) was not built; needed where subgroups are unavailable (for example Dx12 without dxcompiler).
- Quadro P4000: correctness passes (all 5 configs) in container CT 104 on `host2`; its throughput is low-confidence (shared GPU, ramping clocks, `--quick` sweep only). Re-run with the other containers idle.
- No Intel GPU exists on any available machine; Intel (subgroup 8 to 32) untested. Find one.

## CPU path (decide early in Phase 1)
- Add a native CPU backend (GMP-class big-integer modexp) as a first-class engine option: fallback where no usable GPU exists (host1's GeForce 9800 GT can run nothing in wgpu) and the baseline GPU results must beat. A single old quad-core i7-3770K is estimated at about 1,000 to 1,300 tests/s native (`results/2026-11-24-phase0-other-adapters.md` section 5a, from `openssl speed`), against 122 projected tests/s for the shader kernels on llvmpipe on a newer 16-thread CPU, and about 2,300 to 3,400 for the best GPU rows printed so far. Measure properly with GMP before relying on these.
- Revisit the proposed 6,000 tests/s gate in light of that CPU baseline.

## Runner design (Phase 1, per spec section 3)
- `run()` rebuilds shader, pipeline and buffers on every call; reuse them.
- A `Backend` trait and async API; move test helpers out of the public `Batch`.
- Add the groups > 65535 and storage-binding-size limits as first-class errors (a guard exists in `run()`; the `--max-count` clamp is in the example only).
