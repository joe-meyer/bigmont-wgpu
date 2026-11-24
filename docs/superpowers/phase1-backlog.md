# Phase 0 leftovers and Phase 1 backlog (2026-11-24)

Collected from the per-task reviews, the whole-branch review and the remote runs. Nothing here blocks reading the Phase 0 results; items are grouped by when they matter.

## Fix before relying on the numbers (measurement)
- (Done) Long-dispatch re-timing on all GPUs: `results/2026-11-24-phase0-other-adapters.md` section 3a. It showed fixed per-dispatch cost is small (0 to 13 ms), so my first guess (a 60 ms floor on the 4070) was wrong.
- Explain the RTX 4070's slow TPI 16 (2.15 M against 3.1 M for TPI 8 and 3.4 M for TPI 32 at every batch size) and its TPI 8 penalty at wg 128/256 (1.83 M against 3.08 M at wg 64).
- The proposed gate (6,000 projected tests/s) is not met by any measured row (best 3,941, Radeon 780M); re-derive it against the CPU baseline.
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
- Add a native CPU backend (GMP-class big-integer modexp) as a first-class engine option: fallback where no usable GPU exists (host1's GeForce 9800 GT can run nothing in wgpu) and the baseline GPU results must beat. A single old quad-core i7-3770K is estimated at about 1,000 to 1,300 tests/s native (`results/2026-11-24-phase0-other-adapters.md` section 5a, from `openssl speed`), against 122 projected tests/s for the shader kernels on llvmpipe on a newer 16-thread CPU, and about 2,300 to 3,400 for the best GPU rows printed so far. Now measured with GMP on a same-generation Xeon (host2 CT 104): about 265 tests/s per core and about 1,085 tests/s on 4 vCPUs (`tools/cpu-baseline/`); still not run on the 3770K itself or on the laptop.
- Revisit the proposed 6,000 tests/s gate in light of that CPU baseline.

## Dynamic size classes (user idea, 2026-11-24)
- Derive the limb count and lane width from the number's bit length at runtime instead of the fixed 160 limbs: `L = ceil((bits + 1) / 13)` rounded up to a multiple of the chosen TPI (needs R > 4n for the x < 2n invariant), TPI picked by a per-adapter startup autotune (best width differs by vendor and will shift with L), WGSL generated per (L, TPI, wg) and pipelines cached. Make the oracle's `LIMBS` a parameter. Bucket mixed-size batches by class, as Beacon's driver does. Keep gapsieve's unrolled one-thread kernel as the path for small sizes (roughly 260 bits and below).
- Expected effect (arithmetic, not measured): a squaring costs about 2L^2 multiply-adds and a test needs about `bits` squarings, so cost per test scales roughly with the cube of the size (about 8x cheaper at 1,024 bits than at 2,029). All Phase 0 numbers are for the 2,029-bit class only; the GMP CPU baseline scales about the same way, so GPU-versus-CPU ratios should roughly carry over.
- Test matrix grows to sizes x widths x adapters; every combination must pass the oracle before the autotune may select it.

## Runner design (Phase 1, per spec section 3)
- `run()` rebuilds shader, pipeline and buffers on every call; reuse them.
- A `Backend` trait and async API; move test helpers out of the public `Batch`.
- Add the groups > 65535 and storage-binding-size limits as first-class errors (a guard exists in `run()`; the `--max-count` clamp is in the example only).

## After the unrolled kernel (2026-11-24; see `results/2026-11-24-unrolled-cross-vendor.md`)
- (Done) Test B measured: Beacon's CUDA/CGBN driver does about 132 k tests/s on the RTX 4070 Laptop; our unrolled wgpu kernel projects to about 45.7 k, about 34.6% (`results/2026-11-24-test-b-cuda-vs-wgpu.md`). Open: close the 2.9x gap on NVIDIA (squaring symmetry, a native-only 32-bit-limb variant), and a CGBN comparison on an AMD card (needs a HIP port).
- Try TPI 4 (40 limbs per lane; CGBN allows 4) since TPI 8 was best or near-best on all four GPUs and the trend runs toward fewer lanes per number; the generator is parametric in limbs per lane.
- Squaring symmetry (the kernel computes the full product; up to roughly a quarter fewer multiply-adds, not built or measured) is now the main remaining arithmetic saving on AMD, which is near a nominal issue bound.
- Measurement hygiene: GPU timestamp queries, repeat runs for variance, more verified instances per row (25 to 27 now), inspect the NVIDIA, Mesa and Dx12/DXC code generation (only the Vulkan AMD ISA was read), and a negative control that also exercises the limb-normalisation and below-2n checks.
- Wave64 was never observed. Intel remains untested. The unrolled kernel is still not browser-valid (`enable subgroups;`).
- Dynamic size classes (above) are now cheap to try: the unrolled generator takes limbs per lane as a parameter.
- Release the test container resources if no longer needed: CT 104 on host2 (24 GB) and the user-level Rust install on the laptop.
