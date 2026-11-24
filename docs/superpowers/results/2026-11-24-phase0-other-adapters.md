# Phase 0 addendum: other adapters (2026-11-24)

Code under test: commit `63ae502` (same kernels, runner and sweep as the main results doc, `2026-11-24-phase0-results.md`). Raw outputs are in `other-adapters/`. The main doc's definitions apply: **squarings/s is measured; Gmad/s is derived; tests/s is a projection** (squarings/s / 2029), not a measured full Fermat test.

## 1. Where it ran

| Machine | Adapter | API | Subgroup size | Result file |
|---|---|---|---|---|
| Laptop `test-laptop` (Ubuntu 26.04) | NVIDIA GeForce RTX 4070 Laptop (8 GB, driver 580.178) | Vulkan | 32..32 | `raw-tests-laptop-vulkan.txt`, `raw-sweep-laptop-vulkan.txt` |
| same | AMD Radeon 780M iGPU (RADV, Mesa) | Vulkan | 32..64 | same files |
| same | llvmpipe (Mesa software Vulkan, LLVM 21.1.8: a CPU adapter) | Vulkan | 8..8 | same files |
| Windows desktop | Microsoft Basic Render Driver (WARP, CPU) | Dx12 | no SUBGROUP feature | `raw-tests-warp-dx12.txt` (tests only) |

Rust 1.100.0 was installed user-level on the laptop with rustup (no sudo) to build there. The Quadro P4000 on the proxmox host `host2` was **not** tested: the host has no Vulkan loader, so wgpu finds no adapter there (see section 5).

## 2. Correctness (the strict test suite, no skips allowed except where stated)

Each run is the `bigmont-core` suite: 133 instances of random 2029-bit moduli plus two carry-stress moduli, 3 chained squarings, compared with the CPU oracle.

| Adapter | TPI 1 | TPI 8 | TPI 16 | TPI 32 (wg 64 / 128 / 256) |
|---|---|---|---|---|
| RTX 4070 Laptop (Vulkan) | pass | pass | pass | pass / pass / pass |
| Radeon 780M (Vulkan) | pass | pass | pass | pass / pass / pass |
| llvmpipe (Vulkan, `BIGMONT_ALLOW_SKIP=1`) | pass | pass | skipped: subgroup size 8 is not a multiple of 16 | skipped (all three), same reason |
| WARP (Dx12, `BIGMONT_ALLOW_SKIP=1`) | pass | skipped: no SUBGROUP | skipped | skipped |

- On the RTX 4070 and the 780M no configuration skipped. Both are first-time confirmations of the cooperative kernels on an NVIDIA GPU and on an RDNA3 GPU.
- llvmpipe ran the cooperative kernel at TPI 8, which is an independent implementation (a different shader compiler and execution model) of the same WGSL, and it matched the oracle.
- The suite's "5 passed" line counts allowed skips as passes. Read the table above, not the count.

## 3. Throughput (measured squarings/s; best row per adapter; sweep run with `--max-count 131072`)

| Adapter | One-thread (TPI 1), best | Best cooperative row | squarings/s | projected tests/s* |
|---|---|---|---|---|
| RTX 4070 Laptop | 467,666 (wg 128, 32,768 instances) | TPI 32, wg 256, 131,072 instances | 3,407,323 | 1,679 |
| Radeon 780M | 86,949 (wg 64, only 2,048 instances) | TPI 8, wg 256, 65,536 instances | 6,837,139 | 3,370 |
| llvmpipe (CPU) | 57,391 | TPI 8, wg 64, 8,192 instances | 246,666 | 122 |
| *RX 6700 XT, Windows, for reference (main doc)* | about 399,000 | TPI 16, wg 64 (Vulkan) | 4,511,000 (Vulkan), 4,690,000 (Dx12 with dxcompiler) | 2,223 / 2,312 |

\* projection, as defined in the main doc. The proposed gate in the main doc (6,000 projected tests/s) is not met by any row here. The highest printed figure is the 780M's 3,370. That is a printed figure with the caveats below, not a verdict.

Full tables are in `other-adapters/raw-sweep-laptop-vulkan.txt`.

## 4. Why these numbers must not be read as a ranking of the cards

1. **The timings look floored by a fixed per-dispatch cost.** On the RTX 4070, TPI 8 at wg 64 ran 131,072 instances in 85.80 ms, while TPI 8 at wg 128 ran 65,536 instances (half the work) in 72.09 ms. Doubling the work added about 14 ms, which does not fit a throughput-limited kernel. The same pattern appears across the 4070's cooperative rows (61 to 86 ms for batches that differ by 2x). So the 4070's printed squarings/s are probably well below its real rate, and the within-TPI differences (for example TPI 8: 3.06 M versus 1.82 M) are mostly batch-size artifacts. Each dispatch does only 2 squarings, which the earlier review already flagged (finding M6). Re-measuring with many more squarings per dispatch is needed before any comparison.
2. **The Radeon 780M printed faster than the RX 6700 XT** (6.84 M versus 4.51 M squarings/s) although the 780M has roughly a third of the shader count. Not explained. Candidates, none tested: the Linux Mesa (RADV/ACO) shader compiler producing better code for this kernel than AMD's Windows Vulkan driver, RDNA3 versus RDNA2 integer throughput, and the fixed-cost issue in item 1 affecting the cards differently. The only direct way to separate compiler from hardware is to run the 6700 XT under Linux, which has not been done.
3. **Different operating systems and drivers** (Windows with AMD and Dx12 drivers, Linux with NVIDIA proprietary and Mesa RADV), so the 6700 XT row is not like-for-like with the laptop rows.
4. **Batch sizes differ between rows** (the sweep stops growing a batch once a probe dispatch passes 50 ms), for example the 780M's one-thread rows ran only 2,048 to 4,096 instances, so one-thread-versus-cooperative ratios are not meaningful on the 780M.
5. **The best TPI differs by vendor in this data**: TPI 32 on the 4070 (subgroup 32), TPI 8 on the 780M (subgroup 32..64), TPI 8 on llvmpipe (subgroup 8, the only cooperative width it can run). Given the noise in item 1, this is a hint to retest, not an established rule. (Subgroup size is not proof of a mapping to the best TPI.)

## 5. Not tested

- Quadro P4000 on `host2`: no Vulkan loader on the Proxmox host. A test needs a container or VM with GPU passthrough and Vulkan packages (or system package installs on the hypervisor, which were not made).
- RX 6700 XT under Linux (to separate compiler from hardware, item 2 above).
- Wave64 mode (no run has observed subgroup size 64).
- Intel and Apple GPUs; browsers (the cooperative kernels are still not browser-valid because naga 30 rejects `enable subgroups;` and the emitted WGSL omits it).
- CGBN on any card; a full Fermat test.
- Only llvmpipe on Linux and WARP on Windows were tried as CPU adapters. They show the kernels run without a GPU. They are not a performance fallback (llvmpipe: about 0.25 M squarings/s, projected 122 tests/s).

## 6. What would settle the open questions (candidates for Phase 1 or a follow-up, not done)

1. Time with many squarings per dispatch (for example 32 to 256) so fixed costs vanish, then redo the laptop and desktop sweeps.
2. Run the 6700 XT under Linux (a USB-booted Linux or a spare machine; not available here today).
3. A test container on `host2` with GPU passthrough and Vulkan, to cover the P4000.
