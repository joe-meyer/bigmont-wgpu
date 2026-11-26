# Unrolled cooperative kernel versus the rolled kernel: RX 6700 XT results

Date: 2026-11-24. Machine: the desktop with the AMD Radeon RX 6700 XT (adapter reports subgroup size 32..64). Code: commit aa6fcdb (unrolled generator and variant plumbing) plus the Task 2 example changes (the `--variant` flag in `spike`, the optional `unrolled` argument in `pipeline_stats`), commit 84b4f1c. Task 3 (review follow-up, section 6) adds oracle verification of the timed output to `spike`, makes `pipeline_stats` mirror wgpu's compilation mode, and corrects the interpretation in sections 3 and 4; measured numbers in the tables are unchanged. Raw outputs are in `docs/results/other-adapters/unrolled/`; the rolled baseline figures come from the earlier amortised sweeps in `docs/results/other-adapters/amortised/` (commit 0ca00ef, same machine, taken earlier today).

This document reports numbers and says what they do and do not show. It gives no go/no-go verdict.

## 1. What was measured and what was not

Measured on this machine, this run:

- Squarings per second of the unrolled kernel (the `sq/s @long` column of the sweep: instances x squarings per dispatch / time of the longest dispatch, best of 3), on Vulkan and on Dx12 (with `dxcompiler.dll` and `dxil.dll` on `PATH` for that run only), for TPI 8, 16, 32 and workgroup sizes 64, 128, 256, at `--max-count 131070`.
- A same-batch comparison of rolled and unrolled on Vulkan at exactly 65536 instances (TPI 8, 16, 32, wg 64).
- Driver compiler statistics (registers, scratch) and instruction counts from the saved ISA of the Vulkan pipeline, for the unrolled and (re-measured today) the rolled kernel at TPI 8, 16, 32, wg 64. These Task 2 counts (section 3) are from the wave64 `plain` compile, which is not the mode wgpu uses; the matching wgpu-mode counts are in section 6.3. The VGPR, SGPR and scratch figures come from the driver's statistics output, which was not saved; they are not reproducible from the committed ISA text alone (the VGPR counts do agree with the highest vector register index used in the ISA).
- Correctness: every configuration in the sweep passed the spike's own validation (50 random instances, 3 squarings, compared with the oracle). No row printed `VALIDATION FAILED`, `REJECTED` or `bench failed`. The strict Vulkan test suite passes (11 tests in `bigmont-core`, 8 in `bigmont-oracle`, no skips). The Task 2 sweep timed dispatches of up to 256 squarings that this validation did not cover; Task 3 re-ran the headline sweeps with the timed output itself compared with the oracle (section 6), and every row passed.

Derived or projected (not measured):

- `tests/s = squarings/s / 2029`: one squaring per exponent bit of a base-2 Fermat test, assuming the base-2 doubling costs nothing. This is a projection, not a run of a full test. No Fermat test was run.
- `Gmad/s` in the raw files is squarings/s times a nominal 2 x 160 x 160 multiply-adds, not a count of instructions executed.

Not measured by this task:

- The RTX 4070 laptop GPU and host2 (P4000): the controller runs those. Nothing here says the speed-up carries over to NVIDIA.
- Any CPU or CUDA/CGBN baseline, any full primality test, any modulus size other than 2029 bits, any batch below 16384 instances, power, clocks or temperature.
- (Corrected in Task 3, section 6.) The Task 2 text said the subgroup size wgpu used was unknown and that the ISA was from a compile reporting subgroup size 64. Task 3 measured the size through wgpu (32 on Vulkan and Dx12) and found that the Task 2 ISA was compiled in a different mode (wave64, no wgpu stage flag) from the one wgpu uses (wave32). The ISA tables in section 3 are therefore for wave64 code that wgpu does not run; section 6 has the matching-mode counts.

## 2. Speed: unrolled versus rolled

Squarings per second at the longest dispatch measured, best workgroup size per row. Rolled figures are from the amortised raw files (8 squarings per dispatch, about 215 to 245 ms per dispatch); unrolled figures are from this run (128 squarings per dispatch, 204 to 347 ms; the fixed per-dispatch cost is about 5 ms, i.e. 1% to 2% of a dispatch in both cases). Run-to-run differences of a few percent between rows of identical configuration are visible in the raw files (for example the rolled TPI 32 rows), so ratios are good to roughly that precision. Each row is a single run, not repeated. All times are wall-clock around the queue submit and a blocking poll (an `Instant` before `submit` and after `poll(wait)`), best of 3 for the long dispatches and best of 5 for the 2-squaring dispatch; no GPU timestamps were used, so they include submission and completion latency (about 5 ms per dispatch by the slope fit).

### Vulkan

| TPI | rolled best wg | rolled sq/s | unrolled best wg | unrolled sq/s | ratio | rolled tests/s* | unrolled tests/s* |
|---|---|---|---|---|---|---|---|
| 1 | 64 | 404,197 | skipped (rolled only) | - | - | 199 | - |
| 8 | 64 | 4,484,855 | 64 | 82,081,729 | 18.3x | 2,210 | 40,454 |
| 16 | 64 | 4,881,358 | 64 | 64,045,199 | 13.1x | 2,406 | 31,565 |
| 32 | 128 | 4,500,934 | 64 | 48,319,081 | 10.7x | 2,218 | 23,814 |

### Dx12 (DXC)

| TPI | rolled best wg | rolled sq/s | unrolled best wg | unrolled sq/s | ratio | rolled tests/s* | unrolled tests/s* |
|---|---|---|---|---|---|---|---|
| 1 | 64 | 413,226 | skipped (rolled only) | - | - | 204 | - |
| 8 | 64 | 4,839,225 | 64 | 74,132,565 | 15.3x | 2,385 | 36,537 |
| 16 | 128 | 5,021,299 | 64 | 54,314,985 | 10.8x | 2,475 | 26,769 |
| 32 | 256 | 4,842,924 | 64 | 39,329,653 | 8.1x | 2,387 | 19,384 |

\* tests/s = sq/s / 2029, a projection (section 1).

Best unrolled versus best rolled row overall: Vulkan 82.1M versus 4.88M squarings/s (16.8x); Dx12 74.1M versus 5.02M (14.8x). Section 6 repeats the sweeps with oracle-verified timed output and gets the same ordering and ratios within run-to-run noise.

The rolled baseline is a fair baseline in the sense that matters here: its `sq/s slope` column (fixed per-dispatch cost removed) is only 0.6% to 5.9% above its `sq/s @long` column in the amortised raw files (all 24 rows, both backends; the 5.9% is the Vulkan TPI 32 wg 128 row, the others are at most 2.8%), and giving it more work per dispatch did not help: at 16 squarings per dispatch (same-batch check below, 65536 instances) it measured 4.44M, 4.97M and 4.43M squarings/s at TPI 8, 16, 32, against 4.48M, 4.88M and 4.32M to 4.50M at 8 squarings per dispatch. Launch overhead is therefore not what holds the rolled kernel back.

Observations (from the numbers above, causes not tested):

- In this sweep the unrolled kernel is faster in every configuration it can run. There is no configuration where it is slower or equal.
- The ordering between widths changes. The rolled kernel was nearly flat across TPI 8, 16, 32 (about 4.5M to 5.0M). The unrolled kernel falls steadily with width: TPI 8 > TPI 16 > TPI 32 on both backends, TPI 8 being about 1.7x TPI 32 on Vulkan. A plausible but untested reason is that wider instances do the same multiply-adds with more cross-lane traffic per multiply-add. Section 6.4 gives a model that fits the ordering through vector instructions per squaring in the ISA of the code wgpu actually compiles; the earlier suggestion here that each cross-lane broadcast is a costly waterfall loop was withdrawn (section 6.3).
- Workgroup size (64, 128, 256) makes almost no difference to the unrolled kernel (within about 2.5% on both backends).
- Vulkan is faster than Dx12 for the unrolled kernel (by 10% at TPI 8 up to 23% at TPI 32); for the rolled kernel the two backends were within a few percent. The Dx12 TPI 32 rows ended at 64 squarings per dispatch rather than 128 (the next doubling was predicted to exceed the sweep's 0.4 s budget). All dispatches stayed under 0.35 s.

### Same-batch check (Vulkan, exactly 65536 instances, wg 64)

The sweep rows choose their own batch sizes (the growth loop stops once a 2-squaring dispatch takes 50 ms), so a ratio between two sweeps could partly reflect different batch sizes. Both variants were also run at one fixed batch (`--quick --tpi 8,16,32 --fixed-count 65536 --max-count 65536`):

| TPI | rolled sq/s (16 sq/dispatch) | unrolled sq/s (256 sq/dispatch) | ratio |
|---|---|---|---|
| 8 | 4,442,944 | 83,574,847 | 18.8x |
| 16 | 4,966,469 | 63,524,963 | 12.8x |
| 32 | 4,427,873 | 48,441,029 | 10.9x |

These are within about 1% to 2% of the sweep figures, and the rolled figures re-measured today match the earlier amortised file, so the batch-size difference does not explain the gap.

## 3. Compiler evidence (Vulkan, RX 6700 XT, TPI x, wg 64): Task 2 `ash` compile, wave64

**Read this section with section 6.** The counts below are what the driver produces for the SPIR-V from naga's default options with no stage flags (`plain` mode: the driver compiles it as wave64). wgpu compiles with `ALLOW_VARYING_SUBGROUP_SIZE` and different SPIR-V options, and the driver then produces wave32 code with `ds_bpermute` shuffles and no waterfall loops. The `v_mad_u32_u24` counts are the same in both modes (840/220/60) but the shuffle, branch and register figures, and the waterfall interpretation further down, do not carry over to what wgpu timed.

From `pipeline_stats` (VK_KHR_pipeline_executable_properties) and the saved ISA text, counted with `grep -c` on the mnemonic. These are static counts over the program text, not counts of instructions executed. Rolled figures were re-measured today and agree with section 2 of `2026-11-24-cgbn-lessons.md` (VGPRs 71/47/38, scratch 0, 2 `v_mad_u32_u24`, 20 `v_movrel`, 57 to 61 `s_cbranch`).

| | rolled TPI 8 | rolled TPI 16 | rolled TPI 32 | unrolled TPI 8 | unrolled TPI 16 | unrolled TPI 32 |
|---|---|---|---|---|---|---|
| VGPRs | 71 | 47 | 38 | 80 | 48 | 56 |
| SGPRs | 42 | 42 | 41 | 36 | 30 | 30 |
| scratch bytes | 0 | 0 | 0 | 0 | 0 | 0 |
| `v_mad_u32_u24` | 2 | 2 | 2 | 840 | 220 | 60 |
| `v_movrel*` | 20 | 20 | 20 | 0 | 0 | 0 |
| `ds_bpermute` | 0 | 0 | 0 | 0 | 0 | 0 |
| `ds_swizzle` | 0 | 0 | 0 | 0 | 0 | 0 |
| `v_readlane_b32` | 4 | 4 | 6 | 44 | 24 | 21 |
| `v_mov_b32_dpp` | 2 | 2 | 0 | 22 | 12 | 0 |
| `s_cbranch*` (all) | 57 | 57 | 61 | 95 | 56 | 43 |
| of which `execz`/`execnz` | 9 | 9 | 13 | 88 | 48 | 35 |
| ISA text lines | 760 | 738 | 756 | 1949 | 915 | 571 |

What the ISA shows:

- The multiply-adds are now straight-line: 840, 220 and 60 `v_mad_u32_u24` for LPL 20, 10, 5 (about 2 x LPL x LPL plus a few), against 2 in the rolled program. The `v_movrel` indirect moves are gone (20 to 0). The 24-bit-operand masks do steer the compiler to `v_mad_u32_u24`; the only other multiplies are `v_mul_lo_u32` (21, 11 and 6, about one per iteration, for `p * ninv`).
- No scratch use and no spills in any variant. Register use rose at TPI 8 (71 to 80) and TPI 32 (38 to 56) and was flat at TPI 16 (47 to 48). The effect of that on occupancy was not analysed.
- (Superseded by section 6: this reading is true of the wave64 `plain` compile only and does not describe the code wgpu timed.) Total branches did not go down at TPI 8 (57 to 95) but the nature changed: in the unrolled program most are `s_cbranch_execz`/`execnz` pairs belonging to loops of the form `v_readfirstlane_b32` / `v_cmpx_eq_u32` / `v_readlane_b32` / `s_andn2_wrexec_b64` / `s_cbranch_execnz`. My reading of the ISA is that the driver implements each cross-lane broadcast (`subgroupShuffle` with a computed index `base + src_lane`) as one such loop over the distinct index values in the wave (a waterfall), rather than as `ds_bpermute`; there are about two per unrolled iteration at TPI 8 (44 `v_readlane` for 20 iterations). The `down1` shuffle shows as `v_mov_b32_dpp` (22 at TPI 8). I did not execute these loops under a profiler, so how many times each waterfall loops per wave at run time (it depends on how many instances share a wave) is not measured. In the wave64 compile this was the most visible overhead; it is not present in the wave32 code wgpu uses (section 6).
- The ISA grew: 1949 lines at TPI 8 versus 760 rolled. Instruction-cache effects were not examined.
- The ISA text files are now committed under `docs/results/other-adapters/unrolled/isa/`, named `{variant}_tpi{N}_wg64_{mode}.isa.txt` with mode `plain` (this section), `wgpu` (section 6) or `flags` (stage flag only, for separating the effects). The `plain` files reproduce the Task 2 counts above.

## 4. Correctness evidence for the numbers above

- The spike validates each configuration before timing it: 50 random 2029-bit instances, 3 chained squarings, compared with the oracle (`mont_square_chain`), limbs required to be normalised. All unrolled rows passed on both backends. The sweep timings are for dispatches of up to 256 squarings, which that validation does not cover.
- To cover that, a new test `unrolled_long_chain_matches_oracle` (in `crates/bigmont-core/tests/chain.rs`) runs the unrolled kernel at TPI 8, 16, 32 (wg 64) on 72 instances (70 random plus the two carry-stress moduli) for 128 chained squarings and compares with the oracle. It passes on Vulkan. Dx12 was not run through the test suite in this task; the Dx12 rows rely on the 3-squaring spike validation.
- TPI 8 and 16 at wg 128 and 256 were validated only by the spike's 3-squaring check; the test suite covers wg 128 and 256 only for TPI 32.
- The nominal Gmad/s of the unrolled TPI 8 rows (about 4.2 x 10^12 per second) is a derived figure (51,200 multiply-adds per squaring, see the footnote below). The earlier statement that this is "about 60% of peak" with "limited headroom" is withdrawn: the peak used (2560 lanes at about 2.6 GHz, about 6.6 x 10^12 lane-operations per second) is an unverified spec-sheet estimate, and the multiply-adds are not the only instructions issued. Section 6 gives the cycle-budget argument, which accounts for the measured rate with the instruction mix of the code wgpu runs and says nothing reliable about headroom.
- Footnote: `Gmad/s` counts 2 x 160 x 160 = 51,200 multiply-adds per squaring (the algorithmic count). The unrolled kernel also has every lane recompute `p` and `carry0` in each iteration (two extra multiply-adds per lane per iteration), so at TPI 8 the multiply-adds actually issued are about 53,760 per squaring (42 per lane per iteration x 8 lanes x 160 iterations), about 5% more than counted.
- The old `mul_bench` figure of 2,861 G chain-steps/s (section 4 of `2026-11-24-phase0-results.md`) was never a hardware ceiling: that tool's own add-only kernel reached only about 2.4 T/s, so the chain figure reflects the tool, not the ALUs. It is superseded: the unrolled kernel sustains about 4.2 T multiply-adds per second (Vulkan TPI 8), above that "ceiling". Fractions of 2,861 quoted in older documents should not be read as efficiency.

## 5. Reproducing

```
cargo build --release -p bigmont-core --examples
# ISA
./target/release/examples/pipeline_stats.exe 8 64 6700 unrolled     # also 16, 32, and "rolled"
# sweeps (single process, in this order)
BIGMONT_ADAPTER=6700 BIGMONT_BACKEND=vulkan ./target/release/examples/spike.exe --variant unrolled --max-count 131070
PATH="C:\Program Files (x86)\Windows Kits\10\Redist\D3D\x64;$PATH" BIGMONT_ADAPTER=6700 BIGMONT_BACKEND=dx12 ./target/release/examples/spike.exe --variant unrolled --max-count 131070
# same-batch check, each variant
BIGMONT_ADAPTER=6700 BIGMONT_BACKEND=vulkan ./target/release/examples/spike.exe --variant rolled --quick --tpi 8,16,32 --fixed-count 65536 --max-count 65536
# strict tests
BIGMONT_ADAPTER=6700 BIGMONT_BACKEND=vulkan cargo test --release --workspace -- --test-threads=1
```

Task 3 additions:

```
./target/release/examples/pipeline_stats.exe 8 64 6700 unrolled wgpu      # modes: wgpu (default), flags, plain; also 16, 32, rolled
BIGMONT_ADAPTER=6700 BIGMONT_BACKEND=vulkan ./target/release/examples/subgroup_probe.exe     # also BIGMONT_BACKEND=dx12
BIGMONT_ADAPTER=6700 BIGMONT_BACKEND=vulkan ./target/release/examples/spike.exe --variant unrolled --max-count 131070   # now oracle-verifies the timed output
BIGMONT_ADAPTER=6700 BIGMONT_BACKEND=vulkan ./target/release/examples/spike.exe --variant rolled --max-count 131070
# Dx12: same, with the Windows Kits Redist\D3D\x64 directory prepended to PATH and BIGMONT_BACKEND=dx12
```

Raw files: `other-adapters/unrolled/raw-desktop-vulkan.txt`, `raw-desktop-dx12-dxc.txt`, `raw-desktop-vulkan-fixed-count-65536-rolled.txt`, `raw-desktop-vulkan-fixed-count-65536-unrolled.txt`. Each starts with a line giving the commit, backend, variant, time and command; rows now carry a leading `variant` column, and the header block has a `variant:` line, so the files are self-describing.

The Task 3 sweeps were run on example code that was uncommitted at the time (their header lines say so, naming base commit a155f5b); that code was committed afterwards in 713cf93.

Task 3 raw files (timed output verified against the oracle, section 6): `other-adapters/unrolled/raw-desktop-vulkan-verified.txt`, `raw-desktop-dx12-dxc-verified.txt` (unrolled), `raw-desktop-vulkan-rolled-verified.txt` (rolled); ISA text in `other-adapters/unrolled/isa/`.

## 6. Task 3: verified timings, the compilation mode wgpu really uses, and the cycle budget

### 6.1 Timed output compared with the oracle (measured)

`spike` now compares the output of the exact runs it times with the oracle. `run(.., s, r)` chains `r` dispatches of `s` squarings from the previous output, so every instance has had `s x r` squarings. The spike keeps the output of the run whose time it reports (the final long run, and separately the 2 squarings x 5 run behind `sq/s @2`) and checks 25 to 27 instances: the first 4, the last 4, 16 spread across the batch, and the first instance of the second and of the last workgroup and the middle instance. For each: every limb below 2^13, value below 2n, and value mod n equal to `mont_square_chain(n, x0, s x r)`. A failure prints `TIMED OUTPUT FAILED ORACLE` with the failing indices and no speed, and the exit status is nonzero. As a negative control, the oracle chain was temporarily made one squaring too long (`squarings + 1`): every sampled instance failed (the wrong-value path) and no speed was printed; the change was reverted. This exercised only the wrong-value path (not the limb-normalisation or below-2n checks), the process exit status of that run was not recorded, and no artifact of the run is committed.

Result: **every row passed**, in all three sweeps (12 rolled rows, 9 unrolled Vulkan rows, 9 unrolled Dx12 rows; no `FAILED`, `REJECTED` or `bench failed`; exit status 0). Long runs were verified after 384 squarings per instance (128 per dispatch x 3 chained dispatches; Dx12 TPI 32: 192; rolled: 24), the short runs after 10. The timed rows reproduce the earlier sweeps within run-to-run noise. Best rows per TPI:

| Backend, variant | best row | squarings/s @long | tests/s* | earlier figure (section 2) |
|---|---|---|---|---|
| Vulkan, unrolled | TPI 8, wg 64 | 81,774,171 | 40,303 | 82,081,729 |
| Vulkan, unrolled | TPI 16, wg 256 | 64,141,369 | 31,612 | 64,045,199 (wg 64) |
| Vulkan, unrolled | TPI 32, wg 64 | 48,382,260 | 23,845 | 48,319,081 |
| Dx12, unrolled | TPI 8, wg 128 | 75,445,418 | 37,184 | 74,132,565 (wg 64) |
| Dx12, unrolled | TPI 16, wg 128 | 54,406,683 | 26,815 | 54,314,985 (wg 64) |
| Dx12, unrolled | TPI 32, wg 256 | 38,505,978 | 18,978 | 39,329,653 (wg 64) |
| Vulkan, rolled | TPI 8, wg 64 | 4,486,434 | 2,211 | 4,484,855 |
| Vulkan, rolled | TPI 16, wg 64 | 4,867,623 | 2,399 | 4,881,358 |
| Vulkan, rolled | TPI 32, wg 256 | 4,498,482 | 2,217 | 4,500,934 (wg 128) |

The Dx12 TPI 32 rows ran at 64 squarings per dispatch in both sweeps (section 2); the 39.3M versus 38.5M difference is run-to-run noise at the same squarings per dispatch (and a different workgroup size of the best row).

\* projection (squarings/s / 2029), as in section 1. Verification covers a sample of 25 to 27 instances per row, not every instance. The section 2 tables are unchanged; this table is added beside them.

### 6.2 What wgpu does for subgroup size (read from wgpu-hal 30.0.2, confirmed by running)

- Source (`wgpu-hal-30.0.2/src/vulkan/device.rs`, `compile_stage`): when the device has `Features::SUBGROUP`, every pipeline stage is created with `PipelineShaderStageCreateFlags::ALLOW_VARYING_SUBGROUP_SIZE`. wgpu sets no required subgroup size (no required-subgroup-size create info) and not `REQUIRE_FULL_SUBGROUPS`. The `VK_EXT_subgroup_size_control` feature is enabled on the device (promoted to Vulkan 1.3). The SPIR-V is also written with different options from naga's defaults: SPIR-V 1.6, bounds-check policy Restrict on storage buffers (unless robustBufferAccess2 is present), native workgroup zero-initialisation, loop bounding, and the GroupNonUniform capabilities.
- Measured through wgpu with the new `subgroup_probe` example (a 64-invocation workgroup writing the `subgroup_size` and `subgroup_invocation_id` built-ins; 16,384 invocations): `subgroup_size` was 32 for every invocation and the largest `subgroup_invocation_id` was 31, on Vulkan and on Dx12. The adapter's advertised range 32..64 is therefore resolved to 32 for this trivial probe pipeline on this card and driver; the probe did not run on the kernels themselves. Also, the `pipeline_stats` `wgpu` mode is a stand-in, not wgpu's own pipeline: wgpu-hal also enables the core `robustBufferAccess` device feature (`adapter.rs`, `.robust_buffer_access(private_caps.robust_buffer_access)`), which `pipeline_stats` does not, and it creates the device with wgpu's own feature and extension set. "The wave32 code wgpu compiles" therefore rests on the probe (size 32 at run time) together with the stand-in ISA header (`W32`), not on an ISA dump of wgpu's actual pipeline.
- `pipeline_stats` earlier compiled with no stage flags and naga's default options, and the driver produced wave64 (ISA header `UC_VERSION_W64_BIT`). It now has three modes: `wgpu` (default: the stage flag, the device feature and the SPIR-V options above, with the binding map the identity on set 0), `flags` (only the stage flag and feature) and `plain` (the old behaviour). The flag alone is enough to change the compilation: in `flags` mode the TPI 8 program is wave32 (`UC_VERSION_W32_BIT`) with 44 `ds_bpermute` and 0 `v_readlane`. The executable "subgroup size" field the driver reports says 64 in every mode, including the wave32 ones; the ISA header and the probe are the reliable evidence, not that field.
- Not checked: the Dx12 path. wgpu compiles WGSL to DXIL through DXC there and there is no ISA dump for it; only the runtime subgroup size (32) was measured.
- Not separated further: how much of the Vulkan difference at TPI 16 and 32 comes from the SPIR-V options rather than the flag (VGPRs in `flags` mode are 96/64/39 against 96/76/43 in `wgpu` mode at TPI 8/16/32).

### 6.3 ISA counts in the matching mode (Vulkan, RX 6700 XT, wg 64, driver ISA text, static counts)

| | rolled TPI 8 | rolled TPI 16 | rolled TPI 32 | unrolled TPI 8 | unrolled TPI 16 | unrolled TPI 32 |
|---|---|---|---|---|---|---|
| wave size (ISA header) | 32 | 32 | 32 | 32 | 32 | 32 |
| VGPRs | 72 | 56 | 49 | 96 | 76 | 43 |
| SGPRs | 30 | 30 | 30 | 24 | 20 | 20 |
| scratch bytes | 0 | 0 | 0 | 0 | 0 | 0 |
| `v_mad_u32_u24` | 2 | 2 | 2 | 840 | 220 | 60 |
| `v_mul_lo_u32` | 2 | 2 | 2 | 21 | 11 | 6 |
| `v_readlane_b32` | 0 | 0 | 0 | 0 | 0 | 0 |
| `v_readfirstlane_b32` | 10 | 10 | 10 | 5 | 5 | 5 |
| `ds_bpermute_b32` | 4 | 4 | 6 | 44 | 24 | 21 |
| `v_mov_b32_dpp` | 2 | 2 | 0 | 22 | 12 | 0 |
| `v_movrel*` | 20 | 20 | 20 | 0 | 0 | 0 |
| `s_cbranch*` (all) | 49 | 49 | 49 | 8 | 9 | 9 |
| of which `execz`/`execnz` | 1 | 1 | 1 | 1 | 1 | 1 |
| ISA text lines | 690 | 688 | 652 | 1668 | 723 | 425 |

Compared with the wave64 `plain` table in section 3: the `v_mad_u32_u24` counts are identical, `v_readlane` falls from 44/24/21 to 0, branches fall from 95/56/43 to 8/9/9, and each cross-lane broadcast is one `ds_bpermute_b32` (plus `v_mov_b32_dpp` for the within-eight-lane broadcast and the one-lane shift). VGPRs are higher at TPI 8 and 16 (96 and 76 against 80 and 48) and lower at TPI 32 (43 against 56). No scratch anywhere. The rolled kernel keeps its 2 multiply-adds and 20 `v_movrel` in wave32 too.

Instructions in one iteration of the owner loop (from the loop's first label to its back-edge branch in the saved ISA; one owner iteration holds LPL unrolled limb steps, i.e. 20, 10, 5; the light carry pass is inside the loop, executed every owner iteration at TPI 8 and under a branch every 2nd and 4th at TPI 16 and 32, counted statically here):

| unrolled | instructions per owner iteration | vector | scalar | `ds_bpermute` | `v_mad_u32_u24` | per limb step: all / vector / mad |
|---|---|---|---|---|---|---|
| TPI 8 | 1180 | 1087 | 52 | 41 | 840 | 59.0 / 54.4 / 42.0 |
| TPI 16 | 397 | 342 | 34 | 21 | 220 | 39.7 / 34.2 / 22.0 |
| TPI 32 | 162 | 119 | 27 | 16 | 60 | 32.4 / 23.8 / 12.0 |

So one limb step (the "iteration" of the cycle-budget argument; there are 160 per squaring at every TPI) at TPI 8 is about 42 `v_mad_u32_u24` plus about 12 other vector instructions (masks, adds, shifts, one `v_mul_lo_u32`, a conditional select, a DPP move, with the amortised carry pass) and two `ds_bpermute`.

**Does the "readlane waterfall" reading still hold? No.** The waterfall loops (`v_readfirstlane` / `v_cmpx` / `v_readlane` / `s_cbranch_execnz`) exist in the wave64 compile of section 3 only. In the matching (wgpu) mode there is no `v_readlane`, one `execz`/`execnz` branch, and the shuffles are `ds_bpermute_b32`. The Task 2 statement that the waterfall is the most visible remaining overhead, and its link to the width ordering, do not apply to the timed code.

### 6.4 Cycle budget (derived from the counts above: a model, not a measurement)

At the Vulkan TPI 8 row (81.77M squarings/s): limb steps x lanes = 81.77e6 x 160 x 8 = 1.05e11 lane-steps per second. A wave32 instruction covers 32 lanes, so that is 3.28e9 wave-limb-steps per second. At about 54.4 vector instructions per limb step this is about 1.78e11 vector-instruction issues per second (the multiply-adds alone: 42 per step, 1.38e11, i.e. 4.4e12 multiply-adds per second). The card has 40 CUs x 2 SIMD32 = 80 SIMD32 units; at the nominal 2.58 GHz boost clock that is 2.06e11 issue slots per second (6.6e12 lane-operations). **The clock and the assumption of one vector instruction per SIMD per clock are unverified spec-sheet estimates** (clocks were not read during the runs). On that estimate the unrolled TPI 8 kernel runs at about 85% to 87% of the vector-issue bound implied by its own static instruction count; the same calculation gives about 85% at TPI 16 (34.2 vector instructions per limb step) and about 89% at TPI 32 (23.8). Caveats pushing in both directions: static counts weigh the carry pass and loop control by text size, not by execution count; `v_mul_lo_u32` is probably quarter rate (about 3 more slots per limb step at TPI 8, a few percent); the code outside the owner loop (final carry normalisation) is not counted; a lower real clock would push the percentages towards or past 100%.

What this says: if the clock and one-instruction-per-clock assumptions hold, the measured rate is consistent with the wave32 code wgpu compiles (the stand-in compile, see 6.2) and sits near the vector-issue bound implied by that instruction mix; if they do not, the percentages above move and this sentence says nothing. The ordering TPI 8 > 16 > 32 follows the vector instructions per squaring (160 x TPI / 32 x vector instructions per limb step: about 2,180, 2,740 and 3,810 wave-instructions). Consistent with the withdrawal in section 4, it does not say how much faster a better kernel could be: if the model is right the limit is the instruction count of the straight-line step, not the shuffles. The wave64 picture (about 50 vector instructions per step plus two waterfall loops per step) could not have run at 82 M squarings/s within this budget, a further reason to believe the wave64 compile is not what wgpu timed. This was not tested by running the wave64 code.

Not established here: the actual clock during the runs, power or thermal state, per-instruction issue rates (dual issue, the `v_mad_u32_u24` rate, `ds_bpermute` throughput), occupancy (VGPR counts of 96 and 76 imply fewer waves per SIMD than the rolled kernel; the effect was not analysed), and Dx12 code generation (DXC path).
