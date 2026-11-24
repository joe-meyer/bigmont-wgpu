# Unrolled cooperative kernel versus the rolled kernel: RX 6700 XT results

Date: 2026-11-24. Machine: the desktop with the AMD Radeon RX 6700 XT (adapter reports subgroup size 32..64). Code: commit 7654c98 (unrolled generator and variant plumbing) plus the Task 2 example changes made in this task (the `--variant` flag in `spike`, the optional `unrolled` argument in `pipeline_stats`). Raw outputs are in `docs/superpowers/results/other-adapters/unrolled/`; the rolled baseline figures come from the earlier amortised sweeps in `docs/superpowers/results/other-adapters/amortised/` (commit 4f4f3f8, same machine, taken earlier today).

This document reports numbers and says what they do and do not show. It gives no go/no-go verdict.

## 1. What was measured and what was not

Measured on this machine, this run:

- Squarings per second of the unrolled kernel (the `sq/s @long` column of the sweep: instances x squarings per dispatch / time of the longest dispatch, best of 3), on Vulkan and on Dx12 (with `dxcompiler.dll` and `dxil.dll` on `PATH` for that run only), for TPI 8, 16, 32 and workgroup sizes 64, 128, 256, at `--max-count 131070`.
- A same-batch comparison of rolled and unrolled on Vulkan at exactly 65536 instances (TPI 8, 16, 32, wg 64).
- Driver compiler statistics (registers, scratch) and instruction counts from the saved ISA of the Vulkan pipeline, for the unrolled and (re-measured today) the rolled kernel at TPI 8, 16, 32, wg 64.
- Correctness: every configuration in the sweep passed the spike's own validation (50 random instances, 3 squarings, compared with the oracle). No row printed `VALIDATION FAILED`, `REJECTED` or `bench failed`. The strict Vulkan test suite passes (11 tests in `bigmont-core`, 8 in `bigmont-oracle`, no skips).

Derived or projected (not measured):

- `tests/s = squarings/s / 2029`: one squaring per exponent bit of a base-2 Fermat test, assuming the base-2 doubling costs nothing. This is a projection, not a run of a full test. No Fermat test was run.
- `Gmad/s` in the raw files is squarings/s times a nominal 2 x 160 x 160 multiply-adds, not a count of instructions executed.

Not measured by this task:

- The RTX 4070 laptop GPU and host2 (P4000): the controller runs those. Nothing here says the speed-up carries over to NVIDIA.
- Any CPU or CUDA/CGBN baseline, any full primality test, any modulus size other than 2029 bits, any batch below 16384 instances, power, clocks or temperature.
- Which subgroup size wgpu actually used in the timed runs. The adapter reports 32..64 and the run does not record the size; the ISA numbers below are from a separate `ash` compile in which the driver reported subgroup size 64.

## 2. Speed: unrolled versus rolled

Squarings per second at the longest dispatch measured, best workgroup size per row. Rolled figures are from the amortised raw files (8 squarings per dispatch, about 215 to 245 ms per dispatch); unrolled figures are from this run (128 squarings per dispatch, 204 to 347 ms; the fixed per-dispatch cost is about 5 ms, i.e. 1% to 2% of a dispatch in both cases). Run-to-run differences of a few percent between rows of identical configuration are visible in the raw files (for example the rolled TPI 32 rows), so ratios are good to roughly that precision. Each row is a single sweep, not repeated.

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

Best unrolled versus best rolled row overall: Vulkan 82.1M versus 4.88M squarings/s (16.8x); Dx12 74.1M versus 5.02M (14.8x).

Observations (from the numbers above, causes not tested):

- In this sweep the unrolled kernel is faster in every configuration it can run. There is no configuration where it is slower or equal.
- The ordering between widths changes. The rolled kernel was nearly flat across TPI 8, 16, 32 (about 4.5M to 5.0M). The unrolled kernel falls steadily with width: TPI 8 > TPI 16 > TPI 32 on both backends, TPI 8 being about 1.7x TPI 32 on Vulkan. A plausible but untested reason is that wider instances do the same multiply-adds with more cross-lane traffic per multiply-add (the ISA below shows each cross-lane broadcast is expensive).
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

## 3. Compiler evidence (Vulkan, RX 6700 XT, TPI x, wg 64)

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
- Total branches did not go down at TPI 8 (57 to 95) but the nature changed: in the unrolled program most are `s_cbranch_execz`/`execnz` pairs belonging to loops of the form `v_readfirstlane_b32` / `v_cmpx_eq_u32` / `v_readlane_b32` / `s_andn2_wrexec_b64` / `s_cbranch_execnz`. My reading of the ISA is that the driver implements each cross-lane broadcast (`subgroupShuffle` with a computed index `base + src_lane`) as one such loop over the distinct index values in the wave (a waterfall), rather than as `ds_bpermute`; there are about two per unrolled iteration at TPI 8 (44 `v_readlane` for 20 iterations). The `down1` shuffle shows as `v_mov_b32_dpp` (22 at TPI 8). I did not execute these loops under a profiler, so how many times each waterfall loops per wave at run time (it depends on how many instances share a wave) is not measured. This is the most visible remaining overhead in the ISA and is consistent with the width ordering in section 2, but that link is a hypothesis.
- The ISA grew: 1949 lines at TPI 8 versus 760 rolled. Instruction-cache effects were not examined.
- Saved ISA files (not committed): `%TEMP%\bigmont-isa\{rolled,unrolled}_tpi{8,16,32}_wg64_ISA.cs.txt`. `pipeline_stats` file names now start with the variant.

## 4. Correctness evidence for the numbers above

- The spike validates each configuration before timing it: 50 random 2029-bit instances, 3 chained squarings, compared with the oracle (`mont_square_chain`), limbs required to be normalised. All unrolled rows passed on both backends. The sweep timings are for dispatches of up to 256 squarings, which that validation does not cover.
- To cover that, a new test `unrolled_long_chain_matches_oracle` (in `crates/bigmont-core/tests/chain.rs`) runs the unrolled kernel at TPI 8, 16, 32 (wg 64) on 72 instances (70 random plus the two carry-stress moduli) for 128 chained squarings and compares with the oracle. It passes on Vulkan. Dx12 was not run through the test suite in this task; the Dx12 rows rely on the 3-squaring spike validation.
- TPI 8 and 16 at wg 128 and 256 were validated only by the spike's 3-squaring check; the test suite covers wg 128 and 256 only for TPI 32.
- The nominal Gmad/s of the unrolled TPI 8 rows (about 4.2 x 10^12 multiply-adds per second) is a derived figure; it is roughly 60% of a spec-sheet figure for this card that I computed from memory (2560 lanes at about 2.6 GHz), which was not verified here. If that is right, there is limited headroom left on raw multiply-add issue in this kernel as written, but this is an estimate and not a measurement.

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

Raw files: `other-adapters/unrolled/raw-desktop-vulkan.txt`, `raw-desktop-dx12-dxc.txt`, `raw-desktop-vulkan-fixed-count-65536-rolled.txt`, `raw-desktop-vulkan-fixed-count-65536-unrolled.txt`. Each starts with a line giving the commit, backend, variant, time and command; rows now carry a leading `variant` column, and the header block has a `variant:` line, so the files are self-describing.
