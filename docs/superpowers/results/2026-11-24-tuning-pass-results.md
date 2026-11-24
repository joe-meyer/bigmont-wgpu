# Tuning pass, Task 1: cheaper Montgomery factor and TPI 4 (2026-11-24)

Machine: AMD Radeon RX 6700 XT (Vulkan and Dx12 with DXC), subgroup 32..64. Same sweep settings as the earlier verified rows: `spike --variant unrolled --max-count 131070`, every timed output compared with the oracle (no row failed, none printed `FAILED`, `REJECTED` or `TIMED OUTPUT FAILED ORACLE`). Laptop and host2 GPUs are not covered here (the controller runs those).

Raw outputs: `other-adapters/tuning/raw-desktop-vulkan.txt`, `raw-desktop-dx12-dxc.txt` (final kernel), `raw-desktop-vulkan-literal-a.txt` (intermediate kernel, see below), ISA for TPI 4 and 8 in `other-adapters/tuning/unrolled_tpi{4,8}_wg64_wgpu.isa.txt`. "Before" rows are `other-adapters/unrolled/raw-desktop-vulkan-verified.txt` and `raw-desktop-dx12-dxc-verified.txt` (previous commit, same machine).

## What changed

1. Montgomery factor `m = ((p & MASK) * ninv) & MASK` (was `(p * ninv) & MASK`), plus `ninv` masked with `& MASK` once where it is loaded (it is already below 2^13, so values do not change). The second part was not in the task text. With only the first part the compiler cannot see that `ninv` is small and kept the full 32-bit low multiply (`v_mul_lo_u32` count unchanged) while adding a `v_and_b32` per step. That intermediate kernel was swept on Vulkan and is slower than before (table B). With the load-time mask the compiler emits `v_mad_u32_u16` for the factor and `v_mul_lo_u32` drops to the single address multiply (table D).
2. TPI 4 (40 limbs per lane) for both variants. Unrolled generator: light passes every 20 steps at every width. LPL 40: a pass inside the unrolled body after step 19 and after step 39 (no runtime `owner % K` test). LPL 20: one pass per owner iteration (K = 1, runtime test as before). LPL 10 and 5: runtime `(owner % K) == K - 1` with K = 2 and 4. `PERIOD` is now 20 (was 24, which only capped K; effective spacing was already 20). The rolled kernel already works at TPI 4 unchanged (its pass spacing is 16 iterations at any width), so rolled TPI 4 is allowed and tested.

## A. Correctness

Strict tests on this GPU (Vulkan): 13 passed in `bigmont-core` including `tpi4_matches_oracle` (rolled), `tpi4_unrolled_matches_oracle` (wg 64, 128, 256), TPI 4 in `unrolled_long_chain_matches_oracle` (128 squarings) and the validation test; 8 passed in the oracle crate; no skips.

## B. Sweep, Vulkan, squarings/s at the longest dispatch (`sq/s @long`), millions

| tpi | wg | before | after literal (a) only | change | final (a) | change |
|---|---|---|---|---|---|---|
| 8 | 64 | 81.77 | 80.78 | -1.2% | 86.23 | +5.5% |
| 8 | 128 | 81.63 | 80.59 | -1.3% | 86.23 | +5.6% |
| 8 | 256 | 81.52 | 80.38 | -1.4% | 84.04 | +3.1% |
| 16 | 64 | 63.92 | 62.31 | -2.5% | 67.13 | +5.0% |
| 16 | 128 | 63.00 | 62.32 | -1.1% | 67.07 | +6.5% |
| 16 | 256 | 64.14 | 62.45 | -2.6% | 65.48 | +2.1% |
| 32 | 64 | 48.38 | 46.58 | -3.7% | 51.46 | +6.4% |
| 32 | 128 | 48.18 | 47.06 | -2.3% | 51.44 | +6.8% |
| 32 | 256 | 48.30 | 46.50 | -3.7% | 50.45 | +4.5% |

Note: the literal change alone (without masking `ninv` at load) is 1% to 4% slower than before, outside the roughly 1% run-to-run noise. Part of the "final" gain is a longer dispatch: TPI 8 now ran 256 squarings per dispatch (389 ms) where before it ran 128 (205 ms), so less fixed cost is amortised away; the marginal-rate column gives +4.4% (wg 64) for TPI 8, +5.0% for TPI 16 and +6.5% for TPI 32, so the effect stays.

## C. Sweep, Dx12 (DXC), millions

| tpi | wg | before | final | change |
|---|---|---|---|---|
| 8 | 64 | 74.45 | 74.27 | -0.2% (noise) |
| 8 | 128 | 75.45 | 75.97 | +0.7% (noise) |
| 8 | 256 | 73.99 | 73.45 | -0.7% (noise) |
| 16 | 64 | 53.37 | 55.21 | +3.4% |
| 16 | 128 | 54.41 | 55.22 | +1.5% |
| 16 | 256 | 53.40 | 55.75 | +4.4% |
| 32 | 64 | 38.50 | 39.57 | +2.8% |
| 32 | 128 | 38.49 | 39.59 | +2.9% |
| 32 | 256 | 38.51 | 39.51 | +2.6% |

On Dx12 the cheaper factor gives nothing measurable at TPI 8 and 1.5% to 4.4% at TPI 16 and 32 (the Dx12 sweep for the literal-only kernel was not run).

## D. TPI 4 against TPI 8 (same run, final kernel), millions of squarings/s

| backend | wg | TPI 4 | TPI 8 | TPI 4 vs TPI 8 |
|---|---|---|---|---|
| Vulkan | 64 | 96.21 | 86.23 | +11.6% |
| Vulkan | 128 | 95.56 | 86.23 | +10.8% |
| Vulkan | 256 | 94.77 | 84.04 | +12.8% |
| Dx12 | 64 | 87.53 | 74.27 | +17.8% |
| Dx12 | 128 | 87.37 | 75.97 | +15.0% |
| Dx12 | 256 | 75.73 | 73.45 | +3.1% |

TPI 4 is faster than TPI 8 at every row and is now the best unrolled row on both backends. Best unrolled row before: Vulkan 81.77M (TPI 8, wg 64), Dx12 75.45M (TPI 8, wg 128). Now: Vulkan 96.21M (TPI 4, wg 64), Dx12 87.53M (TPI 4, wg 64). Dx12 TPI 4 at wg 256 drops to 75.7M (long dispatch 128 squarings; marginal rate 77.6M agrees, so it is not an amortisation artefact); the cause was not investigated. TPI 4 uses 192 VGPRs (see E), so a large workgroup likely limits occupancy; that is a guess, not measured. With `tests/s*` = squarings/s / 2029 (a projection, not a measured Fermat test), the best Vulkan row projects to about 47.4k tests/s.

## E. ISA (Vulkan driver, wgpu mode, wg 64), counts of mnemonics in the saved ISA

| tpi | VGPRs | scratch B | v_mad_u32_u24 | v_mul_lo_u32 | v_mul_u32_u24 | v_mad_u32_u16 | v_and_b32 | instr. |
|---|---|---|---|---|---|---|---|---|
| 4 final | 192 | 0 | 3280 | 1 | 0 | 41 | 245 | 4606 |
| 8 final | 96 | 0 | 840 | 1 | 0 | 21 | 105 | 1552 |
| 16 final | 76 | 0 | 220 | 1 | 0 | 11 | 55 | 661 |
| 32 final | 45 | 0 | 60 | 1 | 0 | 6 | 29 | 378 |
| 4 literal (a) | 192 | 0 | 3280 | 41 | 0 | 0 | 244 | 4599 |
| 8 literal (a) | 96 | 0 | 840 | 21 | 0 | 0 | 104 | 1579 |
| 8 before | n/a | n/a | 840 | 21 | 0 | 0 | 84 | 1552 |
| 16 before | n/a | n/a | 220 | 11 | 0 | 0 | 44 | 656 |
| 32 before | n/a | n/a | 60 | 6 | 0 | 0 | 23 | 376 |

("before" counts are from the saved ISA of the previous commit; its VGPR figures were not saved. The final kernel's remaining `v_mul_lo_u32` is the address computation, `ic * 160`.) The factor multiply per step changed from a quarter-rate `v_mul_lo_u32` to a `v_mad_u32_u16`. TPI 4 has no scratch and no spill at 192 VGPRs, but it is 4.6k instructions (code size 293 KB of ISA text), about three times TPI 8.

## F. Statements

- Measured: the literal `((p & MASK) * ninv) & MASK` change alone does not remove the 32-bit multiply and is 1% to 4% slower on Vulkan. With `ninv` masked at load it removes it; the gain is 2% to 7% on Vulkan, about 3% at TPI 16/32 and within noise at TPI 8 on Dx12.
- Measured: TPI 4 beats TPI 8 by 11% to 13% on Vulkan and by 15% to 18% on Dx12 at wg 64 and 128 on this GPU; at Dx12 wg 256 the margin is 3%.
- Not measured: the browser. The generated source adds no new WGSL constructs (the kernel is still parsed and validated by naga in the tests), but it was not run with the `enable subgroups;` and `diagnostic(off, subgroup_uniformity);` lines in front, since naga 30 rejects the `enable` line. The light passes inside the loop body use the same subgroup shuffles as before, in uniform control flow.
- Other GPUs (laptop 780M, host2 P4000) have not been run for this change.
