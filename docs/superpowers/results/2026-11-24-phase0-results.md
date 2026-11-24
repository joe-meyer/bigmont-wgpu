# Phase 0 speed spike: results on the RX 6700 XT

Date: 2026-11-24. Spec: `docs/superpowers/specs/2026-11-24-bigmont-wgpu-design.md`, section 4.
Code under test: `crates/bigmont-core` (kernels, runner) and `crates/bigmont-core/examples/spike.rs` (sweep), at the commit that adds the `--max-count` flag.

## Summary

- One 2,029-bit Montgomery squaring (13-bit limbs, 160 limbs) on the RX 6700 XT runs at about **4.5 to 4.7 million squarings/s** in the best cooperative configuration (16 lanes per number, workgroup 64), on both Vulkan and Dx12.
- That projects to about **2,200 to 2,300 "tests/s"** (squarings/s divided by 2,029). This is a projection, not a measured test.
- Cooperative (subgroup) kernels beat one-thread-per-number by about 11x on Vulkan. On Dx12 with its default compiler the one-thread kernel is 4x faster than on Vulkan, so there the ratio is about 2.8x.
- Against Beacon's published 516 to 523k tests/s per card, the projection is about 230x lower. Order of magnitude only; the card is different (much smaller) and I have not confirmed what Beacon's figure counts (see section 6).
- Whether this clears the gate is the user's decision. A proposed threshold is in section 8.

## 1. Machine and adapters

| Item | Value |
|---|---|
| CPU | AMD Ryzen 7 5800X |
| OS | Windows 11 Pro 10.0.26200 |
| GPU | AMD Radeon RX 6700 XT, driver 32.0.21045.16010 (Windows driver date 2026-10-15) |
| Rust / wgpu | rustc 1.100.0, wgpu 30.0.2 (naga 30) |
| Dxc (Dx12 cooperative runs only) | `C:\Program Files (x86)\Windows Kits\10\Redist\D3D\x64\dxcompiler.dll` (file version 1.9.2602.17), prepended to `PATH` for those runs. Already on the machine; nothing was downloaded or installed. |

`spike -- --list` (the raw files record only the adapter line for the adapter each run used):

```
AMD Radeon RX 6700 XT [Vulkan, DiscreteGpu] subgroup 32..64 subgroups=true
AMD Radeon RX 6700 XT [Dx12, DiscreteGpu] subgroup 32..64 subgroups=false
Microsoft Basic Render Driver [Dx12, Cpu] subgroup 4..4 subgroups=false
AMD Radeon RX 6700 XT [Gl, Other] subgroup 4..128 subgroups=false
```

- The 6700 XT appears under Vulkan, Dx12 and Gl. Swept: Vulkan and Dx12. Gl was not swept (no subgroup feature, and Gl is not a target backend).
- `adapter.get_info()` reports subgroup size **min 32, max 64** (RDNA2 can run wave32 or wave64).
- **Observed** subgroup size: 32 on both Vulkan and Dx12. This comes from a throwaway kernel reading the `subgroup_size` built-in at Task 3 (workgroups of 32/64/128/256), not from this task's runs; I did not re-observe it here. Wave64 was never observed, so the case "two 32-lane instances in one 64-lane subgroup" (TPI 32 under wave64) has not been exercised.
- Dx12 exposes `Features::SUBGROUP` only when `dxcompiler.dll` and `dxil.dll` are discoverable. Without them wgpu reports `subgroups=false` for the Dx12 adapter and all cooperative configurations are skipped (section 3).

## 2. Results

Method, from `spike.rs`: each configuration is first validated against the CPU oracle (50 random instances, 3 squarings, limbs normalised, value equal mod n). A configuration that fails validation prints no speed. Then the batch is doubled from about 1,024 instances until a probe dispatch takes at least 50 ms or the batch reaches the cap; the reported time is the best of 5 dispatches of 2 squarings each. Every row below has status `ok`, i.e. it passed validation before it was timed. All dispatches stayed far below 2 s (the slowest best-of-5 time was about 87 ms).

Columns:

- `instances`: batch size (numbers squared in parallel), `sq/d`: squarings per dispatch, `best ms`: best of 5.
- `squarings/s` = instances x sq/d / best time. Measured.
- `Gmad/s` is **derived**, not a hardware counter: squarings/s x 2 x 160^2 / 1e9. It counts 51,200 limb multiply-adds per squaring (schoolbook product plus Montgomery reduction, no use of squaring symmetry), so it is "useful" multiply-adds only.
- `tests/s*` is a **projection**: squarings/s / 2029, assuming one squaring per exponent bit of a base-2 Fermat test with the doubling folded into the reduction. It ignores windowed multiplies, the Gerbicz check, and any host-side work, so a real full test would run somewhat slower than this.

### 2a. Vulkan, cap 65,536 instances (the as-reviewed default). Raw: `raw-6700-vulkan.txt`

```
 tpi   wg instances  sq/d   best ms  squarings/s     Gmad/s     tests/s*  status
   1   64     16384     2     86.45       379052       19.4          187  ok
   1  128     16384     2     86.94       376883       19.3          186  ok
   1  256     16384     2     87.52       374409       19.2          185  ok
   8   64     65536     2     31.62      4144608      212.2         2043  ok
   8  128     65536     2     34.69      3778304      193.4         1862  ok
   8  256     65536     2     35.52      3690599      189.0         1819  ok
  16   64     65536     2     31.69      4135703      211.7         2038  ok
  16  128     65536     2     31.86      4113663      210.6         2027  ok
  16  256     65536     2     32.57      4023810      206.0         1983  ok
  32   64     65536     2     34.21      3831854      196.2         1889  ok
  32  128     65536     2     31.64      4143220      212.1         2042  ok
  32  256     65536     2     33.82      3875164      198.4         1910  ok
```

### 2b. Dx12 with dxcompiler on PATH, cap 65,536. Raw: `raw-6700-dx12.txt`

```
 tpi   wg instances  sq/d   best ms  squarings/s     Gmad/s     tests/s*  status
   1   64     16384     2     83.91       390530       20.0          192  ok
   1  128     16384     2     80.56       406759       20.8          200  ok
   1  256     16384     2     86.21       380076       19.5          187  ok
   8   64     65536     2     31.97      4100203      209.9         2021  ok
   8  128     65536     2     32.67      4011864      205.4         1977  ok
   8  256     65536     2     33.47      3915730      200.5         1930  ok
  16   64     65536     2     30.81      4254880      217.8         2097  ok
  16  128     65536     2     31.06      4219364      216.0         2080  ok
  16  256     65536     2     31.23      4196439      214.9         2068  ok
  32   64     65536     2     31.91      4107179      210.3         2024  ok
  32  128     65536     2     31.87      4113134      210.6         2027  ok
  32  256     65536     2     31.90      4108905      210.4         2025  ok
```

### 2c. Dx12 without dxcompiler (default compiler, FXC), cap 65,536. Raw: `raw-6700-dx12-nodxc.txt`

Only the one-thread rows run; all cooperative rows are skipped (`adapter lacks wgpu::Features::SUBGROUP`).

```
 tpi   wg instances  sq/d   best ms  squarings/s     Gmad/s     tests/s*  status
   1   64     65536     2     79.17      1655612       84.8          816  ok
   1  128     65536     2     78.64      1666732       85.3          821  ok
   1  256     65536     2     79.09      1657316       84.9          817  ok
   8..32  (all workgroup sizes)  skipped: adapter lacks wgpu::Features::SUBGROUP
```

Notable and not explained: the one-thread kernel is about 4.2x faster under Dx12/FXC (1.66 M squarings/s) than under Vulkan or Dx12/DXC (0.38 to 0.41 M). The same WGSL and the same driver; only the shader compiler path differs. I did not investigate the cause (register allocation, loop unrolling and spilling are candidates, none checked). It means the one-thread number is compiler-dependent, and so is any "cooperative versus one-thread" ratio.

### 2d. Cap raised to 131,072 instances (`--max-count 131072`)

The reviewers noted that the cooperative rows in 2a and 2b hit the 65,536 cap while a dispatch still took only about 31 ms (under the 50 ms target), so those figures were lower bounds. Raw: `raw-6700-vulkan-cap131072.txt`, `raw-6700-dx12-cap131072.txt`. Same validation, all rows `ok`. At 131,072 instances the cooperative dispatches take 56 to 64 ms, so the 50 ms target is now reached. The one-thread rows are unchanged because they reach 50 ms at 16,384 instances.

Vulkan:

```
 tpi   wg instances  sq/d   best ms  squarings/s     Gmad/s     tests/s*  status
   8   64    131072     2     62.35      4204489      215.3         2072  ok
   8  128    131072     2     62.74      4178459      213.9         2059  ok
   8  256    131072     2     64.27      4078818      208.8         2010  ok
  16   64    131072     2     58.11      4511029      231.0         2223  ok
  16  128    131072     2     58.18      4506089      230.7         2221  ok
  16  256    131072     2     58.72      4464001      228.6         2200  ok
  32   64  skipped: 65536 workgroups at --max-count 131072 exceeds 65535 per dispatch
  32  128    131072     2     61.91      4234276      216.8         2087  ok
  32  256    131072     2     62.31      4206979      215.4         2073  ok
```

Dx12 (dxcompiler on PATH):

```
 tpi   wg instances  sq/d   best ms  squarings/s     Gmad/s     tests/s*  status
   8   64    131072     2     58.92      4449182      227.8         2193  ok
   8  128    131072     2     58.51      4480198      229.4         2208  ok
   8  256    131072     2     60.49      4333618      221.9         2136  ok
  16   64    131072     2     55.89      4690499      240.2         2312  ok
  16  128    131072     2     56.39      4648776      238.0         2291  ok
  16  256    131072     2     56.68      4625333      236.8         2280  ok
  32   64  skipped: 65536 workgroups at --max-count 131072 exceeds 65535 per dispatch
  32  128    131072     2     59.14      4432293      226.9         2184  ok
  32  256    131072     2     58.00      4519810      231.4         2228  ok
```

(The TPI 1 rows of these two runs are in the raw files; they are within noise of 2a and 2b: 0.377 to 0.399 M squarings/s.)

Effect of the cap on the cooperative best-of-row figures:

| Backend | TPI 8 (wg 64) | TPI 16 (wg 64) | TPI 32 (best wg) |
|---|---|---|---|
| Vulkan, cap 65,536 | 4.14 M | 4.14 M | 4.14 M (wg 128) |
| Vulkan, cap 131,072 | 4.20 M (+1%) | 4.51 M (+9%) | 4.23 M (+2%, wg 128) |
| Dx12, cap 65,536 | 4.10 M | 4.25 M | 4.11 M (wg 128) |
| Dx12, cap 131,072 | 4.45 M (+9%) | 4.69 M (+10%) | 4.52 M (+10%, wg 256) |

**Did the ranking change?** In the as-reviewed Vulkan run the three cooperative widths were a three-way tie (4.14 M for TPI 8/wg 64, TPI 16/wg 64 and TPI 32/wg 128). With the raised cap, TPI 16 is first on both backends: it leads the next-best width by about 6.5% on Vulkan (4.51 vs 4.23 M) and about 4% on Dx12 (4.69 vs 4.52 M). Absolute numbers rose by up to about 10%. So the cap did matter for the numbers, and it turned a tie into a modest lead for TPI 16. The lead is small. I ran each configuration once per process and did not measure run-to-run variance; the one-thread rows, which are the same configuration repeated across runs, vary by about 3 to 6% (0.374 to 0.407 M), so a 4 to 6% lead is within about one to two noise widths and I would not call it established. Whether even larger batches would raise the numbers further is not known (the 262,144 ceiling was not run; TPI 16/wg 64 and TPI 32/wg 64 would be skipped by the 65,535-workgroup rule there).

## 3. Skipped and failed configurations

- Dx12 without dxcompiler: all 9 cooperative configurations skipped, `adapter lacks wgpu::Features::SUBGROUP` (one line: Dx12 on this machine runs only the one-thread kernel, at 1.66 M squarings/s, unless `dxcompiler.dll` and `dxil.dll` are discoverable).
- TPI 32 with workgroup 64 at cap 131,072 on both backends: skipped by the sweep, because 131,072 / 2 instances per workgroup = 65,536 workgroups, above the 65,535 per-dispatch limit. It ran at the default cap.
- No other configuration was skipped. No configuration failed validation and no benchmark errored.
- Not swept: the Gl adapter, the Microsoft Basic Render Driver (CPU), and the workgroup-memory cooperative backend (backend 2 in the spec). Backend 2 was not needed since subgroups are available on Vulkan and on Dx12 with dxcompiler; it was not built.

## 4. Efficiency against the multiply-add ceiling

Ceiling, from the user's gapsieve project (`cargo run --release --example mul_bench`, run only, nothing modified or committed there; the device was created with `SHADER_INT64` successfully on this card; the tool prints the adapter name, `AMD Radeon RX 6700 XT`, but not which backend it used):

```
mad_u32_u24 (24x24+32)       48.04 ms     2861 G chain-steps/s
```

Other lines for context: add only 2438, mul_hi_u32 1081, mul_lo_u32 1126, mad_u32_u16 2909, fma_f32 3010 G chain-steps/s. I read one chain-step of the `mad_u32_u24` kernel as one 24x24+32 multiply-add, so the ceiling is about **2,861 Gmad/s**. This is the measured peak of the one instruction the 13-bit-limb kernel is built from; it is not a vendor specification.

| Best configuration | Gmad/s (derived) | Fraction of 2,861 |
|---|---|---|
| Vulkan, TPI 16 wg 64, cap 131,072 | 231.0 | 8.1% |
| Dx12 + dxcompiler, TPI 16 wg 64, cap 131,072 | 240.2 | 8.4% |
| Vulkan, as reviewed (cap 65,536), best | 212.2 | 7.4% |
| Dx12 FXC, one-thread | 85.3 | 3.0% |
| Vulkan, one-thread | 20.4 | 0.7% |

So the best kernel uses roughly 8% of the multiply-add peak. Why it is not higher was not diagnosed (candidates: the carry-normalisation passes, subgroup shuffles, the non-multiply instructions per multiply-add, and the fact that TPI 8, 16 and 32 land within about 10% of each other, which suggests the limit is not the lane count per number). That is a profiling question for later phases.

## 5. Best configuration per backend, and cooperative versus one-thread

Using the raised cap (the cooperative rows were lower bounds without it):

| Backend | Best configuration | Squarings/s | Projected tests/s* | Best one-thread | Ratio |
|---|---|---|---|---|---|
| Vulkan | TPI 16, wg 64 | 4.51 M | 2,223 | 0.399 M (wg 256) | 11.3x |
| Dx12 + dxcompiler | TPI 16, wg 64 | 4.69 M | 2,312 | 0.407 M (wg 128, cap 65,536 run) | 11.5x |
| Dx12, default compiler | TPI 1 (only option), wg 128 | 1.67 M | 821 | 1.67 M | 1x (cooperative unavailable) |

Against the Dx12/FXC one-thread kernel, the best cooperative configuration (Dx12 + dxcompiler) is 2.8x faster. The cooperative TPI 8, 16 and 32 kernels are all within about 10% of each other on both backends. The one-thread rows were each measured at only the batch size where one dispatch first exceeds 50 ms (16,384 instances for Vulkan and Dx12/DXC); a larger batch might help latency hiding, and was not tried.

With the as-reviewed cap (65,536), the best cooperative figures were 4.14 M on Vulkan (2,043 tests/s) and 4.25 M on Dx12 (2,097 tests/s).

## 6. Comparison with Beacon's figure

Beacon's README (re-read with `gh api repos/djuk8/beacon/readme`) says, of its GPU Fermat driver `cuda/g1_fermat_v2.cu`: "a Montgomery-arithmetic kernel with a Gerbicz error check on every test (zero alarms across 18.5M+ verdicts); measured 516-523k tests/s/card at the 2048-bit class". The task brief names this as per RTX 5090; that line itself says "per card" (elsewhere the README discusses RTX 5090 boxes, so the 5090 reading is likely but I did not confirm it from that sentence).

Ratios, measured projection against that figure: 2,300 / 516,000 is about 1/225, so the projection here is roughly **two to three orders of magnitude lower, about 2.4**. Treat that as an order of magnitude and nothing finer, because:

- It is a different, much larger card (an RTX 5090, with several times the shader count of a 6700 XT; I did not look up the exact numbers and have not verified them).
- I have not confirmed whether Beacon's figure counts full Fermat tests or early-exit tests (the spec read it as possibly early-exit; the README wording about a Gerbicz check on every test hints at full tests but does not settle it).
- Beacon's "2048-bit class" may not be 2,029-bit exactly, and CGBN (32-bit limbs, 64-bit products) was not run on any card here.
- Our "tests/s" is itself a projection (section 2).

## 7. What was not measured

- The RTX 4070 laptop. Nothing was run on it.
- CGBN or any CUDA code, on any card. The comparison in section 6 is to a published number only.
- A full Fermat test. "tests/s" is squarings/s divided by 2,029. There is no windowed exponentiation, no Gerbicz check, no host round trip in these numbers.
- Wave64 behaviour. The observed subgroup size was 32 on both backends; TPI 32 with two instances per subgroup was not exercised.
- Browser behaviour, including browser subgroup support. The cooperative kernels are **not browser-valid yet**: naga 30 rejects the `enable subgroups;` directive as unimplemented, so the emitted cooperative WGSL omits it and relies on `wgpu::Features::SUBGROUP`. Browsers require the directive. The browser build needs to emit it once the toolchain accepts it (marked `TODO(browser)` in `kernel.rs`).
- Run-to-run variance across processes, thermal state, and batches above 131,072 instances.
- Workgroup-memory cooperative kernels (backend 2) and any adapter other than the 6700 XT (Intel, Apple, NVIDIA).
- Why the one-thread kernel is 4x faster under Dx12/FXC than under Vulkan or Dx12/DXC.

## 8. Proposed go/no-go threshold (PROPOSED, for the user to accept or change)

This is a proposal, not a decision.

Proposed gate: **at least 6,000 projected tests/s on the RX 6700 XT** (about 12.2 M squarings/s, about 620 Gmad/s, about 22% of the measured `mad_u32_u24` ceiling).

Derivation, with every assumption visible:

1. Beacon's published figure: about 516,000 tests/s per card (RTX 5090 class, meaning unconfirmed, section 6).
2. Scale to the 6700 XT by shader count only. An RTX 5090 has about 21,760 shader lanes and a 6700 XT has 2,560, a ratio of about 8.5 (figures from memory, not verified here, and clocks and instruction mix are ignored). 516,000 / 8.5 is about 61,000 tests/s: what a per-lane-equal port of Beacon's kernel would give.
3. Allow a portable WGSL engine to be at best a tenth as efficient as hand-tuned CUDA with CGBN: 61,000 / 10 is about 6,000.

Where the current numbers stand against this proposal: the best is about 2,300 projected tests/s, about 38% of 6,000, so about 2.6x short. It would pass a gate of about 2,000 and fail one of about 6,000. Directional notes on the assumptions: if Beacon's figure is an early-exit figure, the true full-test parity target is lower and the gate is easier; if the projection ignores real overheads (windowed multiplies, Gerbicz), the real test rate is lower and the gap is larger. The 10x allowance in step 3 is a judgement, not a derived quantity.

## 9. Decisions only the user can make

- Whether these results clear the gate, and what the gate should be (section 8).
- Whether to run the same sweep on the RTX 4070 laptop. Copy the repo there and run `cargo run --release -p bigmont-core --example spike` (add `-- --max-count 131072`); no CUDA is needed for that run. A direct CGBN run would need the CUDA toolkit (several GB), which needs the user's consent.

## Files

- Raw outputs, this directory: `raw-6700-vulkan.txt`, `raw-6700-dx12.txt`, `raw-6700-dx12-nodxc.txt`, `raw-6700-vulkan-cap131072.txt`, `raw-6700-dx12-cap131072.txt`.
- Commands: `BIGMONT_ADAPTER=6700 BIGMONT_BACKEND=vulkan|dx12 spike [--max-count 131072]`; Dx12 + dxcompiler runs have the Windows Kits redist directory prepended to `PATH`.
