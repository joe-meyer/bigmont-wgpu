# Phase 0 addendum: other adapters (2026-11-24)

Code under test: commit `63ae502` (same kernels, runner and sweep as the main results doc, `2026-11-24-phase0-results.md`). Raw outputs are in `other-adapters/`. The main doc's definitions apply: **squarings/s is measured; Gmad/s is derived; tests/s is a projection** (squarings/s / 2029), not a measured full Fermat test.

## 0. Coverage matrix: every adapter on the four hosts (strict suite vs the CPU oracle)

"All 5" means the five configurations of the suite (TPI 1, 8, 16, 32 at wg 64, plus TPI 32 at wg 128 and 256), none skipped. Cooperative widths skip where an adapter has no subgroup feature or a subgroup size that does not divide the width; that is a limit of the adapter, not a failure.

| Host | Adapter (API) | Result | Raw file |
|---|---|---|---|
| Desktop (Windows) | RX 6700 XT (Vulkan) | all 5 pass | `../raw-tests-6700-vulkan.txt` |
| Desktop | RX 6700 XT (Dx12, dxcompiler.dll on PATH) | all 5 pass | `../raw-tests-6700-dx12-dxc.txt` |
| Desktop | RX 6700 XT (Dx12, default) | TPI 1 only (no SUBGROUP feature) | `../raw-6700-dx12-nodxc.txt` (sweep) |
| Desktop | RX 6700 XT (OpenGL) | TPI 1 passes; cooperative widths skip (no SUBGROUP) | `raw-tests-desktop-gl.txt` |
| Desktop | Microsoft Basic Render Driver (Dx12, CPU) | TPI 1 passes; cooperative skip | `raw-tests-warp-dx12.txt` |
| Laptop | RTX 4070 Laptop (Vulkan) | all 5 pass | `raw-tests-laptop-vulkan.txt` |
| Laptop | Radeon 780M iGPU (Vulkan) | all 5 pass | `raw-tests-laptop-vulkan.txt` |
| Laptop | llvmpipe, Ryzen 7 7840HS (Vulkan, CPU) | TPI 1 and 8 pass; 16, 32 skip (subgroup 8) | `raw-tests-laptop-vulkan.txt` |
| host2 (CT 104) | Quadro P4000 (Vulkan) | all 5 pass | `raw-tests-host2-ct104-p4000-vulkan.txt` |
| host2 (CT 104) | llvmpipe, Xeon E5-2690 v2 (Vulkan, CPU) | TPI 1 and 8 pass; 16, 32 skip (subgroup 8) | `raw-tests-host2-ct104-llvmpipe.txt` |
| host1 | GeForce 9800 GT (2008, DX10-class, nouveau) | **cannot run, not executed**: no compute-capable API (nouveau has no Vulkan; OpenGL 3.3 has no compute; DX12 needs feature level 11_0). Judged from the wgpu source and the card's generation, not from a run on host1 | none |
| host1 | Intel HD Graphics 4000 (i7-3770K iGPU) | **not testable**: disabled in the BIOS, no PCI device 00:02.0 | none |
| host1 | llvmpipe on the i7-3770K (AVX only, no AVX2) | **not run**: host1 has no Vulkan stack, Rust or test container | none |

Not covered anywhere: any Intel GPU, wave64, browsers, OpenGL on the laptop and host2 GPUs, and the Windows Dx12 default path beyond the one-thread kernel.

## 1. Where it ran

| Machine | Adapter | API | Subgroup size | Result file |
|---|---|---|---|---|
| Laptop `test-laptop` (Ubuntu 26.04) | NVIDIA GeForce RTX 4070 Laptop (8 GB, driver 580.178) | Vulkan | 32..32 | `raw-tests-laptop-vulkan.txt`, `raw-sweep-laptop-vulkan.txt` |
| same | AMD Radeon 780M iGPU (RADV, Mesa) | Vulkan | 32..64 | same files |
| same | llvmpipe (Mesa software Vulkan, LLVM 21.1.8: a CPU adapter) | Vulkan | 8..8 | same files |
| Windows desktop | Microsoft Basic Render Driver (WARP, CPU) | Dx12 | no SUBGROUP feature | `raw-tests-warp-dx12.txt` (tests only) |
| `host2` (Proxmox), new container CT 104 `bigmont-test` (Ubuntu 26.04, unprivileged LXC) | Quadro P4000 (Pascal, 8 GB, shared with three other containers; driver 580.178.04) | Vulkan | 32..32 | `raw-tests-host2-ct104-p4000-vulkan.txt`, `raw-sweep-host2-ct104-p4000-vulkan.txt` (commit `04b2049`) |

Rust 1.100.0 was installed user-level on the laptop with rustup (no sudo) to build there, and as root inside the new container on host2. CT 104 was created for this test at the owner's request with the same `/dev/nvidia*` device entries as the existing plex, frigate and llm containers. Getting the NVIDIA Vulkan driver to initialise there needed `libnvidia-gl-580` (same version as the host driver), `libvulkan1` and `libegl1` (without libegl1 the driver fails at init). `vulkaninfo` itself exits with a segfault at teardown in that container; the wgpu test and sweep binaries exited with status 0.

## 2. Correctness (the strict test suite, no skips allowed except where stated)

Each run is the `bigmont-core` suite: 133 instances of random 2029-bit moduli plus two carry-stress moduli, 3 chained squarings, compared with the CPU oracle.

| Adapter | TPI 1 | TPI 8 | TPI 16 | TPI 32 (wg 64 / 128 / 256) |
|---|---|---|---|---|
| RTX 4070 Laptop (Vulkan) | pass | pass | pass | pass / pass / pass |
| Radeon 780M (Vulkan) | pass | pass | pass | pass / pass / pass |
| Quadro P4000 (Vulkan, in CT 104 on host2) | pass | pass | pass | pass / pass / pass |
| llvmpipe (Vulkan, `BIGMONT_ALLOW_SKIP=1`) | pass | pass | skipped: subgroup size 8 is not a multiple of 16 | skipped (all three), same reason |
| WARP (Dx12, `BIGMONT_ALLOW_SKIP=1`) | pass | skipped: no SUBGROUP | skipped | skipped |

- On the RTX 4070 and the 780M no configuration skipped. Both are first-time confirmations of the cooperative kernels on an NVIDIA GPU and on an RDNA3 GPU.
- llvmpipe ran the cooperative kernel at TPI 8, which is an independent implementation (a different shader compiler and execution model) of the same WGSL, and it matched the oracle.
- The suite's "5 passed" line counts allowed skips as passes. Read the table above, not the count.

## 3. Throughput, first sweep (2 squarings per dispatch; best row per adapter; `--max-count 131072`). Superseded for speed by section 3a

| Adapter | One-thread (TPI 1), best | Best cooperative row | squarings/s | projected tests/s* |
|---|---|---|---|---|
| RTX 4070 Laptop | 467,666 (wg 128, 32,768 instances) | TPI 32, wg 256, 131,072 instances | 3,407,323 | 1,679 |
| Radeon 780M | 86,949 (wg 64, only 2,048 instances) | TPI 8, wg 256, 65,536 instances | 6,837,139 | 3,370 |
| Quadro P4000 (shared GPU; `--quick`, wg 64 only) | 75,521 (2,048 instances) | TPI 8, wg 64, 32,768 instances | 676,923 | 334 |
| llvmpipe (CPU) | 57,391 | TPI 8, wg 64, 8,192 instances | 246,666 | 122 |
| *RX 6700 XT, Windows, for reference (main doc)* | about 399,000 | TPI 16, wg 64 (Vulkan) | 4,511,000 (Vulkan), 4,690,000 (Dx12 with dxcompiler) | 2,223 / 2,312 |

\* projection, as defined in the main doc. The proposed gate in the main doc (6,000 projected tests/s) is not met by any row here. The highest printed figure is the 780M's 3,370. That is a printed figure with the caveats below, not a verdict.

Full tables are in `other-adapters/raw-sweep-laptop-vulkan.txt`.

## 3a. Re-timed with long dispatches (commits `4f4f3f8` and `92d8b5d`); supersedes section 3 and the main doc's tables for speed

Each configuration is now also timed at progressively longer dispatches (2 up to 256 squarings, stopping once one dispatch takes 0.2 s, and never planned beyond 0.4 s). "sq/s @long" is total squarings divided by the long dispatch's time; "sq/s slope" is the marginal rate between the 2-squaring and the long dispatch, which removes any fixed per-dispatch cost. Raw files: `other-adapters/amortised/` (`raw-laptop-vulkan.txt`, `raw-host2-ct104.txt`, `raw-desktop-vulkan.txt`, `raw-desktop-dx12-dxc.txt`, and the same-batch comparison `raw-laptop-fixed-count.txt`). Every row validated against the oracle first.

| Adapter | One-thread (TPI 1), squarings/s (projected tests/s*) | Best cooperative row | squarings/s @long | projected tests/s* | fixed cost per dispatch (slope) |
|---|---|---|---|---|---|
| RX 6700 XT, Vulkan (Windows) | 404,197 (199) | TPI 16, wg 64 | 4,881,358 | 2,406 | 3 to 13 ms |
| RX 6700 XT, Dx12 + dxcompiler (Windows) | 413,226 (204) | TPI 16, wg 128 | 5,021,299 | 2,475 | 2 to 8 ms |
| Radeon 780M, Vulkan (Linux, Mesa RADV) | 89,252 (44; only 2,048 instances) | TPI 8, wg 128 (wg 64 and 256 within 0.1%) | 7,996,262 | 3,941 | about 3 ms at 65,536 instances |
| RTX 4070 Laptop, Vulkan (Linux, NVIDIA) | 471,641 (232) | TPI 32, wg 128 (all wg within 0.1%) | 3,413,011 | 1,682 | 0.1 to 2.3 ms |
| Quadro P4000, Vulkan (shared, Linux) | 125,585 (62) | TPI 32, wg 256 | 948,491 | 467 | 0 to 6 ms |
| llvmpipe on Ryzen 7 7840HS (CPU) | 57,282 (28) | TPI 8, wg 64 | 250,333 | 123 | about 8 ms |
| llvmpipe in CT 104 on host2 (4 shared Xeon vCPUs) | 5,939 (3) | TPI 8, wg 64 | 22,201 | 11 | about 2 ms |

\* projected tests/s = squarings/s / 2029, a projection (see the main doc), not a measured full test.

What this changes and what it does not:
1. **The fixed-cost explanation in the first version of this addendum was wrong.** I had read the 4070's rows as floored by a fixed ~60 ms per-dispatch cost. The slope method puts the fixed cost at 0.1 to 2.3 ms on the 4070 and 0 to 6 ms on the P4000. The earlier 2-squaring figures were low by about 7 to 8% on the 6700 XT (4.51 M to 4.88 M Vulkan, 4.69 M to 5.02 M Dx12) and about 17% on the 780M (6.84 M to 8.0 M), and essentially right for the 4070 and P4000.
2. **On the 4070, the TPI 8 gap between rows was a workgroup-size effect, not batch size.** TPI 8 at wg 64 gave 3.08 M squarings/s but at wg 128 or 256 only 1.83 M (`raw-laptop-vulkan.txt`). The same-batch comparison (`raw-laptop-fixed-count.txt`, wg 64, 16,384 to 200,000 instances) is flat in batch size: TPI 8 3.0 to 3.1 M, TPI 16 2.14 to 2.16 M, TPI 32 3.39 to 3.42 M. TPI 16 is the slowest width on the 4070 at every batch size, which is unexplained. The 780M's rates are also flat from 16,384 instances up (TPI 8 about 8.0 M, TPI 16 about 7.6 M, TPI 32 about 6.2 M), once the dispatch is long enough to hide its ~6 ms fixed cost.
3. **The 780M is still about 1.6 times the RX 6700 XT** (8.0 M against 4.9 M) with the fixed-cost explanation ruled out. Remaining candidates, none tested: Mesa RADV on Linux generating better code than AMD's Windows Vulkan driver, and RDNA3 against RDNA2 integer throughput. Only a Linux run of the 6700 XT separates them.
4. **Best width differs by vendor and is stable within each:** AMD prefers TPI 8 or 16 (780M: 8 > 16 > 32; 6700 XT: 8, 16 and 32 within about 10% of each other), NVIDIA prefers TPI 32 (4070: 32 > 8 > 16; P4000: 32 > 8 > 16). Subgroup size alone does not explain it (the 4070 and 6700 XT both report subgroup 32).
5. **Against the native CPU baseline** (GMP: about 1,085 tests/s on 4 vCPUs of a 2013 Xeon, about 265 per core; section 5a): the best GPU rows are about 3.6x that 4-vCPU slice on the 780M, 2.3x on the 6700 XT, 1.5x on the 4070, and 0.4x on the shared P4000. All are projections. **No row reaches the proposed 6,000 projected tests/s gate; the highest is the 780M at 3,941 (66% of it).** That is a measured-and-projected figure, not a verdict.
6. Not re-timed with long dispatches: the desktop OpenGL adapter and WARP.

## 4. Caveats on comparing the cards (read with 3a)

1. **Fixed per-dispatch cost is not the issue** (see 3a item 1; this item originally claimed it was and was corrected after the re-timing). The remaining reason to read rows with care is that each row picks its own batch size and workgroup size, so compare rows at the same instance count and workgroup size, as `raw-laptop-fixed-count.txt` does.
2. **The Radeon 780M is about 1.6 times faster than the RX 6700 XT** although it has roughly a third of the shader count. Not explained (3a item 3). The only direct way to separate compiler from hardware is to run the 6700 XT under Linux, which has not been done.
3. **Different operating systems and drivers** (Windows with AMD and Dx12 drivers, Linux with NVIDIA proprietary and Mesa RADV), so the 6700 XT row is not like-for-like with the laptop rows.
4. **Batch sizes differ between rows** (the sweep stops growing a batch once a probe dispatch passes 50 ms), for example the 780M's one-thread rows ran only 2,048 to 4,096 instances, so one-thread-versus-cooperative ratios are not meaningful on the 780M.
5. **The best TPI differs by vendor** (3a item 4). It held across the batch sizes tried on the 4070 and 780M and across the long-dispatch re-timing, but only on these cards; it is not an established rule for other hardware.

## 5. Not tested

- **Intel GPUs: none exist on any machine used here.** The laptop's CPU is an AMD Ryzen 7 7840HS and its only graphics devices are the RTX 4070 and the AMD Radeon 780M; the Windows desktop shows only the RX 6700 XT and WARP; host2 has the P4000 and a Matrox console chip. Intel (subgroup sizes 8 to 32) therefore remains untested. The closest stand-in here is llvmpipe (subgroup 8), which exercises the small-subgroup path (TPI 8 = one instance per subgroup) but says nothing about Intel's compiler or hardware.
- The P4000 sweep was `--quick` (wg 64 only, TPI 32 skipped by the workgroup-count rule at cap 131072), run while the GPU was shared: at the start the host showed 1% utilisation, SM clock 1,202 MHz, 4.6 GB in use by the other containers' processes; at the end 14% and 1,708 MHz. The clock rise suggests the card was still ramping up, so those first P4000 rows were low-confidence. The full re-timed sweep (all widths and workgroup sizes, start state 0% utilisation, SM clock 1,202 MHz) is in section 3a: best TPI 32 at 0.95 M squarings/s, projected 467. The GPU was still shared with the other containers during that run. The correctness result is unaffected.
- RX 6700 XT under Linux (to separate compiler from hardware, item 2 above).
- Wave64 mode (no run has observed subgroup size 64).
- Intel and Apple GPUs; browsers (the cooperative kernels are still not browser-valid because naga 30 rejects `enable subgroups;` and the emitted WGSL omits it).
- CGBN on any card; a full Fermat test.
- Only llvmpipe on Linux and WARP on Windows were tried as CPU adapters. They show the kernels run without a GPU. They are not a performance fallback (llvmpipe: about 0.25 M squarings/s, projected 122 tests/s).

## 5a. A native CPU baseline (host1, Intel i7-3770K, a 2012 quad-core) — an estimate, not a run of our code

Raw: `other-adapters/raw-openssl-host1-i7-3770k.txt`. One core of the 3770K does an RSA-2048 signature in 0.915 ms (OpenSSL 3.5.9, measured). An RSA-2048 signature is two 1,024-bit modular exponentiations (CRT). A single 2,048-bit modular exponentiation with a 2,048-bit exponent, which is the shape of one Fermat test, costs about 8 times one of those (cubic scaling assumed, not measured), so roughly 3.7 ms, about 270 tests/s per core. With four cores, and hyper-threading assumed to add about 20% (also an assumption), that is **roughly 1,000 to 1,300 tests/s for the whole CPU, using optimised native code**.

**Measured with GMP (added after the estimate above).** `tools/cpu-baseline/gmp_modexp.c` times GMP 6.3.0's `mpz_powm` for one base-2 Fermat exponentiation 2^(n-1) mod n on random odd 2,029-bit n. It ran inside CT 104 on host2, on an Intel Xeon E5-2690 v2 (Ivy Bridge-EP, the same generation as the 3770K; the host was carrying a load average of about 3.7 from other guests, so this is conservative). Raw: `tools/cpu-baseline/raw-gmp-host2-ct104-xeon-e5-2690v2.txt`.
- One thread: 3.70 to 3.85 ms per test, **about 260 to 270 tests/s per core** (three runs of 100 tests; a 2,048-bit run gave 262.7).
- Four processes in parallel (the container's four vCPUs): 268 to 273 tests/s each, **about 1,085 tests/s in total**.
- This agrees with the OpenSSL-derived figure for the 3770K (3.7 ms per test), so the estimate above held up. It was not run on the 3770K itself and not on the laptop (no GMP headers there, and installing them needs sudo).
- A real test also does Gerbicz checking and runs inside Beacon's pipeline, so treat these as upper-bound reference points for a native CPU path, not as Beacon's CPU rate.

How to read it:
- It is directly comparable with the "projected tests/s" columns above (both are "one exponentiation per test"), but both sides are estimates: the GPU figures are projections from squarings/s, this one is extrapolated from RSA signatures.
- The shader path running on a CPU adapter is far slower than native code: llvmpipe on the 16-thread Ryzen 7 7840HS projects to 122 tests/s, so a 2012 quad-core running native code is roughly 8 to 10 times faster than shaders on a much newer CPU.
- Against the GPUs: the best printed projections are 2,312 (RX 6700 XT, Dx12), 3,370 (Radeon 780M) and 1,679 (RTX 4070, probably under-measured, see section 4). One old quad-core is therefore on the order of half of those. This is the most useful sanity check on the proposed 6,000 tests/s gate and on how much a GPU port buys at this size, and it argues for a first-class native CPU backend (fallback for machines with no usable GPU, and a baseline to beat) in Phase 1.
- Caveats: OpenSSL's code is constant-time and includes blinding, a real PRP test would use GMP or Beacon's own code (similar order, not identical), and it ignores Gerbicz checking.

## 6. What would settle the open questions (candidates for Phase 1 or a follow-up, not done)

1. (Done, section 3a: long-dispatch re-timing on all four GPUs and the CPU adapters. Still open: the 4070's slow TPI 16 and its wg 128/256 TPI 8 penalty.)
2. Run the 6700 XT under Linux (a USB-booted Linux or a spare machine; not available here today).
3. Re-run the P4000 sweep when the other containers are idle and with the full set of workgroup sizes (CT 104 exists for this).
4. Find an Intel GPU (a laptop or mini-PC with an Intel iGPU or Arc card) for the small-subgroup, different-compiler case.
