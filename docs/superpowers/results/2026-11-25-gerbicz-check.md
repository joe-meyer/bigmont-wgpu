# Gerbicz check (`--gec`): identity, fault drill and throughput cost (2026-11-25)

Issue #17, branch `feat/gerbicz-check`. This is the gate of the plan: verdicts with and without `--gec` must be
byte-identical on clean runs, the drill must turn the injected candidate into verdict 2 and leave the rest alone, and the
throughput cost is measured and compared with the CUDA driver's figure. We compare; we do not claim a match.

## Setup

- GPU: AMD Radeon RX 6700 XT (discrete), AMD proprietary driver 26.10.1, Vulkan backend (`--vk-backend Vulkan`, `--backend gpu`).
  The DX12 backend of this adapter has no subgroups, so it was not used.
- OS: Windows 11 Pro (build 26200). Rust 1.100.0, release build of the branch head at the time (commit 8169294, clean tree).
- Tuner pick (auto, 2,029-bit candidates in a 2,048-bit container): 13-bit limbs, 157 limbs, TPI 8, workgroup 128, measured
  74.8 M squarings/s by the tuner (the profile run in section 4 tuned on its own and printed 160 limbs at the same TPI and workgroup). With `--limbs 32` the tuner picked 32-bit limbs, TPI 4,
  workgroup 256, 51.2 M squarings/s.
- Tuning cache: one cache file for every plain and `--gec` run, created by one warm-up run before any timing.
- Candidates: `make_candidates` (2,029-bit odd numbers in a 2,048-bit container, plus 3 probable primes appended).
  - `gec_20k`: 20,000 random numbers, seed 1 (20,003 candidates, 28 probable primes).
  - `gec_100k`: 100,000 random numbers, seed 1 (100,003 candidates, 144 probable primes).
  The plan suggested 20,000 candidates; a plain run of that takes about 0.6 s, too short for a stable figure, so the
  throughput runs use 100,000 (about 3 s plain). Identity was checked on both files.

## 1. Identity: `--gec` verdicts equal plain verdicts

```sh
bigmont gec_20k.bin v_plain.bin --backend gpu --vk-backend Vulkan --cache tune.cache
bigmont gec_20k.bin v_gec.bin   --backend gpu --vk-backend Vulkan --cache tune.cache --gec
cmp v_plain.bin v_gec.bin
```

- `gec_20k`, 13-bit limbs: `cmp` reports no difference; both runs print `GEC alarms 0`. The file holds 19,975 zeros and 28
  ones (the last three bytes, the appended probable primes, are 1), so both verdicts are present.
- The same verdicts equal an independent `--backend cpu` run on the same file (`cmp` clean).
- `gec_100k`, 13-bit limbs: the three plain runs and the three `--gec` runs of the throughput table (below) were
  written to files; plain run i and `--gec` run i are byte-identical for i = 1, 2, 3, all with `GEC alarms 0` and 144 probable
  primes.
- `--limbs 32` (the adapter supports int64) on `gec_20k`: plain and `--gec` verdict files identical, equal to the CPU run,
  `GEC alarms 0`.
- Canary: `bigmont canary gec_20k.bin v_cpu.bin` (the expected file is the CPU backend's verdicts, so it is independent of
  the GPU code) passes with and without `--gec` ("canary ok: all 20003 verdicts match"). The Beacon repository's own
  canary files were not available on this machine, so they were not run.

## 2. Fault drill

These runs used a build with a `--inject IDX BDRY` command-line flag (a test hook on the engine) that was removed from the
branch afterwards, so the table below cannot be reproduced with the final code. The same detection (a bit flipped in a
block or in the tail recomputation, with both limb schemes, and a hit probable prime) is now exercised by the runner tests in
`crates/bigmont-core/tests/gec.rs`. They build the runner from mutated kernel sources that simulate a faulty device
(register, memory-read, multiply-unit and memory-cell faults, plus a modulus read wrong on every load), so no fault
handling remains in the runner. The engine's alarm handling is covered by unit tests in `crates/bigmont-cli/src/engine.rs`.
The numbers are kept as measured.

Three candidates of 2,029 bits taken from `gec_20k`: two random composites and one probable prime (truth: 0, 0, 1).
`--gec` alone sends three candidates to the CPU (below the GPU threshold of 64), which was unchecked when this drill
was run (the CPU path got the same check later, see the note at the end of this section); `--inject` sets the threshold
to one candidate, so the GPU runs.

| Command (after `bigmont drill3.bin v.bin --gec ...`, Vulkan, 13-bit limbs) | Verdict bytes | `GEC alarms` |
|---|---|---|
| (none; runs on the CPU) | 0, 0, 1 | 0 |
| `--inject 1 3` | 0, **2**, 1 | 1 |
| `--inject 1 0` | 0, **2**, 1 | 1 |
| `--inject 0 49` (last full block) | **2**, 0, 1 | 1 |
| `--inject 2 50` (BDRY = (2029 - 1) / 40 = 50, the tail recomputation, on the probable prime) | 0, 0, **2** | 1 |
| `--inject 1 50` (tail) | 0, **2**, 1 | 1 |
| `--inject 1 51` (past the tail: injects nothing) | 0, 0, 1 | 0 |

Only the injected candidate changes, and a tail injection is caught as well as a block injection. A probable prime that is
hit (`--inject 2 50`) comes out as 2 and is not re-checked on the CPU (rechecks 0, PRP 0), which is the contract.
With `--limbs 32`, `--inject 1 3` and `--inject 1 50` both gave 0, 2, 1 with one alarm.

Later change: the CPU path (`--backend cpu`, small buckets, buckets after a GPU error, and the CPU re-checks of GPU
verdicts) now runs the same check under `--gec` (`bigmont_cpu::fermat_base2_gec`). Its fault tests run without a GPU in
`crates/bigmont-cpu/tests/cpu.rs`. No drill or throughput figure in this file was re-measured for it.

## 3. Throughput

`gec_100k`, 100,003 candidates, Vulkan, 13-bit limbs TPI 8 workgroup 128, warm cache (warm-up run first, so no run is discarded),
nothing else running (no builds or tests during the six runs). The figure is "tests/s excluding tuning" from the summary
line. Plain and `--gec` alternated, plain first.

| Run | plain (tests/s) | `--gec` (tests/s) | cost of the pair |
|---|---|---|---|
| 1 | 33,404.7 | 29,101.0 | 12.9 % |
| 2 | 34,231.0 | 28,416.8 | 17.0 % |
| 3 | 34,234.5 | 28,039.2 | 18.1 % |
| mean | **33,956.7** | **28,519.0** | **16.0 %** (1 - 28,519.0 / 33,956.7) |

Three runs per mode on one machine: the `--gec` figure drifts down across the runs (29.1, 28.4, 28.0 k) while plain does not
(33.4, 34.2, 34.2 k), so the spread of the cost is wide (13 to 18 %) and the mean is only good to a few points. The
engine figure includes the CPU re-check of the probable primes (165 per run), which the plain and `--gec` paths share.

For comparison, the CUDA driver's own figure: `KIT_v8.55_NOTE.md` of the Beacon repository reports 514,850 -> 581,795 tests/s
on an RTX 5090, about 11.5 % cost. Our 16.0 % on an RX 6700 XT is in the same range but a different GPU, API, driver and
implementation, so it is a comparison and not a match, and this repository's design differs (see the profile).

## 4. Profile (the cost is above about 15 %)

A throwaway example (not committed) ran `Runner::fermat` and `Runner::fermat_checked` alternately on the same candidates in one
process: same adapter and kernel configuration as above (13-bit, TPI 8, workgroup 128, 160 limbs after padding), capacity
32,768 (larger than the engine's default runner capacity of 16,384, so this profile is not the engine's exact configuration), three alternating rounds, without the engine's CPU re-checks. `RunStats::gpu_secs` is the wall
time from submit to completion of the ladder dispatches; "other" is wall time minus `gpu_secs` minus `host_prep_secs`
(uploads, readback, unpacking, verdict comparison, and for `--gec` the extra buffer writes and the final check readback).

| Mode | wall (s) | `gpu_secs` (s) | host prep (s) | other (s) | tests/s |
|---|---|---|---|---|---|
| plain, mean of 3 | 2.741 | 2.511 (91.6 %) | 0.069 (2.5 %) | 0.161 (5.9 %) | 36,493 |
| `--gec`, mean of 3 | 3.277 | 2.784 (85.0 %) | 0.145 (4.4 %) | 0.349 (10.7 %) | 30,517 |
| difference | +0.536 | +0.273 | +0.076 | +0.188 | cost 16.4 % |

Raw rounds (wall s, gpu_secs s): plain 2.739 / 2.495, 2.770 / 2.546, 2.713 / 2.493; `--gec` 3.283 / 2.785, 3.244 / 2.760,
3.305 / 2.806; no alarms in any `--gec` round. So at the runner level the cost is 16.4 % (the engine-level 16.0 % above agrees
within the spread), and about half of the extra wall time (0.273 of 0.536 s) is GPU submit-and-wait time, which grew by about
11 %, close to the CUDA driver's whole-run figure; the other half is host side: host preparation doubled (the extra packing of
the `1~` start values and the per-candidate sum of block exponents, +0.076 s) and "other" more than doubled (+0.188 s: the extra buffer
writes and readbacks of the check are the likely part, but this was not measured). The "other" part was not split further, and no
submission count was recorded, so the share of `gpu_secs` that is per-block submit latency rather than GPU execution is not
measured.

Follow-up options (not implemented here):

1. Several Gerbicz blocks per submission via a pool of `fp` bind groups (today there is one submission and wait per
   40-iteration block, about 50 per chunk at this size), so that the block-boundary multiplies are queued behind the ladder
   dispatches without a wait in between.
2. Fuse the two boundary multiplies (into `A` and `B`) into the last ladder kernel of the block, saving two dispatches per block.
3. Move the host-side per-candidate set-up (the `1~` packing, the block-exponent sums) to the GPU or to the existing `R mod n`
   preparation so that it is not done twice, and fold the final check's readback into the verdict readback.
4. Measure where "other" goes (timestamp queries or per-step host timers) before choosing between 1 to 3.

Change after these measurements (not re-measured): the checked path used to submit each 40-iteration block, the tail and
each check pass whole, whatever the device speed. Its submissions are now bounded by the same adaptive segment length as
the plain path (`Runner::segment_override`, else grown toward `target_secs`, at most doubling per submission), kept across
blocks, passes and chunks; a block's boundary multiplies go into the submission of its last segment. On a device that
runs a block inside the target (the adapter above, at the default 0.1 s) every block after the first few iterations of
the first chunk is still one submission, so the numbers above should hold, but they were not re-run.
`RunStats::dispatches` now counts ladder-kernel submissions, so the submission count asked for in option 4 can be
recorded next time.

## 5. What was not tested

- NVIDIA, Intel and Apple GPUs; software rasterisers (llvmpipe and similar); browsers other than the one in section 6.
- The DX12 backend (no subgroups on this adapter).
- Other AMD GPUs, other driver versions, other candidate sizes: the throughput cost was measured only at 2,029 bits on one
  adapter, with three runs per mode.
- The 32-bit scheme was run for identity and the drill (see above), not for the throughput cost.
- Real hardware faults: the drill injected a deterministic bit flip through a test hook (since removed; the runner tests now simulate faults with mutated kernels), which exercises the detection logic
  but is not evidence about how often a real GPU would err.
- Beacon's own canary files (not available here).

## 6. Browser run (WebGPU)

The checked runner (`Runner::enable_gec`, `Runner::fermat_checked`, built to wasm32 through `bigmont-wasm`'s new
`fermat_checked_hex`) was run on `web/rust.html?gec=1` in the Claude desktop app's built-in browser (Chrome 156.0.8078.100,
Windows 11, WebGPU backend "BrowserWebGpu" on the same Radeon RX 6700 XT, subgroup size 32..64, no 64-bit integers),
13-bit limbs, TPI 8, workgroup 64, 24 candidates per size (random odd numbers, probable primes and the all-ones number),
every verdict compared with a BigInt `2^(n-1) mod n == 1` oracle:

| Bits | Mismatches | Alarms | Note |
|---|---|---|---|
| 20 | 0 | 0 | no full block: the whole run is the recomputed tail |
| 41 | 0 | 0 | one block, no tail |
| 42 | 0 | 0 | one block, one tail iteration |
| 81 | 0 | 0 | two blocks, one tail iteration |
| 255 | 0 | 0 | |
| 1019 | 0 | 0 | |
| 2029 | 0 | 0 | production size |

The same page without `gec=1` also had no mismatches at 255, 1019 and 2029 bits. This shows the checked path (new multiply
kernel, buffers and bind groups, adaptive submissions, async readbacks) works on a browser's WebGPU and never alarms on
clean input. It does not show fault detection in a browser: the fault tests are native only (they build mutated kernel
source), and no other browser, OS or GPU was run. `fermat.html` was not run: the generated kernels are unchanged by this
branch (the squaring and Fermat sources are byte-identical to before).
