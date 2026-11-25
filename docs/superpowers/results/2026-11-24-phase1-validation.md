# Phase 1 validation: hardware matrix, Beacon canary and throughput (2026-11-24)

Code under test: branch `phase1`, commit `b7f6a34` (head of the engine work plus the `fermat_stats` example added for this task; no engine code changed in Task 7). Raw outputs are in `phase1/` (one file per adapter and run; every file starts with a `# cmd: ... | host | adapter | date | commit` line). The kernels, the Runner, the autotune and the engine are the ones described in the Phase 1 plan; the binary is `bigmont`.

Revision note (final-review fix wave): this document was corrected after the final branch review. Section 3 (both failures), section 4.2 (PRP counts), section 5 (canary strength), section 8 and parts of the summary were rewritten to match the raw files; one measurement was added (`phase1/desktop-6700xt-vulkan-tail-dispatch.txt`, section 4.3). The hardware runs themselves were all made with the code at `b7f6a34`. The fix wave changed the code afterwards (stronger `bigmont canary`, a default tuning cache, atomic cache writes, usage exit codes; see section 8), and only the desktop RX 6700 XT (Vulkan) workspace test run was repeated on the changed code (`phase1/desktop-6700xt-vulkan-tests-fixwave.txt`: run 1 crashed at exit in the `sizes` binary, section 3.2; the `--no-fail-fast` re-run passed everything), so the other adapters have not run the changed tests.

Measured, not projected: every rate in this document is a wall-clock rate of a program that completed full base-2 Fermat tests (2,029-bit candidates in 2,048-bit containers), with the output checked as noted. The Phase 0 numbers it is compared with are projections (squarings/s divided by 2,029). Every figure is from a single run on a machine that was not otherwise idle-checked (see section 8).

## 1. Summary

- **Correctness:** the strict workspace suite passed on the RX 6700 XT (Vulkan and Dx12), the Radeon 780M and the Quadro P4000, and on the RTX 4070 after one failed first run (a device loss, section 3.1). On llvmpipe the strict suite fails 7 `chain.rs` tests by design (its subgroup size 8 cannot run TPI 16 and 32); with `BIGMONT_ALLOW_SKIP=1` it passes (section 3.3). Beacon's 400-candidate canary and the 32,000-candidate `big_2048.bin` file give 0 verdict mismatches against Beacon's expected verdicts on every GPU path and on the CPU path, and Beacon's own driver output is byte for byte identical to `bigmont`'s GPU and CPU output (section 5). Two failures are unexplained and are reported in section 3: a one-off RTX 4070 device loss in a test run (not reproduced in 6 later runs of that test binary), and a crash at process exit (0xC0000409) of the desktop `engine` test binary, which did reproduce: 2 of 8 Vulkan completions of that binary, plus one more crash in the `bigmont-core` `sizes` binary in the fix wave's own run (section 3.2).
- **Speed:** on the RTX 4070 the engine runs 36,053 tests/s end to end on a random 131,075-candidate file with default settings (27% of Beacon's measured 132,096 tests/s on the same card), and the raw GPU runner reaches 45,900 tests/s of GPU time when each dispatch holds 131,072 candidates (97% of the Phase 0 projection of 47,426). The engine's default of 16,384 candidates per dispatch is too small: raising it to 65,536 gains 10% (RTX 4070) and 19% (RX 6700 XT) of GPU-time throughput on the raw runner (section 4.3).
- **CPU path:** 1,800 to 2,030 tests/s on 16-thread Ryzen machines, 484 to 521 tests/s on 4 vCPUs of a Xeon E5-2690 v2. On the laptop, llvmpipe through the GPU engine (2,253 tests/s) is slightly faster than the native CPU backend (2,028 tests/s).

## 2. What ran where

| Host | Adapter (API) | Strict workspace tests | Autotune self-test (tune) | Throughput run | Canary (400 and 32,000) |
|---|---|---|---|---|---|
| Desktop, Windows 11, Ryzen 7 5800X (16 threads) | RX 6700 XT (Vulkan), driver 32.0.21045.16010 | pass, but the `engine` test binary crashed at process exit in the first run and in 1 of 6 direct repeats, and the `sizes` binary once in the fix wave run (section 3.2) | 12 of 12 configurations ok at 2,029 and at 1,024 bits | yes (16 k and 131 k files) | 0 mismatches |
| Desktop | RX 6700 XT (Dx12, `dxcompiler.dll` on PATH for those runs only) | pass | 12 of 12 ok at both sizes | not run (canary only) | 0 mismatches |
| Laptop `test-laptop`, Ryzen 7 7840HS (16 threads), driver 580.178.04 | RTX 4070 Laptop (Vulkan) | first run: 1 failure (device lost, section 3.1); full re-run and 5 repeats of that test: pass | 12 of 12 ok at both sizes | yes (16 k and 131 k files, plus Beacon's file) | 0 mismatches |
| Laptop | Radeon 780M (RADV) | pass | 12 of 12 ok at both sizes | yes (16 k and 131 k) | 0 mismatches |
| Laptop | llvmpipe (Mesa software Vulkan, subgroup size 8) | strict: 7 core tests fail by design (TPI 16 and 32 need a subgroup size 16 or 32; section 3.3); with `BIGMONT_ALLOW_SKIP=1`: pass | 6 of 6 ok (only TPI 4 and 8 are offered) | yes (4,096-candidate file) | 0 mismatches |
| host2 container 104 (`bigmont-test`), Xeon E5-2690 v2, 4 vCPU, GPU shared with three other guests | Quadro P4000 (Vulkan, driver 580.178.04) | pass | 12 of 12 ok at both sizes | yes (16 k file) | 0 mismatches |

Counts of the strict workspace run (identical shape on the passing adapters): bigmont-cli engine 5, files 6, tune 6; bigmont-core browser_source 3, chain 13, fermat 6, sizes 6; bigmont-cpu 4; bigmont-oracle 13; all passed, no tests ignored (these counts are for `b7f6a34`; the fix wave added tests afterwards). The run on the desktop under Vulkan and under Dx12 includes every GPU test of cli and core (the Dx12 run is the full workspace, with `BIGMONT_BACKEND=dx12`).

Raw files: `phase1/desktop-6700xt-vulkan-tests-run2.txt`, `desktop-6700xt-vulkan-tests.txt` (first run), `desktop-6700xt-dx12-tests.txt`, `laptop/rtx4070-tests.txt` (first run, failed), `laptop/rtx4070-tests-run2-nofailfast.txt`, `laptop/rtx4070-chain-repeat.txt`, `laptop/radeon780m-tests.txt`, `laptop/llvmpipe-tests.txt` (strict), `laptop/llvmpipe-tests-allow-skip.txt`, `ct104/p4000-tests.txt`.

## 3. Failures and oddities seen

These are reported as they happened; no expectation was edited and no test was skipped to hide them.

### 3.1 RTX 4070: device lost during `cargo test` (once)

In the first strict run on the laptop, `bigmont-core/tests/chain.rs::unrolled_long_chain_matches_oracle` failed with `KernelConfig { tpi: 32, wg: 64, ... }: no adapter: Parent device is lost` after its TPI 4, 8 and 16 steps had passed (`laptop/rtx4070-tests.txt`). That test is a Phase 0 test: each of its four checks opens a new device (`common::open_gpu` calling `Gpu::new()`), and the message comes from the device creation of the fourth check, so the failing path is repeated device creation and teardown within one process, not the `Runner` or the engine. Because that first run stopped at the failing test binary (no `--no-fail-fast`), the later test binaries (fermat, sizes, cpu, oracle) did not run in it. A second full run (`--no-fail-fast`) passed everything, and the failing test binary then passed 5 times in a row (`laptop/rtx4070-chain-repeat.txt`).

**Kernel log: attribution not established.** While investigating I saw two kernel-log lines `NVRM: Xid (PCI:0000:01:00): 32, pid=193667, name=tpi16_matches_o, channel 0x0000000d` (Xid 32 is, as far as I know, an invalid or corrupted push buffer; I did not look it up on the machine and could not read `dmesg` there, only `journalctl -k`). **No timestamped `journalctl` excerpt was saved in the raw files**, so "at the same moment as the failure" is not backed by data in this repository. The quoted name `tpi16_matches_o` is the first test of that binary in alphabetical order, and it passed in the failing run, so it is not the failing test; whether the Xid belongs to this failure, to the earlier test, or was already in the log from an earlier session is not shown. I cannot say from the saved files that the Xid and the device loss are the same event. No further Xid lines appeared in the kernel log after the later runs. Cause not found. Nothing in this phase's code was changed in response.

What would settle it: a timestamped `journalctl -k` saved next to the test log for each run; a loop of 50 or more runs of the `chain` test binary (and of `unrolled_long_chain_matches_oracle` alone with `--exact`) recording each exit code; and a variant of that test that shares one device across its checks, to see whether the repeated device creation is what triggers it.

### 3.2 Desktop: engine test binary crashes at process exit (intermittent, reproduced)

In the first desktop run (`desktop-6700xt-vulkan-tests.txt`), `bigmont-cli`'s `engine` test binary printed `5 passed; 0 failed` and then exited with code 0xc0000409 (`STATUS_STACK_BUFFER_OVERRUN`, which is the fast-fail exit status), so `cargo test` reported an error and stopped (later binaries not run in that run). All tests had passed before the crash, so this is a crash at teardown, and no panic text was printed.

**It did reproduce.** The 6 direct runs of the same executable (`desktop-engine-exit-repeat.txt`) were first read as all exiting 0, but run 6 is recorded as `exit code 127`. In Git Bash, 0xC0000409 shows up as 127 (checked: `cmd //c "exit -1073740791"; echo $?` prints 127 there), so run 6 is a second occurrence of the same crash, again after "5 passed". Counting the Vulkan completions of this test binary that the raw files hold (the first run, the `--no-fail-fast` re-run, and the 6 direct runs), it crashed at exit in 2 of 8. The Dx12 workspace run also completed it once with exit code 0, which makes 2 of 9 over all backends; with so few runs the rate is only roughly "one in four" for this binary.

**A third occurrence, in a different binary.** In the fix wave's own workspace run on the changed code (`desktop-6700xt-vulkan-tests-fixwave.txt`, run 1) the `bigmont-core` `sizes` test binary printed `6 passed` and then exited with 0xc0000409 (shown as 127 by Git Bash); a `--no-fail-fast` re-run of the whole workspace passed with exit code 0. The `sizes` binary completed successfully in 3 other runs I have files for (the first `--no-fail-fast` run, the Dx12 run, and that re-run), so it is 1 crash in 4 completions there. So the crash is not specific to the `engine` binary. I do not know what the two binaries have in common that the others lack: `chain` (which also opens a new device per check) and `fermat` did not crash in any saved desktop run, and `sizes` has no missing-adapter test, which weakens the suspect below.

What is not known: the cause. The desktop `bigmont` CLI never exited abnormally in the 13 recorded desktop CLI runs (2 throughput, 6 canary, 4 tune, 1 CPU throughput; all exit code 0; the 131 k CPU comparison run records no exit code), so the CLI is not shown to be affected, but those runs are few. No panic text and no crash dump exist (no dump was configured). 0xC0000409 is a fast-fail, which bypasses the panic machinery, so this is probably native code aborting during process teardown (a GPU driver or wgpu backend thread, or a destructor ordering problem) and not a Rust-level failure. One suspect, not tested: every `Gpu::open` enumerates all backends (`enumerate_adapters(Backends::all())`), and two of the five engine tests (`auto_falls_back_to_cpu_when_no_gpu_works`, `forced_gpu_backend_errors_when_the_adapter_is_missing`) do that with an adapter selector that matches nothing, so they enumerate every backend and then never open a device; the crash may come from the teardown of such an instance (but see the `sizes` occurrence above, which has no such test).

What would settle it: loop each engine test alone with `--exact` 50 times, recording each exit code (a crash in only the two missing-adapter tests would point at the backend enumeration), and do the same for the `sizes` binary and its tests; enable Windows Error Reporting LocalDumps for the test executable and read the faulting module from a dump; and loop the CLI with `--adapter` (and with `--backend cpu`, as a control) the same number of times. It is a backlog item; I would not call the engine tests clean on Windows without a cause.

### 3.3 llvmpipe: strict tests fail where the adapter cannot run the configuration

With strict skips, 7 `chain.rs` tests fail on llvmpipe because it reports subgroup size 8..8 and TPI 16 and 32 need at least 16 and 32 (`subgroup size 8..8 is not a multiple of tpi 16`). That is the suite working as designed (an unsupported configuration is a failure unless `BIGMONT_ALLOW_SKIP=1`), not a numerical error. With `BIGMONT_ALLOW_SKIP=1` and `--no-fail-fast` all binaries pass; libtest hides the loud skip lines of passing tests, so which configurations were skipped in the passing tests is not in the file. TPI 1 (chain) and TPI 4 and 8 passed. The brief expected TPI 4 might not fit: it does (subgroup size 8 is a multiple of 4), and the autotune offers TPI 4 and 8 only.

## 4. Throughput

### 4.1 Tuned configuration per adapter and size

`bigmont tune` (oracle self-test of each configuration, then a short timing of 256 workgroups, best of 2). Squarings/s from the table of each run; the engine's own autotune can pick a different near-tie in another run (for example the `fermat_stats` runs on the 6700 XT picked TPI 4 wg 128 in one run and wg 256 in another).

| Adapter | Best at 2,029 bits | squarings/s | Best at 1,024 bits | squarings/s |
|---|---|---|---|---|
| RX 6700 XT, Vulkan | TPI 4, wg 128 | 83.85 M | TPI 4, wg 256 | 300.8 M |
| RX 6700 XT, Dx12 + dxcompiler | TPI 4, wg 64 | 73.55 M | TPI 4, wg 256 | 262.6 M |
| RTX 4070 Laptop, Vulkan | TPI 4, wg 256 | 85.56 M | TPI 4, wg 256 | 323.2 M |
| Radeon 780M, RADV | TPI 8, wg 256 | 22.51 M | TPI 8, wg 256 | 71.59 M |
| Quadro P4000, Vulkan | TPI 8, wg 256 | 12.75 M | TPI 4, wg 256 | 50.95 M |
| llvmpipe (7840HS) | TPI 8, wg 128 | 2.84 M | TPI 4, wg 256 | 9.61 M |

Only the 2,029-bit class has end-to-end throughput runs. At 1,024 bits only the autotune (oracle self-test plus timing) and the existing size tests ran; no 1,024-bit Fermat throughput was measured.

Where it can be compared with Phase 0 (`2026-11-24-tuned-cross-vendor.md`): the short autotune timing is about 11% to 13% below the Phase 0 sweep for the same kernels on the two big GPUs (RX 6700 XT Vulkan 83.9 M against 96.2 M; RTX 4070 85.6 M against 96.2 M). On the Radeon 780M the autotune timing misranks TPI 4: it measured 8.0 M for TPI 4 (wg 128) and chose TPI 8 at 22.5 M, but the Phase 0 harness on the same code and card (`laptop/spike-780m-tpi4.txt`, 0.35 s dispatches, 131,072 instances) gives TPI 4 at 24.3 M and TPI 8 at 23.7 M. So on the 780M the autotune picked a configuration about 3% slower than the best one under long dispatches and reported a TPI 4 figure a third of the long-dispatch figure; why was not investigated (short measurement, warm-up, or a clock ramp on the integrated GPU are candidates; none tested).

### 4.2 End-to-end tests/s next to the Phase 0 projection

Engine: `bigmont CAND OUT --backend gpu`, random 2,029-bit odd numbers plus 3 probable primes, tests/s as printed by the binary (excludes autotune time, includes host preparation, GPU time, readback and the CPU re-check of every GPU PRP). Runner: the `fermat_stats` example on the same file (no CPU re-checks), wall time and GPU-dispatch time, default capacity 16,384 candidates per dispatch. Projection: Phase 0 best verified row divided by 2,029.

| Adapter | Phase 0 projection | Engine, 16 k file | Engine, 131 k file | Runner wall, 131 k file | Runner GPU time only, 131 k file | Engine 131 k / projection |
|---|---|---|---|---|---|---|
| RX 6700 XT, Vulkan | 47,418 | 28,229 | 33,842 | 31,396 | 34,415 | 71% |
| RTX 4070 Laptop, Vulkan | 47,426 | 31,285 | 36,053 | 37,194 | 41,086 | 76% |
| Radeon 780M | 12,019 | 10,407 | 10,616 | (16 k: 10,558) | (16 k: 10,925) | 88% |
| Quadro P4000 (shared, noisy) | 6,793 | 5,239 | not run | (16 k: 5,520) | (16 k: 6,048) | 77% (of 16 k) |
| llvmpipe, 7840HS (4,096 file) | 2,620 | 2,253 | not run | (4 k: 2,275) | (4 k: 2,295) | 86% (of 4 k) |

The 16 k and 131 k files hold 16,387 and 131,075 candidates. In the 16 k runs all 16,387 candidates ran on the GPU, which found 23 probable primes (26 re-checks: 23 PRPs plus 3 sampled composites). In the 131 k runs the output reads "gpu 131072, cpu 3": the CLI feeds the engine 32,768-candidate chunks, the last chunk holds 3 candidates, which is below `min_gpu_batch` (64), so those 3 (the 3 probable primes the file generator appends) ran on the CPU. The GPU therefore found 194 probable primes; the 197 in the output is the total (194 + 3). The 222 re-checks are the 194 GPU PRPs plus 28 sampled composites (the 1-in-4,096 sample restarts for every chunk, about 7 per chunk). All re-checks agreed: 0 disagreements and 0 GPU errors in every engine run (`desktop-6700xt-vulkan-throughput*.txt`, `laptop/*-throughput*.txt`, `ct104/p4000-throughput.txt`). For the P4000 and llvmpipe the last two columns are from the smaller file named in the cell. Dx12: the autotune picks TPI 4 wg 64 (73.55 M squarings/s against 83.85 M on Vulkan); no engine throughput run was made under Dx12 (canary runs only, section 5). Raw files: `desktop-6700xt-vulkan-throughput*.txt`, `desktop-6700xt-vulkan-fermat-stats*.txt`, `laptop/*-throughput*.txt`, `laptop/*-fermat-stats*.txt`, `ct104/p4000-throughput.txt`, `ct104/p4000-fermat-stats.txt`.

### 4.3 What limits it

- **Host-side Montgomery preparation is small.** `Runner` stats `host_prep_secs` is 0.3% to 6.2% of runner wall time on every run (4070: 2.5% to 6.2% depending on capacity, 0.206 s of 3.31 s at capacity 131,072; 6700 XT 1.7% to 2.5%; 780M 1.6%; P4000 5.8%). It is not what separates the engine from the projection.
- **The 16 k engine rate is lower than the 131 k one for a different reason than capacity.** The engine rates differ (6700 XT: 28,229 against 33,842 tests/s; 4070: 31,285 against 36,053), but both files run with the same capacity of 16,384 candidates per dispatch, so it is not batch size. What the files show: the 16,387-candidate file is one 16,384-candidate dispatch plus a 3-candidate tail dispatch, and a dispatch of 3 candidates still runs the whole 2,028-iteration ladder (the Runner restarts its segment ramp at 1 iteration per submission on every call and at most doubles it per submission, so covering 2,028 iterations takes at least 11 submit-and-wait round trips), which costs time like a full dispatch. In the 131 k runs the 3 leftover candidates went to the CPU (section above), so there is no such tail. The 16 k run is also half a second long, so a fixed cost of 0.1 s moves its rate by about 15%, and it includes first-call warm-up. I measured the tail directly on the 6700 XT (`desktop-6700xt-vulkan-tail-dispatch.txt`, raw runner, default capacity, files of 16,384, 16,387, 32,768 and 32,771 candidates, 2 rounds, noisy): the exactly-full files run at 35.3 to 35.7 k GPU tests/s (16,384) and 35.3 to 35.6 k (32,768), the same as the 131 k file's 34.4 k, so first-call warm-up is not visible in the GPU time; adding the 3-candidate tail costs 0.05 to 0.13 s on the 16 k file (GPU time 0.510 and 0.586 s against 0.459 and 0.464 s) and 0.02 to 0.10 s on the 32 k file, so the tail explains most or all of the gap in the raw runner's GPU time (0.503 s for the 16 k file in `desktop-6700xt-vulkan-fermat-stats.txt`). I did not measure the engine's remaining difference (re-checks, chunking, host preparation) separately, and did not repeat this on the 4070 or the other adapters, so for those the same explanation is likely but not measured.
- **Instances per dispatch matter more.** The engine's default capacity (16,384) under-fills the GPUs. Raw runner on the 131 k file (`desktop-6700xt-vulkan-capacity-sweep.txt`, `laptop/rtx4070-capacity-sweep.txt`), GPU-time-only tests/s:

  | capacity | RX 6700 XT | RTX 4070 |
  |---|---|---|
  | 4,096 | 23,177 | 40,220 |
  | 16,384 (engine default) | 34,332 | 41,002 |
  | 65,536 | 40,719 | 45,261 |
  | 131,072 | 41,044 | 45,900 |

  At capacity 131,072 the 4070 runner is at 96.8% of the Phase 0 projection (45,900 against 47,426) on GPU time, and 39,593 tests/s on wall time (84%); the 6700 XT is at 86.6% on GPU time (41,044 against 47,418) with the autotune's own figure (83.6 M to 85.7 M squarings/s) about 11% below the Phase 0 sweep's 96.2 M, so the remaining gap there is not capacity. The engine does not expose capacity as a flag (`Options::capacity`), so no engine run at larger capacity was made.
- **CPU re-checks dominate on PRP-dense input.** Every GPU probable prime is re-run on the CPU. On random numbers this is about 0.15% of candidates (26 re-checks of 16,387), which is nothing. On Beacon's `big_2048.bin`, half of whose candidates are primes, 16,003 re-checks take about 8 s of the engine's 8.5 s of non-tuning time on the laptop (at the CPU's 2,028 tests/s): the engine reports 3,768 tests/s there while the raw runner does 43,591 tests/s of GPU time on the same file (`laptop/bigmont-big-gpu4070.txt`, `laptop/fermat-stats-big-4070.txt`). That figure describes the cost of the safety re-check on a PRP-dense file, not the GPU.
- **Remaining wall time** (runner wall minus GPU time minus host prep: for example 0.22 s of 3.52 s on the 4070 at the default capacity) is upload, readback and API overhead; not broken down further.

### 4.4 Against Beacon on the same card

The same RTX 4070 Laptop GPU, Beacon `g1_fermat_v3` (kit v8.57, no Gerbicz-Li check, 8 lanes at 2,048 bits), `big_2048.bin` (32,000 candidates, 16,000 of them probable primes, 400 distinct numbers repeated 80 times):

| Driver | Run | tests/s | Notes |
|---|---|---|---|
| Beacon | `--repeat 10` (320,000 tests, 2.42 s) | 132,096 | agrees with the 132,100 of Test B |
| Beacon | `--repeat 1` (32,000 tests, 0.275 s) | 116,294 | one pass, the fair comparison for the rows below |
| bigmont Runner, GPU time only | same file, one pass | 43,591 | no CPU re-checks; wall 39,069 (0.819 s) |
| bigmont Runner, GPU time only | 131 k random file, capacity 131,072 | 45,900 | wall 39,593 |
| bigmont engine, default settings | 131 k random file | 36,053 | |
| bigmont engine, default settings | `big_2048.bin` | 3,768 | bound by 16,003 CPU re-checks |
| Phase 0 projection | | 47,426 | squarings/s divided by 2,029 |

So on this card the measured wgpu Fermat tests are about 29% to 35% of Beacon's rate depending on which of our figures is used (GPU time at best capacity 45,900 is 34.7% of 132,096; engine default settings 27.3%; one pass over the same 32 k file, runner wall against Beacon's single pass, 33.6%). The Phase 0 ratio (projection against Beacon) was 35.9%, so the projection was roughly right for the GPU time of a well-filled dispatch. The comparison keeps the Test B caveats: one card, no CUDA parameter sweep for this card, and Beacon's wall time includes whatever its own driver does around its kernel (not inspected). During Beacon's 2.4 s run the SM clock sampled at 2.46 to 2.48 GHz; over the 16 k engine run (which includes tuning) it averaged 2.43 GHz (minimum 2.18; `laptop/nvidia-smi-clocks.csv`, 1 s samples; throttle-reason bit 0x4 appeared once in that window, 0x1 is idle), so the clocks were in the same band for both, to the resolution of 1 s sampling.

### 4.5 CPU fallback speed

`bigmont ... --backend cpu`, all threads, base-2 Fermat, 2,029-bit; tests/s:

| Machine | 2,048-candidate file | Large file |
|---|---|---|
| Desktop, Ryzen 7 5800X, 16 threads | 1,845 | 1,835 (131,075 candidates, 71 s); 1,836 (Beacon's 32,000) |
| Laptop, Ryzen 7 7840HS, 16 threads | 1,824 | 2,028 (131,075 candidates, 65 s); 2,026 (Beacon's 32,000) |
| CT 104, Xeon E5-2690 v2, 4 vCPU (shared host) | 484 | 521 (Beacon's 32,000) |

GPU engine against the CPU path on the same machine (engine, 131 k file against CPU, 131 k file): RX 6700 XT 33,842 against 1,835 (18x); RTX 4070 36,053 against 2,028 (18x); Radeon 780M 10,616 against 2,028 (5.2x). P4000 5,239 (16 k file) against 521 (10x, different files, shared container). llvmpipe 2,253 (4,096 file) against 2,028: about 11% faster than the native CPU backend on the same 16 threads, but the llvmpipe engine run also does its re-checks on those threads. The CPU verdicts match the GPU verdicts on the common subset (first 2,045 and last 3 of the 16 k file: 0 mismatches on every machine) and on the whole 131,075-candidate file for the desktop and the laptop 4070 and 780M (0 mismatches; `desktop-cpu-131k.txt`, `laptop/laptop-cpu-131k.txt`).

## 5. Beacon canary

Inputs, read-only on the laptop (`~/beacon-test/p3`; byte-identical copies are in `~/beacon-test/kit/o1_kit` for the first two); sha256 in `laptop/inputs.sha256`:

| File | Size | Content |
|---|---|---|
| `canary_2048.bin` | 102,416 bytes | Beacon's canary candidates: magic `G2C1`, 2,048 bits, 400 candidates (200 prime, 200 composite; every candidate is odd and exactly 2,029 bits) |
| `canary_expected.bin` | 416 bytes | `G2V1`, 400 verdicts, 200 ones, 200 zeros: the kit's expected verdicts |
| `big_2048.bin` | 8,192,016 bytes | `G2C1`, 2,048 bits, 32,000 candidates: the canary repeated 80 times (byte-checked: body equals canary body times 80) |
| `big_expected.bin` | 32,016 bytes | `G2V1`, 32,000 verdicts: the expected canary verdicts repeated 80 times (byte-checked). This file was made in an earlier session, it is not from the kit; it is used below only as the file Beacon's output is compared against, and Beacon's own output matches it |
| `g1_fermat_v3` | built binary | Beacon's driver, run with `~/cuda-13.0/lib64` on the library path, `--device 0` |

The formats are exactly the contract (`G2C1`: u32 bits, u64 count, count x bits/32 u32 limbs; `G2V1` likewise with one byte per candidate; Beacon's expected files hold only 0 and 1, so verdict 2 (GEC alarm) was not exercised). Beacon reports `gec_alarm=0` and `prp=200` on the canary and `prp=16000` on the big file.

Results (all in `laptop/`, `desktop-canary-*.txt`, `ct104/canary-*.txt`):

- `g1_fermat_v3 canary_2048.bin` output equals `canary_expected.bin` byte for byte (`laptop/beacon-canary.txt`).
- On `big_2048.bin`, the verdict files of Beacon (`--repeat 1` and `--repeat 10`), `bigmont --backend gpu --adapter 4070` and `bigmont --backend cpu` are **byte-identical** (same sha256 `bea2e2d7...05ad`, equal to `big_expected.bin`); PRP count 16,000 in all four (`laptop/compare-big.txt`). That is 0 mismatches over 32,000 verdicts, 3 independent implementations (CGBN on CUDA, the wgpu kernel, the pure-Rust CPU path).
- `bigmont canary` (exit 0, `canary ok`) on both the 400-candidate canary and the 32,000-candidate file: RTX 4070, Radeon 780M, llvmpipe, laptop CPU; desktop RX 6700 XT under Vulkan and Dx12 (dxcompiler on PATH) and the desktop CPU; Quadro P4000 and the CT 104 CPU. In total 9 adapter-or-CPU paths, 0 mismatches each, 0 disagreements between the GPU result and its CPU re-check, 0 GPU errors (these counters are read from each run's summary line in the raw files; every GPU canary run there shows `gpu errors 0`, `disagreements: prp 0, composite 0` and all candidates counted as GPU tests).
- **Strength of the canary at `b7f6a34`.** `bigmont canary` at that commit passed or failed on the final verdicts only. The engine already replaces every GPU probable prime and a sample of GPU composites by the CPU verdict, and redoes a bucket on the CPU after a GPU error, so a wrong or unused GPU could still have passed. The runs above are not weakened by this, because the summary lines show the counters and GPU test counts directly, but a canary exit code of 0 from that commit did not by itself prove the GPU was right or used. The fix wave changed `bigmont canary` to exit 1 also on any PRP or composite disagreement, on any GPU error, and on a GPU run with 0 GPU tests (section 8), with automated exit-code tests; the 400-candidate canary samples no composites at the default rate (section 8), so it still exercises only the PRP re-check path.
- A negative control from Task 6 (one flipped byte makes `bigmont canary` exit 1 and name the index) is not repeated here as a hardware run; the fix wave made it an automated test (`crates/bigmont-cli/tests/canary.rs`, CPU backend).

Caveat: the 32,000-candidate file contains only 400 distinct numbers. It is a good correctness file (half primes) and a fair kernel-time file for Beacon (a Fermat test does the same squarings whatever the number), but it exercises few distinct inputs.

## 6. Quadro P4000 in CT 104

Shared GPU: during the runs the other containers (plex, frigate, llm) could have used the same card; at the start `nvidia-smi` on host2 showed 1% utilization and 4.8 GB in use, and I did not monitor it during the runs. Timings are therefore noisy and one run each. All tests passed under the strict suite, the autotune passed 12 of 12 configurations at both sizes (best TPI 8 wg 256 at 2,029 bits, TPI 4 wg 256 at 1,024 bits, the third-as-fast TPI 4 at 2,029 bits as in Phase 0), the engine ran 16,387 candidates at 5,239 tests/s, and the canary matched. The `vulkaninfo` exit-139 quirk from the memory notes was not exercised (I did not run `vulkaninfo`); every `bigmont`, example and test process in the container exited 0.

## 7. Browser

The `Target::Browser` form of the generated Fermat kernel (TPI 8, wg 64, 2,029-bit class, 60,741 bytes, `web/kernels/fermat_tpi8_wg64.wgsl`, written by `dump_wgsl ... unrolled fermat-browser`) was compile-checked in Chrome 156 on the RX 6700 XT (WebGPU adapter amd/rdna-2, subgroups feature present, subgroup size 32..64): `createShaderModule` + `getCompilationInfo()` returned no messages (no errors, no warnings), `createComputePipelineAsync` with an automatic layout succeeded and the validation error scope was empty (`phase1/browser-fermat-compile-check.txt`). **It was not dispatched and no result was checked against an oracle in the browser.** Other browsers and GPUs were not tried. The check needed one small change to the `dump_wgsl` example (a `fermat-browser` argument), committed separately.

## 8. Not verified, limits

- **Gerbicz-Li error check is not implemented.** Verdict 2 (GEC alarm) is never produced by `bigmont` and was not exercised; the only integrity check is the CPU re-check of GPU primes and a 1-in-4096 sample of GPU composites. A GPU error that turns a prime into a composite on a number that is not sampled would not be caught (the canary mitigates this at the start of a run, as Beacon's does).
- **Single runs, no variance estimates.** Throughput numbers are one run per cell (the capacity sweep also ran each capacity once); run-to-run noise from Phase 0 was about 1% but was not re-measured here. The desktop GPU is shared with whatever the user runs and its clocks and load were not monitored; its measurements were made back to back (`desktop-6700xt-vulkan-throughput.txt` run 28,229 tests/s on the 16 k file, 33,842 on the 131 k file, the difference is explained in section 4.3 as the 3-candidate tail dispatch of the 16 k file plus a short run, not as batch size and not as drift, as far as I can tell; the tail explanation was measured on the raw runner only). The laptop GPU clock was sampled (section 4.4); the P4000 and the iGPU were not.
- **Intel and Apple GPUs, wave64, Firefox and Safari: untested.** No Intel GPU is available; AMD wave64 was not observed.
- **Browser: compile check only** (section 7). No browser run of the Fermat kernel, no browser throughput.
- **Only one size class end to end.** 2,029-bit Fermat throughput only; 1,024 bits ran through the autotune self-test and timing only. Mixed-size buckets are covered by the engine tests, not by a throughput run.
- **Dx12:** the full test suite passed with `dxcompiler.dll` on PATH; Dx12 without it (no subgroups, so no cooperative kernel) was not run in this phase; no Dx12 engine throughput run.
- **Autotune accuracy.** Its short timing is about 11% to 13% low on the two big GPUs and misranked TPI 4 on the 780M (section 4.1). The engine picks by that timing. Not fixed.
- **Engine default capacity 16,384** under-fills both big GPUs (raising it to 65,536 gains 10% and 19% of GPU-time throughput on the raw runner, section 4.3); no flag exposes it, so the best-capacity numbers are for the raw runner only.
- **CPU re-check cost** on PRP-dense files (section 4.3), and the CPU backend's comparison with a GMP-class library on the same machine, were not measured here (the Phase 0 GMP figure on the same CPU model is about 265 tests/s per core and about 1,085 tests/s on 4 vCPUs; the pure-Rust backend measured 521 tests/s on 4 vCPUs here, about half of it).
- **Two failures are not explained** (section 3): the 4070 device loss (once, not reproduced) and the desktop test binaries' exit crash (reproduced: the `engine` binary 2 of 8 Vulkan completions, the `sizes` binary 1 of 4; cause unknown, no dump). The Xid log lines are not tied to the device loss by any saved timestamp.
- **Not compared:** Beacon's `--ee` mode, `--inject`, `--bins`, `--chunk`, `--growth`, `--marker` (unsupported by `bigmont`); Beacon on other cards; the engine on host1 (no usable GPU).
- **Canary strength before the fix wave.** At `b7f6a34` `bigmont canary` compared final verdicts only (section 5). Fixed afterwards: it now exits 1 when `prp_disagreements + composite_disagreements + gpu_errors > 0`, or when the resolved backend is the GPU and `gpu_tests == 0`, and prints which condition failed (`--backend cpu` is exempt from the last one). Automated tests cover the decision function and the exit codes (matching file 0, flipped byte 1, length mismatch 1). Hardware canary runs were not repeated with the new code.
- **A lost device or an uncaptured wgpu error aborts the run.** Only `Err` returns from autotune, `Runner::new` and `Runner::fermat` are caught and fall back to the CPU (counted as GPU errors). `Runner` submits, `write_buffer` and polls run outside error scopes, and `Gpu::open` installs no device-lost or uncaptured-error handler, so wgpu's default handler panics and the process exits without writing a verdict file (safe: the verdict write is atomic, so no partial file is left, but there is no CPU fallback for such a failure). No handler was implemented in this phase; no test forces a device loss. Earlier comments and docs that said GPU failures are "counted, never propagated" were too strong and were corrected.
- **No default tuning cache before the fix wave (now fixed in code, not re-measured on hardware).** At `b7f6a34` the cache was used only with `--cache`, so every plain `bigmont` run autotuned fully for each distinct bit length (8 to 21 s of tuning in the raw files) and `bigmont tune` ignored `--cache`. The fix wave added a per-user default (`BIGMONT_CACHE`, else `%LOCALAPPDATA%\bigmont\tune.cache` on Windows, else `$XDG_CACHE_HOME/bigmont/tune.cache` or `$HOME/.cache/bigmont/tune.cache`), `--cache PATH`, `--no-cache`, atomic cache writes, and `bigmont tune` now writes the cache. The tuning times in the raw files therefore vary: a miss tunes every configuration (8 to 21 s), a hit with an explicit `--cache` still runs the self-test (1.5 to 1.7 s in the 131 k runs); no run here used the new default.
- **`--device N` counts adapters across all backends.** The index runs over the non-CPU adapters of every backend wgpu enumerates, so one GPU appears once per API (Vulkan, Dx12, GL duplicates). It selects the intended GPU only while Vulkan enumerates first. Redesign deferred; use `--adapter` with `--vk-backend` as this document's runs did.
- **CLI chunk tails can run on the CPU.** The CLI feeds the engine 32,768-candidate chunks; a final chunk (or any bit-length bucket) below `min_gpu_batch` (64) runs on the CPU. This is why the 131 k runs print "gpu 131072, cpu 3" (section 4.2). Bucket size is the only size-based reason a bucket skips the GPU, besides a failed autotune or runner `Err` (after this report the GPU is used down to 2-bit numbers, so bit length is not a reason; see the backlog entry "Small bit lengths"). At 40 bits the GPU measured slower than the CPU (about 0.35 to 0.5 M tests/s end to end against 4.3 to 4.6 M; the tuner picked the narrowest 4-limb kernel, so it is per-call overhead, not kernel padding): `results/phase1/desktop-6700xt-vulkan-40bit-gpu-vs-cpu.txt`, `results/phase1/desktop-6700xt-vulkan-tune-40.txt`.
- **Composite sampling restarts on every `Engine::verdicts` call.** The 1-in-4,096 re-check counter is per call, so a call with fewer composites than the rate samples none; the 400-candidate canary (200 composites) re-checks no composite, and a 32,768-candidate chunk samples about 7. Deferred.
- **Fermat mode with one limb per lane had no direct test** (at the time of this report). The `lpl == 1` path of the Fermat kernel was covered only by the autotune self-test at run time. Since then `bigmont-core/tests/fermat.rs` covers it: tpi 8 with 8 limbs and tpi 4 with 4 limbs at every length from 2 to 12 bits (exhaustive) and at 20, 26, 31, 32, 33, 39, 47 and 48 bits, and the 2-bit n = 3 at tpi 4, 8, 16 and 32.
- **Capacity has no CLI flag, and the capacity code ignores `max_buffer_size`.** The engine's 16,384 candidates per dispatch cannot be changed from the command line (65,536 gave +10% on the 4070 and +19% on the 6700 XT in the raw runner, section 4.3); the capacity is clamped to the dispatch-group and storage-binding limits but not to `max_buffer_size`.
- Environment changes made this task: none on any host beyond files under `~/bigmont-test` (laptop) and `/root/bigmont-test` (CT 104), plus a background `nvidia-smi` sampler on the laptop that was stopped afterwards. No packages, drivers or hypervisor settings were touched; CT 104 was already running.
