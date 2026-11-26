# Device losses, adapter loss and exit crashes: investigation (2026-11-24)

Three different failures turned up while validating Phase 1. They looked like one problem ("lots of device losses") and are not. This document says what is established, what is not, and what changed in the code. Raw logs are in `docs/results/phase1/devloss/`. Single machines, small samples, p-values rough (Fisher exact, no multiple-comparison correction).

## Summary

| Symptom | Where | Cause | Status |
|---|---|---|---|
| "no adapter matches 4070", or silently a different adapter, after about 10 `Gpu::open` calls in one process | Linux laptop, RTX 4070 | glibc static-TLS exhaustion: every new wgpu `Instance` re-`dlopen`s the NVIDIA driver library and glibc cannot return its static TLS | **Root cause shown, fixed** (one shared `Instance`), regression test added |
| Test binary exits with 0xC0000409 (Git Bash exit 127) after printing "N passed" | Windows desktop, RX 6700 XT | Initialising wgpu's GL backend (and, by implication, AMD's `atio6axx.dll`) next to Vulkan; **mechanism not shown** | **Trigger shown, avoided** (default backends are now Vulkan, Dx12, Metal) |
| NVIDIA Xid 32, "Parent device is lost", occasional hang | Linux laptop, RTX 4070 | **Not our kernels, not the GL backend, not the TLS problem.** Happens with a no-op dispatch and with other programs; mechanism unknown | **Not fixed** (driver); rate about 0.7% to 4% of processes; the engine now survives it (section 4: re-open and retry, then CPU fallback); a driver change is still the user's decision |

## 1. Linux: the adapter disappears after about ten devices (fixed)

`Gpu::open` built a new `wgpu::Instance` every call. Observed with `BIGMONT_ADAPTER=4070` and the new default backends: the first 10 opens in a process succeed and the 11th fails with "no adapter matches" on every run (80 of 80 processes, `laptop-churn3.log`). With `VK_LOADER_DEBUG=error` the loader prints, once per failed ICD load: `libnvidia-tls.so.580.178.04: cannot allocate memory in static TLS block` and "Failed loading library associated with ICD JSON libGLX_nvidia.so.0. Ignoring this JSON" (`laptop-vkdbg.log`). File descriptors (46) and threads (9) stayed constant over all cycles, so it is not an fd or thread leak.

Controlled test (`laptop-tls.log`, 40 cycles, same binary): default glibc, dies after 10 cycles with 4 TLS errors; `GLIBC_TUNABLES=glibc.rtld.optional_static_tls=65536`, 40 of 40 pass with 0 TLS errors.

With the old default (all backends) the same exhaustion was invisible in name-filtered runs because the GL backend exposes an NVIDIA adapter whose name also contains "4070"; unnamed selection silently fell back to another adapter. This is very likely the root cause of the old "enumerate_adapters intermittently misses the requested adapter" problem that `INIT_LOCK` in `gpu.rs` was a mitigation for (not re-verified on that older code).

Fix (commit "Share one wgpu Instance per process"): a process-wide `OnceLock<wgpu::Instance>`; the driver library is loaded once. Verified: 60 of 60 named-adapter cycles with no TLS errors, and `tests/many_devices.rs` (40 opens must keep the same adapter) passes with and without a name filter. Not verified on Windows/macOS (the TLS limit is a glibc property).

## 2. Windows: exit crash with all backends (trigger shown, mechanism not)

Interleaved A/B on the `sizes` test binary, same machine, same session (`desktop-sizes-ab-allbackends-vs-vulkan.log`, `desktop-sizes-primary-backends.log`, `desktop-sizes-new-default.log`):

| wgpu Instance backends | exit crashes (127 = 0xC0000409) |
|---|---|
| all (the old default) | 6 of 60 |
| Vulkan only | 0 of 60 |
| `primary` (Vulkan + Dx12 + Metal) | 0 of 60 |
| new default (= `primary`) | 0 of 60 |

Fisher exact, 6/60 vs 0/60: p about 0.03. Other loops of the same binary under the old default gave 1 of 12 and 0 of 25 (sizes) and 0 of 25 (chain): the rate is noisy, 0 to 10%.

Windows Application event log: exactly one logged crash in two days is from our work, a Phase 0 debug `chain` binary at 09:54: faulting module `atio6axx.dll` (AMD's user-mode driver), exception 0xc0000409, fault offset 0x1083405. The later crashes of the release binaries left no Application Error entries (not understood). The same machine's event log also holds older kernel-level GPU events from before this work (LiveKernelEvent 141 on Nov 14 and 23, and WHEA 124 hardware-error records on Nov 23); they were not analysed.

What is not established: why GL initialisation crashes at exit (the tests run each check on its own spawned thread and create devices on it; the `device_churn` diagnostic with new devices per cycle, with and without threads and backends restricted, produced 0 failures in about 480 process runs, so the failing condition needs something the diagnostic lacks). The fix is avoidance: our kernels need subgroups, which GL does not have, so initialising GL could never help. `BIGMONT_INSTANCE_BACKENDS` (`vulkan,dx12,metal,gl,primary,all`) restores any combination.

## 3. Linux: NVIDIA Xid 32 and device loss (not fixed; not ours)

The kernel log shows `NVRM: Xid ... 32 ... intr0 00040000` events; 98 Xid lines in total since boot (`laptop-xid-lines.txt`, each event appears twice). Facts:

- **They predate Phase 1 work and hit other programs.** Nov 23 22:53 `gapsieve` (your other project); Nov 24 00:11 to 00:27 `broken_fast_pat` (three events); then ours. The `name=` field is the thread that first opened the GPU, not the test that faulted (the hung test process had its thread of `unrolled_long_chain_matches_oracle` in a poll while the Xid line said `tpi16_matches_o`).
- **It follows device creation/use, not our arithmetic.** Fixed code (one shared Instance), `device_churn`, named adapter, 8 threads x 70 instances per cycle (`laptop-churn4.log`, `laptop-churn5.log`):

| Condition | processes with an Xid / loss |
|---|---|
| new default backends, a new device each of 12 cycles | 4 of 100 |
| all backends, a new device each of 12 cycles | 4 of 100 |
| new default, one device reused for 12 cycles | 2 of 100 |
| one reused device, a **no-op kernel** (0 squarings) | 1 of 150 |
| one reused device, the real kernel (128 squarings) | 1 of 150 |
| `vulkaninfo --summary` alone (instance and enumeration; no logical device) | 0 of 150 |

So backends (GL or not), the TLS problem, the number of devices and the kernel arithmetic are all excluded as the cause of the Xid (the earlier 8/160 vs 1/160 difference between all backends and Vulkan only was measured on code that still had the TLS problem and is not reproduced after the fix: treat it as chance or confounded). What remains: creating a Vulkan device (or running anything on it) and tearing it down on this driver (NVIDIA open kernel module 580.178.04, kernel 7.x), at about 1 in 100 processes. Rare variant: the process hangs instead of failing (at least six hangs killed after 90 s across the loops, rc124 in `laptop-churn*.log`; one observed live with threads in `futex_do_wait`/`poll`).
- A single-device process (one `Gpu` for the whole run, as in the `bigmont` CLI) is exposed once per run, so the per-run risk is in the order of 1%, not 0.

## 4. What this means for the engine

- The CLI opens one device per process; with the shared Instance that is as little exposure as possible. Before the handling below, a failure aborted the run with no verdict file (the write is atomic), so a wrapper that re-runs a failed invocation was, and still is, safe.
- **Implemented since (device-loss handling):**
  - `Gpu::open` (and `Gpu::from_parts`) install a device-lost callback and an uncaptured-error handler that record into the `Gpu` instead of panicking. `Gpu::check_health` reports the first problem (a timeout, a loss, or the first uncaptured error; problems are sticky), after a non-blocking poll that delivers a pending lost callback.
  - `Gpu::wait` is the only wait: a `PollType::Wait` with a timeout (`Gpu::poll_timeout`, from `BIGMONT_POLL_TIMEOUT_MS`, default 30 s) on a helper thread. A timeout poisons the `Gpu` ("GPU wait timed out ..."). wgpu 30 *panics* inside `Device::poll` when the driver reports the device lost (`Error in Device::poll: Parent device is lost`), and that panic is caught and turned into "device lost: ...". If the poll ignores its timeout (the hang mode seen here), the caller gives up after the timeout plus 2 s and leaves the helper thread behind. Readback (`Gpu::read_back`) maps through the same wait and a channel. `run`, `Runner::new`, `Runner::fermat` and the tune functions check health before submitting and after waiting.
  - A poisoned or driver-lost `Gpu` leaks its wgpu device and queue on drop on purpose. Dropping the last wgpu `Queue` runs `vkQueueWaitIdle`, which blocks for as long as the outstanding work does, and then asserts that the queue is empty. Measured on the 4070 (`phase1/devloss/handling-poisoned-drop.txt`): a normal drop after a timed-out 0.82 s dispatch took 0.85 s, the leaking drop about 2 µs. Truly stuck work would block the drop forever. A device lost through `destroy()` is idle and is dropped normally.
  - The engine treats a lost device, a timeout and an uncaptured error as a device-level failure. It counts `device_losses` (and `poll_timeouts`), drops the device and all runners, re-opens the device (counted in `device_recoveries`) and re-runs the whole bucket. The tuned configurations are kept only when the re-opened device is the same adapter (`gpu::same_adapter`: name, vendor and device id, type, PCI bus, backend, driver and its version, subgroup sizes). On a different adapter they are dropped, so the next bucket autotunes and self-tests again. The `Engine::open` rules apply as well: under `Auto` an adapter without subgroups makes the engine CPU-only, and under `Gpu` a switch to a different adapter without subgroups makes `verdicts` return `Err`. Re-opens are bounded twice. `Options::device_retries` (default 2) limits them per `verdicts` call, and successful re-opens count too. `Options::device_retries_total` (default 8) limits them over the engine's life, because the CLI calls `verdicts` once per 32k chunk. When either limit is used up, the engine answers that bucket and the rest of the process on the CPU (`gpu_errors` + 1). A failed tuning caused by a device failure is not cached. The summary line prints the three counters, and `bigmont canary` fails on any device loss. The test hooks are `Engine::inject_device_loss` and `Options::inject_loss_on_open` (both call `device.destroy()`), `Engine::inject_poll_panic` / `Gpu::inject_poll_panic` (the next wait's poll panics as wgpu 30's does on a real driver loss, so the `catch_unwind` path is exercised), `Gpu::wait_with` (injects the blocking poll, used to test a poll that never returns), and `Engine::inject_adapter_change`.
  - **Limits:** verified with simulated losses (`destroy()`, an injected poll panic), forced timeouts and an injected never-returning poll, on the RTX 4070 laptop only. No real Xid 32 has struck under the new code yet. A hang inside a driver call other than the poll (`queue.submit`, `write_buffer`, device creation in `Gpu::open`) is not covered: those calls have no timeout. The default panic hook still prints wgpu's poll panic to stderr before it is caught. Every leaked device keeps its memory until the process exits. Each wait spawns a short-lived thread, which costs microseconds per dispatch.
- **Left to you (system-level, not touched):** whether to try a different NVIDIA driver build or the proprietary instead of the open kernel module; whether to loosen `kernel.yama.ptrace_scope` or enable GPU profiling counters for deeper tracing.

## 5. Not verified / limits

- The Xid cause is unknown; only what it is not was shown. Different driver/kernel/hardware not tried. The P4000 and the 780M were not looped for Xid-like events; the Windows desktop has no NVIDIA driver.
- The Windows exit crash mechanism (why GL init crashes AMD's driver) is unknown; the fix is avoidance, verified on 60 runs of one test binary. Update 2026-11-25: the `sizes` (40), `engine` (30) and `chain` (30) binaries were looped under the new default on the desktop: 100 runs, 0 exit crashes (`desktop-new-default-all-three-binaries.log`); the full workspace suite with the device-loss code merged passed 120 of 120 on the desktop (`desktop-full-suite-after-devloss-handling.log`).
- The TLS fix is a Linux/glibc fix, verified on the laptop; the many-devices test also passes on the Windows desktop (where the limit does not apply).
- All samples are single machines and small (60 to 150 processes); rates are indicative.

## 6. Raw files (`docs/results/phase1/devloss/`)

Desktop: `desktop-sizes-ab-allbackends-vs-vulkan.log`, `desktop-sizes-primary-backends.log`, `desktop-sizes-new-default.log`, `desktop-sizes-chain-loop.log`, `desktop-sizes-loop-part1.log`, `desktop-churn-new-vs-reuse.log`, `desktop-churn-threads-allbackends-vs-vulkan.log`. Laptop: `laptop-loop_chain.log` and `laptop-exps.log` (chain test binary loops), `laptop-churn.log`, `laptop-churn2.log` (old code, all backends), `laptop-churn3.log` (named adapter, TLS failures), `laptop-vkdbg.log` and `laptop-tls.log` (TLS evidence), `laptop-churn4.log`, `laptop-churn5.log` (fixed code), `laptop-xid-lines.txt` (kernel log). Device-loss handling (laptop): `handling-red.txt` and `handling-red-engine.txt` (the new tests on the stubs, failing), `handling-green.txt` (the same tests passing), `handling-poisoned-drop.txt` (drop time of a poisoned `Gpu`, leaked vs normal), `handling-full-suite.txt` (whole workspace on the 4070); after the review fixes, `handling-fix1-green.txt` (the device-loss, canary and unit tests) and `handling-fix1-full-suite.txt`. Machine-specific home paths in these logs are replaced with `~`.
