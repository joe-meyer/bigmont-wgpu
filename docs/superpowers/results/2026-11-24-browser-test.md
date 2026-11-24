# Browser test: do the kernels run in a real browser, and how fast? (2026-11-24)

Page: `web/index.html` (kernels in `web/kernels/`, generated with `crates/bigmont-core/examples/dump_wgsl.rs`; serve with the `bigmont-web` entry in a `.claude/launch.json`, e.g. `python3 -m http.server --directory web`). Run in the Claude desktop app's built-in browser pane: **Chrome 156.0.8078.100**, WebGPU adapter reported as vendor `amd`, architecture `rdna-2` (this machine's RX 6700 XT), `subgroups` feature present, subgroup size 32 to 64. Each kernel is compiled by the browser's own WebGPU implementation (Tint), checked against a BigInt oracle (same arithmetic as the Rust oracle), and timed with `performance.now()` around `submit` plus `onSubmittedWorkDone()`, best of 3, 32,768 instances. Measured: squarings/s. Projected: tests/s = squarings/s / 2029 (not a measured Fermat test).

## 1. What it took to make the kernels valid in a browser

1. **`enable subgroups;` is required in browsers** and Chrome accepts it. Our native path omits it because naga 30 rejects it, so the generated source is not browser-valid as is; the test page prepends it.
2. **Chrome's Tint rejected the first compile** (even with `enable subgroups;`): `'subgroupShuffleUp' must only be called from subgroup uniform control flow`, because `full_normalize` leaves its loop through a `break` driven by a `subgroupAny` vote and Tint's analysis cannot prove that vote is uniform across the subgroup. It is uniform by construction (every lane in the subgroup is active, dead lanes are clamped not returned, and the vote is the same for all of them), so the page also prepends **`diagnostic(off, subgroup_uniformity);`**. With both lines the kernels compile with no messages. Native naga never raised this, so it is a browser-only requirement. The override is only safe because of the property above; a future kernel that lets lanes exit early would break it.

## 2. Correctness in the browser

All four kernels tested (unrolled TPI 8, 16, 32 and rolled TPI 8, workgroup 64) passed all three checks: 133 instances (random 2,029-bit moduli plus the all-ones and 2^2028+1 carry-stress moduli) after 3 squarings; 72 instances after 128 chained squarings; and 24 sampled instances of the timed run itself after 768 chained squarings. Each check compares the full value mod n, that every limb is below 2^13 and that the value is below 2n. No compilation messages.

## 3. Speed in the browser (RX 6700 XT, long dispatch of 256 squarings)

| Kernel | squarings/s | slope (64 to 256 sq) | projected tests/s | Native wgpu, same card (best row, for reference) |
|---|---|---|---|---|
| unrolled TPI 8 | 70.8 M | 73.1 M | 34,889 | 81.8 M Vulkan, 75.4 M Dx12 |
| unrolled TPI 16 | 54.2 M | 56.4 M | 26,708 | 64.1 M Vulkan, 54.4 M Dx12 |
| unrolled TPI 32 | 39.4 M | 40.7 M | 19,392 | 48.4 M Vulkan, 38.5 M Dx12 |
| **rolled TPI 8** | **74.4 M** | 75.1 M | 36,685 | 4.5 M Vulkan (rolled) |

(Earlier run of unrolled TPI 8: 70.7 M, slope 73.5 M. The first measurement, 43.7 M at only 16 squarings per dispatch, included one-time warm-up and was withdrawn.)

What this says:
- The unrolled kernel runs at about 87 to 94% of native wgpu speed on this card, in the same TPI order (8 > 16 > 32).
- **The rolled kernel is as fast as the unrolled one in the browser** (74.4 M against 70.8 M; 16 times its native Vulkan speed). Chrome's compiler on Windows evidently unrolls the constant-bound loops that the AMD Vulkan driver leaves rolled (it also compiled the native Dx12 path differently: native Dx12 with dxcompiler ran the rolled kernel at about 5 M). The unrolled generator is therefore a fix for specific native compilers, not something the browser path needs; it costs nothing there but also gains nothing, and the rolled kernel was slightly faster here (about 5%, a single run each, not shown to be outside noise).
- Browser speeds like these are in the same range as native on the same hardware, which is encouraging for the web goal; it is one GPU, one browser and one OS.

## 4. Caveats

One GPU (RX 6700 XT), one browser build (Chrome 156 inside the Claude desktop app), one OS (Windows), one or two runs per kernel. Not tested: Firefox, Safari, other GPUs or vendors in a browser (the laptop's NVIDIA and AMD GPUs and the P4000 could be tried by opening this page in a browser on those machines), mobile, WebGPU on the browser's software fallback adapter, power and thermal limits over long runs, the browser's behaviour for very long dispatches (watchdog), memory limits with many instances, and how subgroup size is chosen at run time (the page does not read it from the kernel; the adapter reports 32 to 64). Timing includes submit and completion latency (small at these dispatch lengths, 100 to 200 ms). Verification samples 24 instances of the timed run, not all.
