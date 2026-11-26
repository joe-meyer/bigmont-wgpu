# Browser test page

`index.html` compiles a generated WGSL kernel (from `kernels/`) in the browser's own WebGPU implementation, checks it against a BigInt oracle and times it. The page prepends `enable subgroups;` and `diagnostic(off, subgroup_uniformity);` (see `docs/superpowers/results/2026-11-24-browser-test.md` for why).

Serve the folder and open it in a WebGPU browser with subgroups support (tested on Chrome 156):

    python3 -m http.server 8765 --directory web
    # then open http://localhost:8765/index.html?k=unrolled_tpi8_wg64
    # several at once: ?k=unrolled_tpi4_wg64,unrolled_tpi8_wg64,rolled_tpi8_wg64

Regenerate a kernel file with `cargo run --release -p bigmont-core --example dump_wgsl -- TPI WG rolled|unrolled > web/kernels/NAME.wgsl`. Results are shown in the page and kept in `window.__runs`.

The Fermat-ladder kernel (`Runner`'s shader) can be emitted in browser form too: `cargo run --release -p bigmont-core --example dump_wgsl -- 8 64 unrolled fermat-browser > web/kernels/fermat_tpi8_wg64.wgsl` (2,029-bit size class; the file already starts with `enable subgroups;` and `diagnostic(off, subgroup_uniformity);`). `index.html` does not load it (it drives the squaring-mode kernels and prepends the same two lines itself). In phase 1 this file was only compile-checked in Chrome 156 (`createShaderModule` plus `getCompilationInfo()` reported no messages, `createComputePipelineAsync` succeeded; `docs/superpowers/results/phase1/browser-fermat-compile-check.txt`); it was not dispatched or checked against an oracle in the browser.

## Fermat-ladder page (`fermat.html`)

`fermat.html` runs the Fermat ladder end to end: it compiles the kernels in `kernels/matrix/` (browser form, written by `dump_wgsl`), runs the base-2 Fermat test on a mixed batch on the GPU (random composites, probable primes, the all-ones number) and compares every verdict with a BigInt `2^(n-1) mod n == 1` oracle, then times a larger batch of random numbers (verdicts sampled against the oracle).

    # served by the bigmont-web entry of .claude/launch.json, or: python3 -m http.server 8765 --directory web
    http://localhost:8765/fermat.html?k=fermat_b2029_tpi8_wg64,fermat_b254_tpi4_wg64
    ...&compile=1      compile and create the pipeline only
    ...&count=32768    throughput batch size (0 skips it)

Names are `fermat_b<bits>_tpi<T>_wg<W>`. Names starting `limb32_` are the native-only 32-bit kernels: the page expects the browser to REJECT them (negative control, they use `u64`). Regenerate the matrix:

    for bits in 64 254 1019 2029; do for tpi in 4 8 16 32; do
      cargo run --release -p bigmont-core --example dump_wgsl -- $tpi 64 unrolled fermat-browser --bits $bits > web/kernels/matrix/fermat_b${bits}_tpi${tpi}_wg64.wgsl; done; done
    cargo run --release -p bigmont-core --example dump_wgsl -- 8 64 unrolled fermat-native --bits 2029 --scheme 32 > web/kernels/matrix/limb32_native_fermat_b2029_tpi8_wg64.wgsl

`dump_wgsl ... --scheme 32` with a `-browser` form exits 1 with the refusal (the 32-bit scheme is native-only). Results: `docs/superpowers/results/2026-11-25-browser-fermat-validation.md`.

## The Rust engine in the browser (`rust.html`)

`rust.html` runs the real Rust Fermat runner (`bigmont-core`, compiled to wasm32 by `crates/bigmont-wasm`) on the page's WebGPU, for several sizes and kernel shapes, and compares every verdict with a BigInt oracle. Only the 13-bit limb scheme runs in a browser (no 64-bit shader integers), the kernel shape is chosen by the page (`?tpi=8&wg=64`; there is no autotuner in the browser yet), and nothing times out: a device that hangs without being lost keeps the page waiting, while a reported device loss rejects the call.

Build and serve:

    rustup target add wasm32-unknown-unknown
    cargo install wasm-bindgen-cli --version 0.2.130 --locked     # must match the wasm-bindgen in Cargo.lock
    cargo build -p bigmont-wasm --target wasm32-unknown-unknown --release
    wasm-bindgen --target web --out-dir web/pkg target/wasm32-unknown-unknown/release/bigmont_wasm.wasm
    python3 -m http.server 8765 --directory web
    # then open http://localhost:8765/rust.html?bits=255,1019,2029&tpi=8
    # add &gec=1 to run with the Gerbicz error check (every verdict must still match, no alarms)

**wgpu is pinned to trunk for now.** Released wgpu 30.0.2 does not map the browser's `subgroups` feature to `wgpu::Features::SUBGROUP`, so a device cannot ask for it and the kernels' `enable subgroups;` is rejected (the page would report "needs subgroups, which the adapter lacks"). That is fixed upstream by [gfx-rs/wgpu#10265](https://github.com/gfx-rs/wgpu/pull/10265) (fixing [#10146](https://github.com/gfx-rs/wgpu/issues/10146)), merged to trunk on 2026-10-10 but not in 30.0.2, which is a patch release off an older branch. So `bigmont-core` and `bigmont-cli` depend on wgpu at trunk commit `60cef2be` (see the comment in their `Cargo.toml`), and the builds above need no patch. Switch back to a crates.io version as soon as a release contains #10265. Compiling for wasm32 does not depend on it; only running on the GPU does.
