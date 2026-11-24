# Browser test page

`index.html` compiles a generated WGSL kernel (from `kernels/`) in the browser's own WebGPU implementation, checks it against a BigInt oracle and times it. The page prepends `enable subgroups;` and `diagnostic(off, subgroup_uniformity);` (see `docs/superpowers/results/2026-11-24-browser-test.md` for why).

Serve the folder and open it in a WebGPU browser with subgroups support (tested on Chrome 156):

    python3 -m http.server 8765 --directory web
    # then open http://localhost:8765/index.html?k=unrolled_tpi8_wg64
    # several at once: ?k=unrolled_tpi4_wg64,unrolled_tpi8_wg64,rolled_tpi8_wg64

Regenerate a kernel file with `cargo run --release -p bigmont-core --example dump_wgsl -- TPI WG rolled|unrolled > web/kernels/NAME.wgsl`. Results are shown in the page and kept in `window.__runs`.

The Fermat-ladder kernel (`Runner`'s shader) can be emitted in browser form too: `cargo run --release -p bigmont-core --example dump_wgsl -- 8 64 unrolled fermat-browser > web/kernels/fermat_tpi8_wg64.wgsl` (2,029-bit size class; the file already starts with `enable subgroups;` and `diagnostic(off, subgroup_uniformity);`). `index.html` does not load it (it drives the squaring-mode kernels and prepends the same two lines itself). In phase 1 this file was only compile-checked in Chrome 156 (`createShaderModule` plus `getCompilationInfo()` reported no messages, `createComputePipelineAsync` succeeded; `docs/superpowers/results/phase1/browser-fermat-compile-check.txt`); it was not dispatched or checked against an oracle in the browser.
