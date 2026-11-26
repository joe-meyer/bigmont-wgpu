# Instruction-throughput probe: which arithmetic could beat 13-bit multiply-adds? (2026-11-24)

Tool: `crates/bigmont-core/examples/mul_probe.rs` (8 independent dependent chains per lane, best of 5 dispatches, wall-clock; adapted from gapsieve's `mul_bench`). Measured at commit `cc56b72` plus the new example. Each "step" includes the small extra operations written in the probe kernel (masks, an add, for the u64 step a shift and two adds), so **only ratios between rows on the same GPU mean anything; absolute rates are lower bounds** (our real unrolled kernel already runs 1.6x above the probe's own 13-bit "ceiling" on the 4070, because its multiply-adds need fewer extra instructions than the probe's step).

## Measured (G chain-steps per second)

| Instruction | RX 6700 XT (Vulkan) | RX 6700 XT (Dx12) | RTX 4070 Laptop | Radeon 780M |
|---|---|---|---|---|
| add only (baseline) | 2,362 | 2,325 | 4,249 | 2,019 |
| 24-bit multiply-add (our 13-bit limbs) | 3,128 | 987 | 2,967 | 1,005 |
| 32-bit multiply low + add | 1,168 | 1,148 | 4,392 | 409 |
| 64-bit multiply-add (32x32 to 64, carry shift, adds) | 455 | 515 | 1,007 | 137 |
| 4-wide 8-bit dot product (`dot4U8Packed`) + add | 2,331 | 2,144 | 2,487 | 1,392 |
| f32 fma | 3,232 | 2,896 | 4,307 | 2,996 |

## Rough relative cost per 2,029-bit squaring if a kernel were built only from that instruction (lower is better)

Counts: 13-bit limbs 51,200 steps (2 x 160^2); 32-bit limbs with 64-bit products 8,192 steps (2 x 64^2), counting each probe step as one limb product including its carry handling; 8-bit digits with 4-wide dot products 32,768 steps (2 x 256^2 / 4), ignoring all packing and column-arrangement overhead. Units are steps divided by G steps/s.

| | 13-bit mad24 | 32-bit limbs, u64 | 8-bit digits, dot4 |
|---|---|---|---|
| RX 6700 XT (Vulkan) | 16.4 | 18.0 (about 10% worse) | 14.1 (about 16% better) |
| RTX 4070 Laptop | 17.3 | 8.1 (about 2.1x better) | 13.2 (about 1.3x better) |
| Radeon 780M | 50.9 | 59.8 (about 15% worse) | 23.5 (about 2.2x better) |

## Reading it (estimates, not predictions)

- On AMD the 64-bit path is slow (hardware multiplies are quarter rate and 64-bit adds expand), so 32-bit limbs with 64-bit products would not help there; the 13-bit scheme is already close to the best of these on the RX 6700 XT.
- On NVIDIA a 32-bit-limb, 64-bit-product variant looks like the biggest lever: perhaps 1.3x to 2x over the 13-bit kernel once the probe's own overhead is discounted (the 2.1x above is the optimistic end). That would move the RTX 4070 from about 35% of CUDA to roughly 45% to 70%. It needs wgpu's `SHADER_INT64`, so it is native-only and not valid in a browser.
- The 4-wide dot product is attractive on the Radeon 780M (up to about 2x in this crude model) but needs a convolution-style loop restructure whose packing overhead is unknown; it is browser-valid (a WGSL language feature).
- Squaring symmetry (up to about 25% fewer multiply-adds) applies on every vendor but is not in this table.
- No single approach wins everywhere, which is why CGBN ships a different arithmetic per GPU generation.

Not measured: any of these inside a real kernel, dot-product packing cost, the Intel and Apple GPUs, Dx12 on NVIDIA.
