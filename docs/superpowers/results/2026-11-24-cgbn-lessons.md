# What CGBN does that our kernels do not (design ideas, no code copied)

Date: 2026-11-24. Source read: NVlabs/CGBN `docs/CGBN.md` and `include/cgbn/core/core_mont*.cu` (public, MIT-style licence). Everything below is my own summary of its approach. Per the plan's constraint, no CGBN code is copied into this project; this note lists ideas to reimplement in WGSL.

## 1. How CGBN is organised

- **Size and lane count are compile-time template parameters chosen by the caller**, not discovered at run time: `TPI` (4, 8, 16 or 32 threads per instance) and the bit length are template arguments, and precisions are not mixed in one call. So the "pick lanes from the bit length" idea is not something CGBN does for you; it is something a caller builds on top (a size-class table), as Beacon does by compiling one driver per class.
- **Three Montgomery implementations, one per GPU generation** (named for the instruction they use: a 32-bit multiply-add, a wide multiply-add, and a 16-bit "xmad" multiply-add for older NVIDIA parts). The 16-bit variant is the same idea as our 13-bit limbs: use the cheap small multiply on hardware where the wide one is slow. Choosing the arithmetic per architecture is a design feature, not a hack.
- **Loop structure of the Montgomery multiply:** an outer loop over the lanes (not unrolled) and an inner loop over each lane's limbs (fully unrolled, straight-line code with all limbs in registers). Per limb-iteration it does one broadcast of a word of one operand, a multiply-accumulate of the other operand into the accumulator, one broadcast of the accumulator's lowest word to form the reduction factor, a second multiply-accumulate of the modulus, and a one-word shift of the accumulator that is folded into the accumulate (written one position down) instead of being a separate copy. About three cross-lane shuffles per iteration; ours does four plus per-limb register moves.
- **Carries use hardware carry chains** (multiply-add instructions that set and consume a carry flag), so no carry normalisation is needed during the loop. Cross-lane carries are resolved once at the end with a constant-step propagation based on the lane ballot, not by repeated passes.

## 2. What our own kernel does on the AMD driver (measured, RX 6700 XT, Vulkan; `examples/pipeline_stats.rs`)

| Kernel | VGPRs | Scratch (spill) bytes | Notes |
|---|---|---|---|
| TPI 1 | 170 | 1,920 | our three 160-limb arrays live in scratch memory (343 scratch instructions) |
| TPI 8 | 71 | 0 | no spills |
| TPI 16 | 47 | 0 | no spills |
| TPI 32 | 38 | 0 | no spills |

So the cooperative kernels do not spill. But the saved machine code shows the arithmetic loops were **not unrolled**: the whole program contains only two `v_mad_u32_u24` instructions, about 20 `v_movrel` (indirect register indexing), and 57 to 61 branches. Each limb of work is done in a rolled loop that reads and writes the register arrays through indexed moves, with loop-counter and branch overhead around every multiply-add. That alone plausibly explains a large part of the gap between our 3.5% to 8% of the multiply ceiling and what a straight-line kernel can reach (not yet measured).

## 3. Ideas to try, in the order I would try them

1. **Generate fully unrolled WGSL with named scalar variables** per (limbs, lanes) instead of arrays and loops. The structure: outer loop over the owning lane (uniform, rolled), inner body unrolled over that lane's limbs. gapsieve's `limb13.rs` already generates this style at small sizes. Expected: the largest single gain; size unknown until built.
2. **Fuse the one-limb shift into the accumulate** (compute each new accumulator limb from the old limb above it plus the shifted operand limbs), with the operand and modulus pre-shifted once per squaring. Removes the copies and one of the four shuffles per iteration.
3. **Resolve cross-lane carries once with a ballot** (`subgroupBallot` is in the same WGSL extension) instead of looping light passes under `subgroupAny`.
4. **Per-architecture arithmetic, as CGBN does.** Keep the browser-valid 13-bit-limb variant. Add a native-only variant with 32-bit limbs and 64-bit products for NVIDIA (needs wgpu's `SHADER_INT64`, which is not in WebGPU, so it would sit behind the same backend trait and would not run in a browser). Measure first with gapsieve's `mul_bench` whether 32x32 high-multiply is cheap on each vendor.
5. **Per-adapter autotune of lane width** (and the size-class table from the backlog).

## 4. What this means for the gate

The forecast in `2026-11-24-proposed-gate-v2.md` was that today's kernels would sit at about 2% (RTX 4070) to 4% (RX 6700 XT) of native CUDA. The bar the user set is within an order of magnitude (10%). That needs about a 2.5x to 5x gain from today's kernels. Idea 1 targets exactly the inefficiency found in section 2, so it is the right first experiment. Whether it, plus ideas 2 to 4, gets there is unknown until built and measured, ideally against CGBN on the same card (Test B).
