# Proposed go/no-go gate, version 2 (PROPOSED, for the user to accept, change or reject)

Date: 2026-11-24. This replaces the 6,000 projected tests/s proposal in section 8 of `2026-11-24-phase0-results.md`. It is a draft for you to decide on, not a decision and not a verdict on the Phase 0 results.

## 1. What the gate is for

Spec section 4 asks one question of Phase 0: is a portable (wgpu) engine fast enough to be worth building, or should the project fall back to native per-vendor code (or just the CPU)? A good gate answers that with comparisons that can be measured on the same hardware, not with absolute numbers borrowed from other cards.

## 2. Why the old gate (6,000 projected tests/s on the RX 6700 XT) should be dropped

- It rests on Beacon's "516-523k tests/s per card", whose meaning I could not confirm (full tests or early-exit tests) and whose card is much larger than anything tested here.
- It scales that figure by shader count (21,760 / 2,560 = 8.5, from memory, unverified), then divides by 10 as a judgement. Each step is a guess, and they multiply.
- The re-timed results (addendum section 3a) show no measured row reaches it (best 3,941, the Radeon 780M; 6700 XT 2,475). A gate no card can meet, set from unverified inputs, tells us nothing about which way to go.

## 3. Evidence available now (all projected tests/s = squarings/s / 2029; GMP rows are measured full Fermat tests)

| Row | tests/s | Source |
|---|---|---|
| Native CPU, GMP, one core of a 2013 Xeon E5-2690 v2 | about 265 | measured (`tools/cpu-baseline/`) |
| Native CPU, GMP, 4 vCPUs of the same Xeon (shared host) | about 1,085 | measured |
| Quadro P4000 (shared), best wgpu row | 467 | projected from 0.95 M squarings/s |
| RTX 4070 Laptop, best wgpu row | 1,682 | projected |
| RX 6700 XT, best wgpu row (Dx12 + dxcompiler / Vulkan) | 2,475 / 2,406 | projected |
| Radeon 780M, best wgpu row | 3,941 | projected |
| Beacon CUDA on an RTX 5090-class card | 516,000 to 523,000 | README claim, meaning unconfirmed, different card |

No CUDA or CGBN measurement exists on any card here, and no CPU measurement exists on the laptop or the desktop.

## 4. Proposed gate: three tests, all measured on the same machine as the GPU

**Test A: the GPU path must be clearly worth having over the CPU.** On each target machine, the best wgpu configuration must reach at least **3x the native CPU rate (GMP, all cores, same machine)**. Reason: a CPU path is needed anyway, so a GPU path that is not far ahead of it is not worth its complexity.
- Status: cannot be evaluated properly yet, because the only CPU measurement is a 2013 Xeon slice. Against that slice only the Radeon 780M passes (3.6x; the 6700 XT is 2.3x, the 4070 1.5x, the P4000 0.4x). A modern desktop CPU will be faster per core and may have more cores, so the true bar on the desktop and laptop could be higher than any GPU row.
- Needed: GMP timings on the desktop and the laptop (Python `gmpy2`, a small user-level wheel, or building GMP in a user directory; both are small downloads that need your OK).

**Test B: the portability tax against native GPU code on the same card.** On the RTX 4070 laptop, compare the best wgpu configuration with Beacon's own CUDA/CGBN Fermat driver run on the same card, same size class. Proposed reading, as the wgpu fraction of native:
- 33% or more: go (a portable engine at a third of native speed is a good trade for running everywhere);
- 10% to 33%: go only if the portability and web value clearly outweigh the speed loss, your call;
- below 10%: stop treating wgpu as the main engine; keep it as a fallback and add native backends.
- Status: not measured. This needs CUDA on the laptop (several GB: either `apt` with sudo, or NVIDIA's installer in a user directory; plus a small clone of NVlabs' CGBN) and building Beacon's driver for the 4070. I would not start that without your consent for the download. It is the single measurement that replaces the guesswork in the old gate.

**Test C: headroom.** Current kernels reach only about 3.5% to 8% of the card's measured `mad_u32_u24` ceiling (addendum section 4 and the main doc section 4, with the caveat that the ceiling is itself uncertain). Pass if a short list of concrete optimisations has a credible path to at least 2x (candidates, none measured: raising the carry-pass period from 16 toward 24 to 31, using the squaring symmetry that the kernel does not exploit, fewer shuffles per limb, per-vendor tuning of the 4070's slow TPI 16 and its wg 128/256 penalty). This test is a judgement and should be reviewed by someone before it counts.

**Overall:** go if A passes on at least the machines you care about and B is at least 10% (with 10% to 33% your call), and C is credible. No go if B is below 10%, or if A fails everywhere even after a fair CPU baseline.

## 5. Where this leaves the current numbers

- Test A: provisional pass only on the Radeon 780M, against a weak CPU baseline. Unknown elsewhere.
- Test B: unmeasured.
- Test C: plausible but unmeasured.
So the honest statement is "undecided", not "met" or "failed". Two measurements settle most of it: native CPU numbers on the desktop and laptop (small), and CGBN on the 4070 (large download).

## 6. Decisions for you

1. Adopt this three-test gate, change its thresholds (3x, 33%, 10%, 2x are my proposals, not derived values), or keep the old 6,000 number.
2. Whether I may measure the CPU baseline on the desktop and laptop (small downloads: `gmpy2` wheels or GMP source).
3. Whether I may install CUDA on the laptop for Test B (several GB; I'd ask which route you prefer: `apt` with sudo, or a user-directory installer).
