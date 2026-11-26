# Limb-scheme factory: validation (2026-11-25)

Branch `limb-factory` (cut from `phase1` at d7aeb10), head `bc10df3`, not pushed. Plan:
`docs/superpowers/plans/2026-11-25-limb-scheme-factory.md` at commit 9d35813 (since removed from the tree; see git history). Raw logs: `results/phase1/factory/` and, for the
experiments that led here, `results/phase1/schemes/` and `results/2026-11-24-limb-scheme-experiment.md`.

## 1. What changed

- The limb width is a kernel parameter. `LimbScheme::{Limb13, Limb32}` owns everything that depends on the radix
  (limbs for a bit length, R, host pack/unpack, `-n^-1`, the oracle chain). `KernelConfig` carries a `scheme`
  and `requires() -> Caps` (subgroups, int64).
- `Limb32` is a cooperative kernel with 32-bit limbs and u64 products (`crates/bigmont-core/src/limb32.rs`), in
  squaring and Fermat-ladder modes. Carries leave a lane's last limb one step late and are added to the same lane
  by the same lane (no ripple inside the loop); a final resolve loop ripples the leftover carries.
- It is native-only. `try_wgsl_for(Target::Browser)` is an `Err` that names `int64`; a naga matrix test validates
  every generated source (squaring and Fermat, both schemes, both targets) with exactly that target's
  capabilities.
- The tuner enumerates (scheme, TPI, workgroup) for what the adapter can run, self-tests and times each, and caches
  the winner. No vendor or device name is consulted. `--limbs auto|13|32` forces or leaves it to the tuner.
- Cache: 13-bit winners keep the line text `unrolled <tpi> <wg>|<rate>`; 32-bit winners are `unrolled32 ...`. The Auto
  key now carries a token (`name|backend|driver|any|<class>`) so caches written before this change are
  re-measured once (otherwise they would pin a 13-bit winner on a card where 32-bit is faster); forced runs use
  `only-13|only-32`.
- A tuned shape is applied per exact bit length with `retarget(bits)`. A 32-bit kernel's limb count can shrink
  inside one 13-bit tuning class (for example 256 bits needs 9 limbs, 252 bits needs 8), so the engine self-tests
  a retargeted kernel once per (class, limb count) before using it.

## 2. Correctness

| Machine | Backend | Workspace suite at `bc10df3` |
|---|---|---|
| RTX 4070 Laptop (Linux, NVIDIA 580.178.04) | Vulkan | 163 passed, 0 failed |
| RX 6700 XT (Windows) | Vulkan | 163 passed, 0 failed |
| Quadro P4000 (host2 container, shared) | Vulkan | 163 passed, 0 failed |

(`workspace-tests-*-final.txt`; `cargo test --release --workspace --no-fail-fast -- --test-threads=1`.)

- The 32-bit squaring kernel is checked against the oracle at TPI 2 to 32 (TPI 4 to 32 in the factory API),
  64 to 2,029 bits, including all-ones and sparse moduli and x = 0, 1, n - 1. The Fermat ladder is checked
  exhaustively for odd n in 3..4095 at TPI 4 and 8, against Mersenne numbers (including the 2^2029 - 1 pseudoprime),
  at word and limb-class boundary lengths, and through segmented dispatches.
- Mutation checks: deleting the `a >> 32` term from the deferred carry fails every 32-bit squaring test; replacing
  the lane-below bit in the Fermat doubling by 0 fails 7 of 11 Fermat tests. Deleting the `wrap` term is NOT
  caught by any test (see section 5).
- Engine verdicts for the 131,075-candidate 2,029-bit file are byte-identical across `--limbs 13`, `32` and `auto`
  on each machine, and on the 4070 equal to the Phase 1 verdict file (sha256 prefix 76b44291e86faa81).

## 3. Speed (what the tuner picks and what the engine does)

Tuner at 2,029 bits, best kernel per scheme, squarings/s (`tune-4070-auto.txt`, `tune-and-engine-*.txt`):

| GPU | best 13-bit | best 32-bit | Auto picks |
|---|---|---|---|
| RTX 4070 Laptop | 85.7 M (TPI 4, wg 256) | 152.5 M (TPI 4, wg 256) | 32-bit, 1.78x |
| Quadro P4000 | 12.7 M (TPI 8, wg 256) | 20.9 M (TPI 16, wg 256) | 32-bit, 1.65x |
| RX 6700 XT | 84.2 M (TPI 4, wg 128) | 49.6 M (TPI 8, wg 256) | 13-bit; 32-bit is 0.59x |

Engine, `bigmont CAND OUT --backend gpu`, tests/s excluding tuning, runs after the first (the cache is warm). The 4070
and 6700 XT used a 131,075-candidate file, the P4000 a 16,387-candidate file (compare with its Phase 1 figure of
5,239):

| GPU | `--limbs 13` | `--limbs 32` | `--limbs auto` |
|---|---|---|---|
| RTX 4070 Laptop | 35,740 / 35,701 | 61,038 / 61,232 | 61,299 / 61,299 (63,330 / 61,701 with the final binary) |
| Quadro P4000 | 5,190 / 5,503 | 7,834 / 7,739 | 8,153 / 8,451 |
| RX 6700 XT | 32,871 / 27,463 | 20,862 / 20,802 | 31,916 / 30,523 |

- On the RTX 4070 the engine went from 27% to about 46% of Beacon's 132,096 tests/s (53% of its 116,294 single-pass
  figure). The tuner's best 32-bit kernel does 1.525e8 squarings/s; at about 2,028 squarings per test that is about
  75k tests/s for the kernel alone (my arithmetic, ignoring the doublings), about 57% of Beacon's rate. The rest of
  the gap is the kernel (about 1.8x) and engine overhead (about 18% between 75k and 61k).
- The scheme that wins is not a vendor question: the Pascal P4000 and the Ada 4070 both choose 32-bit limbs, the
  RDNA2 6700 XT chooses 13-bit limbs. The tuner measures, and a card or driver we have not tried gets the right
  answer without a code change.
- Tuner cost on a first run: 13.5 s (4070), 29 s (P4000, shared GPU), 16 s (6700 XT) for 24 candidates; later runs
  read the cache (about 0.6 to 1.5 s of self-test).

## 4. Limits of these measurements

- One to three runs per cell, no variance estimates. The 6700 XT is the desktop the user also uses: its 13-bit
  figures varied from 27.5 k to 32.9 k and its first Auto run gave 24.3 k. The P4000 shares its card with three other
  guests (plex, frigate, llm), so its speeds are noisy; its correctness results are clean. GPU clocks were not locked.
- The engine speeds were measured at `4dce4ec` for all three machines; the 4070 Auto run was repeated at the final
  commit (`bc10df3`, same results). The two later commits change the cache key and add the retarget self-test, which
  does not trigger at 2,029 bits (the retargeted kernel has the same limb count), so they do not change these
  speeds. The 6700 XT and P4000 engine rows were not repeated.
- The 13-bit TPI 8 kernel at workgroup 64 runs at 3.9 M squarings/s on the 6700 XT, against 84 M for TPI 4. That
  was so before this work (see the Phase 1 report) and was not investigated here.
- Not tested: an adapter without `SHADER_INT64` (the capability rule and pure tests cover that path), Intel, Apple,
  DX12/Metal int64 behaviour, and a real browser compile of the 13-bit sources (naga validates them, Tint/Dawn was
  not run).
- The `wrap` branch of the 32-bit carry logic (a u64 overflow of `q + pend`) needs a modulus limb and a Montgomery
  factor both within a few units of 2^32 - 1 at the same step, about a 2^-30 event per element, so no test reaches
  it. It was reviewed twice by arithmetic argument (documented on `step` in `limb32.rs`).
- Not done: hardware carry/extended multiply (WGSL has none, so the compiler picks the instructions); squaring
  symmetry; any inspection of what the NVIDIA compiler emits for the 32-bit kernel; the engine latency work.

## 5. Process notes

Every task was implemented by a subagent and reviewed; the reviews produced fix rounds that were applied: a missing
Fermat-source check in the naga matrix, one device per test in the 32-bit size tests, two existing tests that assumed
Auto picks 13-bit (they would have failed on the 4070), the engine test lengths that never crossed a 32-bit limb
boundary, the Auto-cache key, and the retarget self-test. One review subagent ran `git checkout` in the main
checkout and switched back; the tree was verified clean afterwards.
