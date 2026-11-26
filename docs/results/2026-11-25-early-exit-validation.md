# In-driver early exit: validation against Beacon's CUDA driver (2026-11-25)

Issue #16, branch `feat/early-exit`. Machine: the RTX 4070 Laptop (Linux, NVIDIA 580.178.04, Vulkan), which also hosts Beacon's CUDA
toolkit and a built `g1_fermat_v3` (CUDA/CGBN). Raw logs and scripts: `results/phase1/early-exit/`.

## 1. Correctness: the two drivers agree byte for byte

`ee_validate.py` runs both drivers on the same inputs with the same `--ee`, `--chunk`, `--growth`, `--marker` (and `--bins` where noted)
and compares the verdict files with `cmp` plus the `tested=` and `waves=` figures of the summary lines.

- Inputs: Beacon's own fixtures from its repository: `canary_2048.bin` (400 candidates, half probable primes) and `big_2048.bin`
  (32,000 candidates, half probable primes), both 2,048-bit containers.
- k layouts (sizes files): regular (8 and 40 positions per k), random (1 to 40 and 5 to 120 positions per k, seeded), and one k of
  32,000 positions.
- Parameters: (chunk 6, growth 1.15, marker 3), (2, 1.0, 7), (1, 1.5, 3), (50, 1.0, 3).
- `--bins`: the big file split in two (uneven halves, a k spanning the split), a list file with CRLF line endings, random sizes.

**Result: 19 of 19 comparisons identical** (same verdict bytes, same `tested`, same `waves`).

Two further combinations were not comparable because the CUDA driver itself fails: a single k of 32,000 positions with chunk 2 or chunk 50
needs more than 512 schedule boundaries, and the reference stores them in `int bnd[512]` (stack smashing; for chunk 50 it crashes after
writing a file that still equals ours). `bigmont` has no such limit (it refuses schedules over 1,000,000 boundaries with an error). The
reference's first-bin check also rejects a list line with a trailing space although its own list reader strips it; `bigmont` accepts both.

Plain mode is unchanged (its tests and Beacon's canary comparison from the Phase 1 report still stand). The full workspace suite passes on
the RX 6700 XT at the merge commit (see the PR).

## 2. Speed: correct, but not yet competitive

The same 32,000-candidate file, 800 ks of 40, chunk 6, growth 1.15 (5,360 candidates tested, 2 waves), wall time of the whole process:

| Driver | Wall time | Notes |
|---|---|---|
| CUDA, in-driver early exit | 0.26 to 0.39 s (first run 2.09 s: CUDA warm-up) | gpu-phase 0.057 to 0.063 s, about 90k tests/s |
| `ee_drive.py` waves, plain CUDA driver | 0.53 s | |
| `bigmont`, in-driver early exit | 2.1 to 2.3 s | gpu-phase about 2.08 s, about 2,570 tests/s |
| `ee_drive.py` waves, plain `bigmont` | 2.96 to 2.99 s | |

All four outputs are identical. In-driver early exit is faster than the wave path for `bigmont` (2.1 s against 3.0 s), as designed.

Where the 2.1 s goes (engine summary of that run: `tests 5360 (gpu 5360, cpu 0) | 2.08 s (0.55 s tuning) | ... PRP 2560 | rechecks 2560`):
- **CPU re-check of every probable prime:** 2,560 of the 5,360 tested candidates are probable primes (the fixture is half primes), and the
  engine re-runs each on the CPU at about 2k tests/s, about 1.3 s. A real prime-gap workload is overwhelmingly composite, so this fixture
  overstates the cost; but in early-exit mode every k that retires does so on a probable prime, so the number of re-checks is at least the
  number of ks that hit.
- **Process start:** 0.55 s is the self-test of the cached kernel configuration plus device setup, paid once per process. The feeders start one
  process per mega-ring.
- The kernel time itself is small (5,360 tests at the tuner's measured rate is on the order of 0.1 s).

Follow-up: issue #21 (startup cost and the CPU re-check policy in early-exit mode); the general per-call overhead is #4.

## 3. Not covered

Beacon's gates (`gates/gate_o1.sh`, `gate_r4.sh`, `gate_serve.sh`) were not run; they need the full pipeline (fixtures and CUDA tooling) and
are the next step of the Beacon swap. The feeders (`ee_serve.py`) were read, and their command line parses, but not run against `bigmont`.
One card, one run per cell on the timing table (the first two CUDA rows show warm-up noise).
