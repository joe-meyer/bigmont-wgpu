# In-driver early exit for the bigmont driver (issue #16) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `bigmont` accept and honour Beacon's in-driver early-exit interface (`--bins`, `--ee SIZES`, `--chunk`, `--growth`, `--marker`, `--lock`), so it can be installed as Beacon's `g1_fermat_v3` and run by Beacon's feeders.

**Architecture:** Candidate numbers are grouped into "k"s by a sizes sidecar. The driver tests them in waves; after each wave a k that produced a probable prime is retired and its remaining positions are never tested. Each wave is one call of the existing engine (`Engine::verdicts`) on the candidates still alive. The wave logic is a pure module (`early_exit.rs`) over a small `VerdictSource` trait, so it is tested without a GPU; the CLI wires files, flags, the lock and the summary line around it.

**Tech Stack:** Rust (workspace edition 2024, rustc 1.100), existing `bigmont-cli` crate; std `File::lock` for the lock (no new dependency).

**Spec:** Issue #16 and, authoritatively, `run_ee`, `schedule`, `read_sizes`, `list_files` and `write_verdicts` in Beacon's `cuda/g1_fermat_v3.cu` (djuk8/beacon at commit `ad414e4`, Apache-2.0). Read-only reference on this machine: `N:\Programming\beacon\cuda\g1_fermat_v3.cu` (lines 423-535 and 538-565). The C code is the specification; the semantics are restated below so a task brief is self-contained. Where issue #16 and the C code disagree, the C code wins (issue #16's "`--chunk 0` is one wave" comes from the Python wrapper `ee_drive.py`, not from the driver).

## Global Constraints

- Branch `feat/early-exit` (cut from `main` at the README commit `b6a7163`). Do not push. Nothing under `.superpowers/` or `.claude/` is committed. Commit messages end with `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`; do not reference Beacon's source text in code comments beyond naming the contract (this repository is MIT; do not copy Beacon code, write from the semantics below).
- No Beacon fixture files (Apache-2.0 data) are committed; tests build their own inputs.
- Existing behaviour is unchanged: plain mode (`bigmont CAND VERDICT`), `canary`, `tune`, and every current test keep passing. `--inject` stays unsupported (issue #12).
- Defaults copied from the reference driver: `--chunk 6`, `--growth 1.15`, `--marker 3`.
- Verdict bytes: the engine returns 0 (composite) or 1 (probable prime). This driver never emits the Gerbicz alarm byte `2` (issue #17); a verdict byte of 1 or 2 counts as a hit for retiring a k (the reference checks both; only 1 can occur here).
- Input errors exit with status 2 and a message on stderr (the reference's `exit(2)`); an internal "untouched instance" inconsistency exits 3.
- GPU tests are strict; one engine/Gpu per test where possible. Run the suite as `cargo test --release --workspace --no-fail-fast -- --test-threads=1`. The implementer's machine has an AMD RX 6700 XT; the controller runs the RTX 4070 laptop.
- Scrub machine-specific user paths from any committed log or doc.

## Reference semantics (restated)

- **Command line:** `g1_fermat_v3 INPUT OUT [--bins] [--ee SIZES] [--chunk N] [--growth G] [--marker M] [--lock FILE] [--device N] [--repeat N]`. Early-exit mode is on when `--ee` is given. `--bins` without `--ee` is an error ("`--bins requires --ee sizes.bin`", exit 2). Without `--bins`, `INPUT` is one candidate file; with it, `INPUT` is a text file listing candidate files.
- **List file:** one path per line; trailing `\n`, `\r` and spaces are stripped; empty lines are skipped. Every listed file is a `G2C1` candidate file; all must have the same container `bits` (otherwise "mixed bits in list", exit 2); the candidate array is their concatenation in listed order.
- **Sizes file (`G2S1`):** little-endian: magic `0x31533247` (u32), `nk` (u32), then `nk` u32 counts. `sizes[k]` = number of consecutive candidates belonging to k number `k`. The counts must sum to the total candidate count (otherwise "sizes sum X != count Y", exit 2). Truncated or bad-magic files are an error (exit 2).
- **Schedule** (`schedule(chunk, growth, S)` with `S` = the largest `sizes[k]`): `b = [0]; c = chunk as f64; while b.last < S { step = round_ties_even(c) as i64; if step < 1 { step = 1 }; next = min(b.last + step, S); b.push(next); if growth > 1.0 { c *= growth } }`. (So `chunk <= 0` gives steps of 1.)
- **Waves:** state per k: `off` (index of its first candidate), `size`, `tested = 0`, `alive = true`. For `w in 1..b.len()`: build the list of candidate indices `off + p` for every alive k and every `p` in `[tested, min(size, b[w]))`; if the list is empty, stop; run those candidates through the engine; store each verdict byte; then for every alive k: `hit` = any verdict in the positions just tested is a probable prime; set `tested = min(size, b[w])`; `alive = false` if `hit` or `tested >= size`. Waves counted = number of engine launches (waves that had at least one candidate).
- **Output:** a `G2V1` verdict file (same writer as plain mode, container `bits` = the candidate files' header bits): one byte per candidate, initialised to `--marker`; tested positions hold the engine verdict (0 or 1); skipped positions keep the marker. A k with no probable prime is tested to the end (its bytes equal a plain full run).
- **Lock:** with `--lock FILE`: open (create) the file, take an exclusive lock after the inputs are read and before any GPU work (including opening the engine and tuning), release it after the last wave and before the output is written.
- **Summary line on stderr** (the feeders parse `waves=(\d+)`; keep the whole format): `g1_fermat[EE]: bits=<bits> count=<count> ks=<nk> tested=<tested> (<tested/count %.3f>) waves=<waves> | <secs %.3f>s gpu-phase (lock wait <wait %.2f>s) | <tests/s %.1f> tests/s | prp=<n prp> gec_alarm=<n alarm>`; `secs` is the time from after the lock to the end of the last wave, `tests/s` = tested / secs, `gec_alarm` is always 0 here.

## File Structure

- Modify `crates/bigmont-cli/src/formats.rs`: sizes file and bin-list readers (and a sizes writer for tests).
- Create `crates/bigmont-cli/src/early_exit.rs`: `schedule`, `VerdictSource`, `run_waves`, `EeOutcome`.
- Modify `crates/bigmont-cli/src/lib.rs`: `pub mod early_exit;`.
- Modify `crates/bigmont-cli/src/main.rs`: flags, `cmd_early_exit`, lock, summary line, usage text.
- Tests: `crates/bigmont-cli/tests/early_exit.rs` (pure logic, no GPU), `tests/files.rs` additions, `tests/cli.rs` additions (end to end with the real engine).

---

### Task 1: File readers and the schedule (pure, no GPU)

**Files:**
- Modify: `crates/bigmont-cli/src/formats.rs`, `crates/bigmont-cli/src/lib.rs`
- Create: `crates/bigmont-cli/src/early_exit.rs` (only `schedule` in this task)
- Test: `crates/bigmont-cli/tests/files.rs`, `crates/bigmont-cli/tests/early_exit.rs`

**Interfaces:**
- Consumes: existing `formats::{read_candidates, CandidateFile, CANDIDATE_MAGIC}`.
- Produces (exact signatures later tasks use):

```rust
// formats.rs
pub const SIZES_MAGIC: u32 = 0x3153_3247; // "G2S1" little-endian
pub fn read_sizes(path: &Path) -> Result<Vec<u32>, String>;
pub fn write_sizes(path: &Path, sizes: &[u32]) -> Result<(), String>;
pub fn read_bin_list(path: &Path) -> Result<Vec<PathBuf>, String>;      // >= 1 path, else Err
/// Reads every file of `paths` (all G2C1, same container bits) as one concatenated array.
pub fn read_candidate_files(paths: &[PathBuf]) -> Result<CandidateFile, String>;
// early_exit.rs
pub fn schedule(chunk: i32, growth: f64, s_max: u32) -> Vec<u32>;
```

- [ ] **Step 1: Write the failing tests**

`tests/early_exit.rs` (pure):

```rust
use bigmont_cli::early_exit::schedule;

#[test]
fn schedule_matches_the_reference_semantics() {
    // growth 1.0: constant steps, last one capped at S
    assert_eq!(schedule(6, 1.0, 20), vec![0, 6, 12, 18, 20]);
    // growth 1.15 from chunk 6: steps round_ties_even(6), (6.9), (7.935), (9.12...), ...
    let b = schedule(6, 1.15, 100);
    assert_eq!(b[0], 0);
    assert_eq!(*b.last().unwrap(), 100);
    assert!(b.windows(2).all(|w| w[0] < w[1]));
    assert_eq!(&b[..5], &[0, 6, 13, 21, 30]); // 6, then +7 (6.9), +8 (7.935), +9 (9.12525)
    // chunk <= 0 gives steps of 1
    assert_eq!(schedule(0, 1.15, 3), vec![0, 1, 2, 3]);
    assert_eq!(schedule(-4, 1.0, 2), vec![0, 1, 2]);
    // nothing to test
    assert_eq!(schedule(6, 1.15, 0), vec![0]);
    // round half to even: 2.5 -> 2, 3.5 -> 4 (growth 1.0 keeps c constant, so use chunk 2 and growth 1.25)
    // c: 2.0 -> step 2; c = 2.5 -> step 2; c = 3.125 -> step 3
    assert_eq!(schedule(2, 1.25, 7), vec![0, 2, 4, 7]);
}
```

(The implementer must check each expected vector by hand against the semantics above before trusting it; fix a test only if the hand calculation, not the code, is wrong, and say so in the report.)

`tests/files.rs` additions: `write_sizes` then `read_sizes` round-trips; bad magic, truncated body, a header claiming more entries than the file holds (without allocating them) and an empty file are `Err`; `read_bin_list` strips `\r\n` and trailing spaces, skips blank lines, errors on a file with no paths and on a missing file; `read_candidate_files` concatenates two files in order, and errors with a message containing `mixed bits` when their container widths differ and with the path when a listed file is missing.

- [ ] **Step 2: Run to verify they fail** — `cargo test --release -p bigmont-cli --test early_exit --test files` fails to compile (`early_exit` module, new functions missing).
- [ ] **Step 3: Implement** `schedule` exactly as in "Reference semantics" (use `f64::round_ties_even`; the step is clamped to at least 1; `i64` arithmetic so a huge chunk cannot overflow), and the four `formats.rs` functions with the same hostile-header discipline as the existing `parse_header` (all size arithmetic checked, no allocation before the length is validated). Add `pub mod early_exit;` to `lib.rs`.
- [ ] **Step 4: Run** the focused tests (pass), then the whole `bigmont-cli` suite.
- [ ] **Step 5: Commit** (`Add the sizes and bin-list readers and the early-exit wave schedule`).

---

### Task 2: The wave runner (pure over a fake verdict source)

**Files:**
- Modify: `crates/bigmont-cli/src/early_exit.rs`
- Test: `crates/bigmont-cli/tests/early_exit.rs`

**Interfaces:**
- Consumes: Task 1's `schedule`; `num_bigint::BigUint`; `Engine::verdicts(&mut self, ns: &[BigUint]) -> Result<Vec<u8>, String>` (verdicts 0 or 1 in input order).
- Produces:

```rust
pub trait VerdictSource {
    fn verdicts(&mut self, ns: &[BigUint]) -> Result<Vec<u8>, String>;
}
impl VerdictSource for crate::engine::Engine { /* delegates to Engine::verdicts */ }

#[derive(Debug, PartialEq, Eq)]
pub struct EeOutcome {
    pub verdicts: Vec<u8>,   // one byte per candidate: engine verdict if tested, `marker` if skipped
    pub tested: u64,         // candidates sent to the source
    pub waves: u32,          // source calls with at least one candidate
    pub prp: u64,            // verdict bytes equal to 1 in `verdicts`
}

/// Runs the waves. `sizes` must sum to `numbers.len()` (checked: Err otherwise).
pub fn run_waves(src: &mut impl VerdictSource, numbers: &[BigUint], sizes: &[u32], boundaries: &[u32], marker: u8) -> Result<EeOutcome, String>;
```

- [ ] **Step 1: Write the failing tests** (a `FakeSource` that answers from a lookup table of prime candidates, records every call's input, and can be told to fail), covering:
  1. **Death and survival:** 3 ks of sizes `[5, 5, 3]` with chunk 2 / growth 1.0 (`boundaries = schedule(2, 1.0, 5)`); k0 has its first prime at position 3, k1 has none, k2 has its prime at position 0. Assert the exact verdict bytes (marker on skipped positions), `tested`, `waves`, and that each source call received exactly the positions the semantics require (k2 appears only in wave 1; k0 in waves 1 and 2 and not in wave 3; k1 in all waves).
  2. **Survivor equals a plain run:** a k with no prime has the same bytes as the source's answers for all its positions; and a k whose prime is in the first wave's range is cut at that wave (positions beyond it hold the marker even if the source would say prime).
  3. **Prime at the last tested position of a wave** retires the k at that wave (no further positions requested), and a k whose size is not a multiple of the chunk ends at `size`.
  4. **Edge cases:** zero candidates and zero ks (`waves == 0`, empty verdicts); a k of size 0 in the middle; sizes not summing to `numbers.len()` is an `Err` naming both sums; a source that returns the wrong number of verdicts is an `Err`; a source error is propagated.
  5. **Marker value:** the marker byte is used as given (e.g. 3 and 7), and `prp` counts only bytes equal to 1.
  6. **Equivalence property:** for 200 random size/prime layouts (seeded `XorShift64` from `bigmont_oracle`) and random chunk/growth, `run_waves` output equals a straightforward reference model written independently in the test (loop over ks, simulate waves with the boundaries) byte for byte.
- [ ] **Step 2: Run to verify they fail.**
- [ ] **Step 3: Implement** `run_waves` exactly as in "Reference semantics > Waves". Keep per-k state in a `Vec` of a small private struct; build each wave's index list once per wave; check the source's reply length. Add the `impl VerdictSource for Engine`.
- [ ] **Step 4: Run** focused tests, then the whole crate suite. Mutation check (report outcomes): (a) change `tested >= size` to `tested > size` in the retire rule; (b) skip the retire-on-hit rule; (c) initialise the output to 0 instead of `marker`. Each must fail at least one test; restore after each.
- [ ] **Step 5: Commit** (`Add the early-exit wave runner over a verdict source`).

---

### Task 3: Command line, lock, summary line and end-to-end tests

**Files:**
- Modify: `crates/bigmont-cli/src/main.rs`
- Test: `crates/bigmont-cli/tests/cli.rs`

**Interfaces:**
- Consumes: Tasks 1-2; `Engine`, `Options`; `formats::{read_candidates, read_candidate_files, read_bin_list, read_sizes, write_verdicts}`; std `File::lock`.
- Produces: the driver behaviour described in "Reference semantics".

- [ ] **Step 1: Write the failing tests** in `tests/cli.rs` using the real binary (`CARGO_BIN_EXE_bigmont`) on the CPU backend (`--backend cpu`, so they run anywhere) with small generated inputs (a few dozen 64-130-bit odd numbers: primes and composites; helper to write candidate files with `formats::write_candidates` and sizes with `write_sizes`):
  1. `--ee SIZES` on one candidate file: output header and length correct; for survivors the verdict bytes equal a plain run of the same file; for retired ks, tested positions equal the plain run and later positions equal the marker.
  2. `--bins` with a list of two files (`\r\n` line endings in the list file) equals running the concatenation as one file.
  3. Marker, chunk and growth flags change the output as the semantics require (`--marker 7`; `--chunk 1 --growth 1.0` versus a large chunk).
  4. Errors, all exit status 2 with a message on stderr and no output file left: `--bins` without `--ee`; sizes sum mismatch; a bin list with mixed container bits; bad sizes magic; a missing candidate file.
  5. The stderr summary line exists, starts with `g1_fermat[EE]: bits=` and contains `waves=<n>` with the `n` the test computes independently, and `gec_alarm=0`.
  6. `--lock FILE`: the lock file is created; while another thread holds an exclusive lock on it the process waits and the summary line reports a lock wait of at least the hold time (use a 0.5 s hold); plain mode with `--lock` also works (the flag is no longer ignored).
  7. `--inject` is still refused with a message; plain mode without the new flags behaves exactly as before (existing tests).
- [ ] **Step 2: Run to verify they fail.**
- [ ] **Step 3: Implement.** Remove `--ee --bins --chunk --growth --marker` from `UNSUPPORTED` (keep `--inject`); parse `--bins` (no value), `--ee PATH`, `--chunk N`, `--growth F`, `--marker N` (0-255), `--lock FILE` into `Args`; update `USAGE`. `cmd_verdicts`: when `--ee` is present call a new `cmd_early_exit`; `--bins` without `--ee` is a usage error exiting 2. In `cmd_early_exit`: read inputs (bin list or single file), sizes, validate sums, take the lock (after reading, before `Engine::open`), open the engine, `run_waves` with `schedule(chunk, growth, max size)`, release the lock, write the verdict file, print the engine's usual summary and then the `g1_fermat[EE]:` line. Plain mode with `--lock` takes the lock around `run_engine`. Input errors exit 2; keep the existing exit codes elsewhere.
- [ ] **Step 4: Run** the focused tests, then the whole workspace suite (`cargo test --release --workspace --no-fail-fast -- --test-threads=1`); also run one GPU engine test of your own choosing end to end (`--backend gpu`) on a small generated input and report the output.
- [ ] **Step 5: Commit** (`Add the in-driver early-exit mode to the bigmont driver`).

---

### Task 4: Validation against Beacon's own tools (controller, no subagent)

- [ ] Build `bigmont` on the laptop (RTX 4070, where Beacon's kit and CUDA driver live) and compare, on inputs generated there with Beacon's tooling (its `ee_drive.py` wave path with the CUDA driver as the reference, and its own gates where they can run without the full pipeline): byte-identical verdict files for the same chunk/growth/marker, and the same `tested`/`waves` numbers.
- [ ] Throughput of in-driver early exit against the `ee_drive.py` wave path with the plain binary; record what the per-wave engine overhead (issue #4) costs.
- [ ] Write the results document and update README (`--bins/--ee` supported), issue #16 text (the `--chunk 0` correction), and open the PR (`Closes #16`).

---

## Self-review

**Spec coverage:** flags (Task 3), bin list and sizes files (Task 1), schedule (Task 1), waves and retire rule (Task 2), marker/output (Tasks 2-3), lock (Task 3), summary line with `waves=` (Task 3), acceptance 1-3 of #16 (Task 4).

**Placeholders:** none; the Task 1 schedule test vectors are given with a hand-check instruction because they encode round-half-even arithmetic.

**Type consistency:** `schedule`, `VerdictSource`, `EeOutcome`, `run_waves`, `read_sizes`, `write_sizes`, `read_bin_list`, `read_candidate_files` are spelled the same in every task.
