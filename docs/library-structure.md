# meganum: one library for algorithms on very large numbers (draft spec, 2026-11-27)

Status: **draft for review, nothing implemented.** Decisions below were made in conversation; section 13 lists what is still open.

## 1. Goal and scope

**Long-term goal.** A general Rust library for algorithms on very large numbers, from a few hundred digits to megaprime sizes
(a million digits and more), on GPUs and CPUs, natively and in a browser. Today it does batched base-2 Fermat testing at about
2,000 bits; the roadmap (section 11) grows it towards Miller-Rabin and BPSW, NTT arithmetic, single-number megaprime runs and
special-form tests.

**This spec's scope is phase 1: consolidation.** All of today's logic moves into one library, renamed `meganum` (today's
`bigmont-core`). The command-line binary, the wasm shim and the web pages become thin shells over it. They own only what a library
should not: argument parsing, files and locks, printing, and browser storage. They double as the integration suite: they show how an
application uses the library. Phase 1 builds nothing new beyond the work section 9 names; it gives the later phases their seams.

**One library.** The later phases add modules to the same crate, never new user-facing crates. Optional parts (an NTT arithmetic,
megaprime checkpointing, anything with heavy dependencies) sit behind cargo features, so a small or browser build can leave them
out. Features gate code; they do not split the public structure. If build times ever hurt, the library may split into private
helper crates that `meganum` re-exports; users would not see it.

Non-goals for phase 1: new algorithms, new arithmetic, a new GPU backend, changing any result byte, or changing what `bigmont`
(the binary) does as a drop-in for the CUDA driver.

## 2. Name

`meganum` says "very large numbers" without claiming primes or a method. "bigmont" (big-integer Montgomery) names one arithmetic
inside the library, and "big" suggests a general bignum library (the space of `num-bigint`, `rug`, `crypto-bigint`), which this
is not: it is algorithms on very large numbers. No crate named exactly `meganum` was found in a web search; that is not proof, so
crates.io and GitHub must be checked before the rename. Nothing has been released yet (`0.1.0-alpha.1` is still "TBD"), so the
rename is cheapest now. Whether the repository and the binary are renamed too is open (section 13).

## 3. Layout: organised by algorithm, over a shared arithmetic

```
meganum/src/                    the library (today's crates/bigmont-core)
  reference/      exact big-int arithmetic: the oracle every algorithm is checked against (today's bigmont-oracle)
  arith/          the arithmetic seam (section 5)
    montgomery/   today's arithmetic: limb schemes, the 64-bit CPU arithmetic (today's bigmont-cpu core),
                  KernelConfig, Squaring, the WGSL generators (private)
    ntt/          (later, feature-gated) number-theoretic-transform multiplication
  device/         Gpu, AdapterSelector, device health: any GPU algorithm needs a device, whatever its arithmetic
  tune/           implementation choice, shared by every algorithm: candidates, measure, self_test, autotune, Cache
  algorithm/
    fermat/       verdict logic, the CPU ladder, the GPU kernel mode and runner, the Gerbicz-checked mode, tests
    miller_rabin/ (later) same shape
  executor/       Executor (where a batch runs) and Hybrid (a policy over two executors)
  search/         Groups, WavePlan, RetireRule, Search (step API): runs any test over groups
  formats/        G2C1 / G2V1 / G2S1 as bytes <-> types, no filesystem
  canary/         expected-verdict checks
shells/
  cli/            the `bigmont` binary: args, files, lock, printing
  wasm/           wasm-bindgen glue, browser storage
  web/            pages over the wasm shell
integration tests run the real binary and the web pages
```

Dependencies point down only: `search` over `executor` over `algorithm` over `tune` over `arith` and `device` over `reference`. No
module knows a shell exists. `bigmont-oracle` and `bigmont-cpu` stop being crates (they re-export from `meganum` for one release).

`reference` and the CPU path in `arith/montgomery` are both CPU code and both stay: one is ground truth (exact `BigUint` arithmetic),
the other the production CPU path (Montgomery limbs). The tuner's self-test and the re-check of GPU verdicts use the second; tests
check both against the first.

**Rule: the convenience layer uses only public pieces.** The tuned factories and `Hybrid` are built entirely from public parts.
There is no private back door, so anything they do, an application can do (section 9).

## 4. Algorithms, executors and policies

**Algorithms are the top-level nouns.** Each is a probable-prime test with its own verdict logic, a CPU implementation and a GPU
kernel mode. They share the trait

```
trait ProbablePrimeTest { async fn verdicts(&mut self, ns: &[BigUint]) -> Result<Vec<Verdict>, Error>; }
```

| Algorithm | What it checks | Status |
|---|---|---|
| `Fermat` | `2^(n-1) = 1 mod n` (today's base-2 test, with the optional Gerbicz-checked mode) | exists |
| `MillerRabin` | `n - 1 = 2^s d`; `a^d`, then `s` squarings looking for `-1` or `1` | later |
| `Lucas` / `Bpsw` | a Lucas recurrence; BPSW combines it with a base-2 strong test | later |
| `Proth` / `Llr` | special forms `k 2^n +- 1`, where megaprimes are found | later |

On the GPU a ladder is never run one operation at a time from the host (that would be one dispatch per squaring): each algorithm is
a kernel mode of the generator for its arithmetic, as `Fermat` and `Mul` are today, with its host-side verdict logic. Miller-Rabin's
mode adds a comparison after every squaring. On the CPU the ladder is written once against the arithmetic trait (section 5).

**An `Executor` says where a batch runs:** `Executor::Cpu { threads }` or `Executor::Gpu(KernelConfig)`. It is a value, not a type,
which is what the tuner chooses. An algorithm is constructed on an executor:

- **Concrete factory:** `Fermat::on(Executor::Cpu { threads: 0 })`, `Fermat::on(Executor::Gpu(config))`.
- **Tuned factory:** `Fermat::tuned(&gpu, bits, &cache)` picks the executor and implementation for this adapter and size. The CLI
  and the web client call this. The tuner's choices (arithmetic, scheme, TPI, workgroup, squaring structure) are shared by every
  algorithm that uses the same arithmetic.

**`Hybrid` is a policy, not an executor.** It is today's `Engine`: use the GPU when the batch is big enough (`min_gpu_batch`),
re-check probable primes and disagreements on the CPU, fall back to the CPU on device loss. It holds two instances of an algorithm
and decides between them, so it works for any algorithm.

**Decorator or mode?** A feature is a decorator (a wrapper over `ProbablePrimeTest`) only if it can be written knowing nothing but
the inputs and the returned verdicts. `Hybrid`, device-loss retry and verification sampling qualify. The Gerbicz-Li check does not:
it needs the ladder's state at every 40-iteration block boundary (the running products `A` and `B`, the previous block's `y_prev`,
the Montgomery multiply kernel, and a doubled recomputation of the tail iterations), none of which leave the runner. So "checked"
is an option of the algorithm on both executors, and its only visible effect is the `Retest` verdict. (It applies to any
exponentiation ladder, so later algorithms get the same option.)

`async fn` in traits is not object-safe, so "whichever algorithm or executor" is an enum (`AnyTest`), not a boxed trait object.
The set is closed, the enum is zero-cost, and it works on wasm.

**Verdicts.** One `Verdict` enum for the whole library, replacing today's three overlapping sets (the engine's `Verdict`, the CPU
backend's `Checked`, and raw `u8`): `Composite` (0, certain: a failed Fermat or Miller-Rabin test is a proof), `ProbablePrime` (1,
passed, not proven), `Retest` (2, a Gerbicz alarm). It is `#[non_exhaustive]`. The reference arithmetic returns the same enum, and
differential tests compare `passes()`, not full equality. A skipped candidate is not a verdict; it carries the output marker byte
(section 7).

**Deferred: a proven `Prime`.** No test here proves primality, and at 2,029 bits that needs a different algorithm (ECPP or a
Pocklington-style certificate). When a `PrimalityProof` algorithm exists it takes a `ProbablePrime` and returns `Prime` with a
certificate, or `Unproven`; the variant is added then. The verdict file bytes (0, 1, 2; 3 to 255 are skip markers) would need a
format version bump for it.

## 5. The arithmetic seam

Every algorithm needs the same few operations: enter and leave a modular representation, square, multiply, double, compare. Which
multiplication is fastest depends on size: schoolbook, then Karatsuba or Toom, then NTT or FFT. GMP picks between them with
measured crossover thresholds; here the tuner does the same job, one level below kernel shapes.

- **CPU:** a `ModArith` trait (enter, leave, square, multiply, double, equals) with implementations per tier. CPU ladders are
  written once, generic over it. Today's `MontOps` trait inside the CPU backend (used for fault injection) is its seed.
- **GPU:** the ladder cannot be generic at run time (it is fused into one kernel), so each GPU arithmetic is a generator that emits
  a kernel per algorithm mode. The seam is the generator interface: given a `KernelConfig` and a mode, emit WGSL with the shared
  bindings and contract.
- **Tuner:** candidates span arithmetics as well as kernel shapes; `Cache` entries name the arithmetic.
- **Execution shape:** today's arithmetic keeps a number in a few lanes' registers (up to 1,024 limbs, 32,768 bits) and runs
  thousands of numbers per batch. A megaprime-size arithmetic keeps one number in device memory across the whole GPU, runs for
  hours, and needs checkpoints. Phase 1 does not build the second shape, but keeps the executor and runner interfaces free of the
  assumption that a batch holds many numbers.

Phase 1 only names this seam and moves today's code under `arith/montgomery`; the trait is extracted in phase 2.

## 6. Seams

| Seam | Today | After |
|---|---|---|
| Cache | `PathBuf` inside `Options`; `std::fs` and a process-wide map in `tune` | a `Cache` trait (lookup, store); an in-memory impl in the library, a file impl in the CLI, localStorage in wasm. No process-wide mutable state in the library |
| Files | `formats` reads `&Path` | the library parses `&[u8]` and writes to `impl Write`; the shell does the I/O |
| Logging | `eprintln!` in engine and tune | an `Observer` trait with events (tuned, device lost, recheck, alarm...) |
| Blocking | `pollster` inside the tuner | all `async`; shells block once at the top (`pollster::block_on`) |
| Threads | `std::thread` inside the CPU batch; traps on wasm32 (issue #48) | the CPU path runs single-threaded on wasm32 and threaded elsewhere, gated by `cfg(target_arch)` as AGENTS.md requires, not by a feature |
| Environment | `AdapterSelector::from_env` in the library | the shell reads env vars and passes values |
| Fault injection | `Options::inject_*` | a cargo feature, for tests |
| Locking (`--lock`) | the CLI | stays in the CLI: it is process-level |

## 7. Search

`Search` is a pull-based driver, not an `Iterator`. A batch of candidates from many groups per GPU dispatch is the unit of work, so
a per-item iterator or stream would starve the GPU.

```
let mut search = Search::new(groups, WavePlan::default(), FirstHit);
while let Some(wave) = search.step(&mut test).await? { /* progress, cancel, yield to the page */ }
let outcome = search.finish();
```

The test is passed in (`&mut impl ProbablePrimeTest`), so the same search runs Fermat now and Miller-Rabin later. The pieces of
today's `run_waves`, separated:

- **`Groups`**: candidates plus group boundaries (data only), built from a sizes file by `formats` or by hand. The public API says
  "group"; "k" stays in the CLI's help text.
- **`RetireRule`**: after each wave, per group still alive, retire or keep. `FirstHit` retires a group when any verdict in the
  wave is `ProbablePrime` or `Retest` (a Gerbicz alarm retires the group so the candidate is retested, as today), or when the group
  has been tested to its end. Whether this is a trait or a plain function is open (section 13).
- **`WavePlan`**: which positions to test next. A wave is a range of positions applied to every group still alive. Today's plan is
  the stateless boundaries from `schedule(chunk, growth, s_max)` and is the default, so output stays byte-identical with the
  CUDA driver's. The plan is a small trait (`next_end(state) -> position`) so an adaptive plan that targets a batch size can be
  added without touching the driver.
- **Marker byte**: encoding skipped positions (3 to 255) belongs to the output side (`formats`), not the driver. Beacon also reads
  `0xFF` as FATAL (its fold aborts); bigmont never writes it today. `formats` owns that encoding too, so a failed run can be written
  as an explicit FATAL file instead of no file.

`Search::first_hit(groups, plan)` is a preset that builds the above; the name follows the rule, not the domain, because a hit is
only a *probable* prime.

## 8. Orchestration (gap hunting and similar)

The library owns the search, the tests and the executors. An application owns the pipeline around them:

```
CandidateSource (sieve) -> Groups -> Search::step over a test -> Outcome -> Sink (gap logic, persistence)
```

The sieve runs ahead of the GPU on the CPU. Multi-GPU is one test instance per device, each pulling batches from the
application's queue. `CandidateSource` and `Sink` are small traits in the library so the integration suite can show a worked
pipeline; the sieve, gap arithmetic, queue and persistence live in the application (`gapsieve`, Beacon). Whether the library
should also ship a generic `Pipeline` is not decided; the default is no.

## 9. Performance rules (what must not be lost)

The split must not cost throughput or control. Each is a requirement with a test.

| Control | Requirement |
|---|---|
| Config override | `KernelConfig` (TPI, workgroup, squaring, scheme) is a public input to `Executor::Gpu`; the tuner is a default |
| Batch size | capacity is a parameter; `WavePlan` is pluggable; the tuner times candidates on the engine's batch size |
| Overlap | the GPU path splits *submit* from *collect*, so several dispatches can be in flight. **Probably new: as far as I have read, `fermat_async` awaits each dispatch to completion; to be confirmed against `runner.rs` before the plan** |
| Host prep | accept packed limb words as input, not only `BigUint`, so a pipeline that already holds limbs skips the conversion (1% to 2.7% of wall today) |
| Buffers | the GPU path keeps its buffers across calls |
| Dispatch | no virtual calls in the hot path (enum dispatch; concrete types inside) |
| Measurement | `RunStats` per dispatch stays public |
| Multi-GPU | no process-wide mutable state; one value per device |
| Device-resident waves | a search uploads its candidates to the device once and each wave selects positions by index, as Beacon's CUDA driver does ("loads bins -> device once; waves in-kernel by index"). **New: today `run_waves` clones and re-uploads every wave's candidates** |
| Phases | preparing a batch (host conversion, upload) is separable from running it on the GPU, so a shell can hold a per-card lock only around the GPU phase and let siblings prepare meanwhile (Beacon runs 4 processes per card this way) |

## 10. Checked against Beacon

Beacon (the prime-gap pipeline this library serves) was read for what it needs. Its pipeline is residue cache, sieve (C and CUDA),
Fermat driver, early-exit fold, harvest, gap walk, notary, and ECPP certification of endpoints.

| Beacon needs | Covered by | Gap |
|---|---|---|
| batched base-2 Fermat at the 2,048-bit class, the drop-in driver CLI and byte formats | `Fermat`, `formats`, the CLI shell | none |
| early exit with byte-identical prefixes, schedule chunk 8, growth 1.5, round-half-even | `Search` with the default `WavePlan` | none (pinned by tests) |
| mega-rings: many candidate files in one process (`--bins list`) | `formats` concatenation | none |
| device-resident waves | performance rules (section 9) | **new work** |
| lock held only around the GPU phase | performance rules (phases) | **new work** |
| `0xFF` FATAL verdict files | `formats` | **new work** |
| optional Gerbicz check (Beacon's v8.55 driver dropped it for an hourly canary) | checked mode, `canary` | none |
| the sieve, queue, claims, ledger, venue tuples, candidate order within a group | the application (section 8) | none: Beacon's |
| strong probable-prime tests for the notary and walk (it uses gmpy2 today) | `MillerRabin`, `Bpsw` | roadmap phase 6 |
| ECPP certificates for gap endpoints (it uses PARI today) | deferred `PrimalityProof` | roadmap phase 7 |
| venues beyond 16,000 bits (PFGW past 32,768 bits) | `KernelConfig` allows up to 1,024 limbs (32,768 bits with 32-bit limbs) | beyond that needs NTT/FFT arithmetic (roadmap phases 3 and 4) |
| a GPU sieve in the library one day (issue #42) | not a `ProbablePrimeTest` | a second family of nouns (a `CandidateSource` implementation), not a test |

## 11. Roadmap (each phase gets its own spec and plan)

1. **Consolidation** (this spec): one library, the layout above, the seams, thin shells, the rename, and the new work of section 9.
2. **Arithmetic trait:** extract `ModArith` from today's CPU code, generalise the generator interface, let the tuner choose between
   arithmetics.
3. **NTT arithmetic:** integer NTT multiplication first, because WGSL and WebGPU have no f64; f64 FFT later where a native backend
   offers it.
4. **Large-number execution:** one number per device, held in device memory, long runs with checkpoints and resume, the Gerbicz
   check on by default.
5. **Special-form tests:** `Proth` and `Llr` for `k 2^n +- 1`.
6. **Strong general tests:** `MillerRabin`, `Bpsw` (Beacon's notary and walk).
7. **Proofs:** `PrimalityProof` (certificates), and with it the proven `Prime` verdict.

Phases 2 to 7 are ordered by dependency, not by priority; 6 can come before 3 to 5 since it reuses today's arithmetic.

## 12. Phase 1 migration, in order

Each step is its own PR, keeps `cargo test --workspace` green, and keeps every output byte identical.

1. Tuner into the library: async `candidates`, `measure`, `self_test`, `autotune`; a `Cache` trait; the cache value format. The CLI
   keeps a file cache.
2. `formats` into the library as `&[u8]` parsing and `Write` output, including the `0xFF` FATAL file; the CLI does the I/O.
3. One `Verdict` enum across the reference, the CPU path, the runner and the engine.
4. `bigmont-oracle` becomes `reference`; `bigmont-cpu`'s arithmetic joins `arith/montgomery` and its ladder joins
   `algorithm/fermat` (the old crates re-export for one release); the CPU path gets its single-thread form on wasm32; device handling
   moves to `device/`.
5. `algorithm/fermat` gathers today's runner, Fermat kernel mode and the Gerbicz-checked mode; `Executor` replaces the scattered
   backend choices; the engine into the library as `Hybrid`, with `Observer` and fault injection behind a feature.
6. `Search` split (`Groups`, `WavePlan`, `RetireRule`) with the step API; candidates uploaded once and waves selected by index on the
   device; `schedule` and `run_waves` become thin wrappers.
7. Public surface: `prelude`, the `ProbablePrimeTest` trait, factories, tiers documented; submit/collect split on the GPU path, with
   a separate prepare phase so a shell can lock only the GPU phase.
8. Shells thin (`shells/cli`, `shells/wasm`, `shells/web`): no logic left in them; the integration suite runs the real binary and
   the web pages.
9. Rename `bigmont-core` to `meganum` (after the name is confirmed free), with the old crate names re-exporting for one release.

Every step touching the library also needs the wasm32 compile check and, per AGENTS.md, the browser pages run in a WebGPU browser.

## 13. Open questions

- Name: confirm `meganum` is free on crates.io and GitHub. Rename the repository (today `bigmont-wgpu`) too? Keep the binary
  called `bigmont` (the drop-in for Beacon) or rename it? Default: keep the binary name, rename the library and the repository.
- Rename first or last? The plan does it last (step 9) so the moves happen under known names; doing it first avoids touching every
  import twice.
- `RetireRule` as a trait now, or a plain `FirstHit` function until a second rule exists. Default: the function.
- `AnyTest` enum against boxed futures: the spec says enum; confirm when writing the checked-mode option on the GPU executor.
- Whether the CPU path and `reference` get cargo features so a browser build can leave the CPU path out. Default: always compiled;
  the wasm32 single-thread path makes it harmless.
- A generic `Pipeline` in the library (section 8). Default: no.
- Sequencing against the open performance work (the symmetric squaring branch and the tuner fix touch `tune` and `kernel`): step 1
  should start from `main` after that PR merges.
