# One library, thin shells: structure and public API (draft spec, 2026-11-27)

Status: **draft for review, nothing implemented.** Decisions below were made in conversation; section 9 lists what is still open.

## 1. Goal

All of the project's logic lives in one library, `bigmont-core`, usable natively and in a browser. The command-line binary, the
wasm shim and the web pages become thin shells over it. They own only what a library should not: argument parsing, files and
locks, printing, and browser storage. They double as the integration suite: they show how an application uses the library.

Non-goals: new algorithms, a new GPU backend, changing any result byte, or changing what `bigmont` does as a drop-in for the CUDA
driver.

## 2. Layout

```
crates/bigmont-core/src/        the library
  reference/   exact big-int arithmetic oracle (today's bigmont-oracle)
  cpu/         64-bit-limb Montgomery backend (today's bigmont-cpu)
  kernel/      KernelConfig, Squaring, LimbScheme (public); WGSL generators (private)
  gpu/         Gpu, AdapterSelector, device health
  runner/      Runner: the Fermat ladder and the Gerbicz check on a device
  tune/        candidates, measure, self_test, autotune, the Cache trait
  engine/      the hybrid: tuned GPU + CPU re-check + device-loss recovery
  search/      Groups, WavePlan, RetireRule, Search (step API)
  formats/     G2C1 / G2V1 / G2S1 as bytes <-> types, no filesystem
  canary/      expected-verdict checks
crates/bigmont-cli/             bin only: args, files, lock, printing
crates/bigmont-wasm/            wasm-bindgen glue, browser storage
web/                            pages over bigmont-wasm
integration tests               run the real binary and the web pages
```

Dependencies point down only (`engine` over `tune` and `cpu` over `runner` over `kernel` and `gpu` over `reference`). No module
knows a shell exists. `bigmont-oracle` and `bigmont-cpu` stop being crates; they become the `reference` and `cpu` modules.

**Rule: the convenience layer uses only public pieces.** `Engine` and the tuned factory are built entirely from the public tier-2
parts. There is no private back door, so anything the facade does, an application can do (section 7).

## 3. Algorithms are the top-level nouns

Each is a trait that says what it computes, not how.

| Algorithm | Meaning | Today's code |
|---|---|---|
| `FermatTest` | verdicts for a batch of odd numbers (base-2 probable prime) | `Runner`, `Engine`, `fermat_base2` |
| `Search` | iterates a `FermatTest` over groups (section 5) | `schedule::run_waves` |

Implementations are strategies: `CpuFermat`, `GpuFermat(KernelConfig)`, and `HybridFermat` (today's `Engine`).

**Decorator or mode?** A feature is a decorator only if it can be written knowing nothing but the inputs and the returned
verdicts. `HybridFermat` (GPU first, CPU re-check of probable primes and disagreements), device-loss retry and verification
sampling qualify. The Gerbicz-Li check does not: it needs the ladder's state at every 40-iteration block boundary (the running
products `A` and `B`, the previous block's `y_prev`, the Montgomery multiply kernel, and a doubled recomputation of the tail
iterations), none of which leave the runner. So "checked" is an option on `CpuFermat` and `GpuFermat`, and its only visible effect
is that a mismatch yields the `Retest` verdict.

**Two ways in.** A concrete factory returns a specific implementation (`Fermat::cpu()`, `Fermat::gpu(&gpu, config)`). An abstract
factory returns the best implementation for this adapter and size, using the tuner and a `Cache` (`Fermat::tuned(&gpu, bits, &cache)`).
The CLI and the web client call the abstract one. The kernel choices made by the tuner (scheme, TPI, workgroup, squaring
structure) are the reason it picks one `GpuFermat` over another; they are not part of the algorithm's name.

`async fn` in traits is not object-safe, so "return whichever implementation" is an enum, `AnyFermat`, not a boxed trait object.
The set of backends is closed, the enum is zero-cost, and it works on wasm.

**Verdicts.** One `Verdict` enum for the whole library, replacing today's three overlapping sets (the engine's `Verdict`, the
CPU backend's `Checked`, and raw `u8`): `Composite` (0, certain: a failed Fermat test is a proof), `ProbablePrime` (1, passed, not
proven), `Retest` (2, a Gerbicz alarm). It is `#[non_exhaustive]`. The reference arithmetic returns the same enum, and differential
tests compare `passes_fermat()`, not full equality. A skipped candidate is not a verdict; it carries the output marker byte
(section 5).

**Deferred: a proven `Prime`.** No test we run proves primality, and at 2,029 bits that needs a different algorithm (ECPP or a
Pocklington-style certificate). When a `PrimalityProof` algorithm exists it takes a `ProbablePrime` and returns `Prime` with a
certificate, or `Unproven`; the variant is added then. The verdict file bytes (0, 1, 2; 3 to 255 are skip markers) would need a
format version bump for it.

## 4. Seams

| Seam | Today | After |
|---|---|---|
| Cache | `PathBuf` inside `Options`; `std::fs` and a process-wide map in `tune` | a `Cache` trait (lookup, store); an in-memory impl in core, a file impl in the CLI, localStorage in wasm. No process-wide mutable state in the library |
| Files | `formats` reads `&Path` | core parses `&[u8]` and writes to `impl Write`; the shell does the I/O |
| Logging | `eprintln!` in engine and tune | an `Observer` trait with events (tuned, device lost, recheck, alarm...) |
| Blocking | `pollster` inside the tuner | all `async`; shells block once at the top (`pollster::block_on`) |
| Threads | `std::thread` inside the CPU batch; traps on wasm32 (issue #48) | the CPU module runs single-threaded on wasm32 and threaded elsewhere, gated by `cfg(target_arch)` as AGENTS.md requires, not by a feature |
| Environment | `AdapterSelector::from_env` in core | the shell reads env vars and passes values |
| Fault injection | `Options::inject_*` | a cargo feature, for tests |
| Locking (`--lock`) | the CLI | stays in the CLI: it is process-level |

## 5. Search

`Search` is a pull-based driver, not an `Iterator`. A batch of candidates from many groups per GPU dispatch is the unit of work, so
a per-item iterator or stream would starve the GPU.

```
let mut search = Search::new(groups, WavePlan::default(), FirstHit);
while let Some(wave) = search.step(&mut fermat).await? { /* progress, cancel, yield to the page */ }
let outcome = search.finish();
```

The algorithm is passed in (`&mut impl FermatTest`). The pieces of today's `run_waves`, separated:

- **`Groups`**: candidates plus group boundaries (data only), built from a sizes file by `formats` or by hand. The public API says
  "group"; "k" stays in the CLI's help text.
- **`RetireRule`**: after each wave, per group still alive, retire or keep. `FirstHit` retires a group when any verdict in the
  wave is `ProbablePrime` or `Retest` (a Gerbicz alarm retires the group so the candidate is retested, as today), or when the group
  has been tested to its end. Whether this is a trait or a plain function is open (section 9).
- **`WavePlan`**: which positions to test next. A wave is a range of positions applied to every group still alive. Today's plan is
  the stateless boundaries from `schedule(chunk, growth, s_max)` and is the default, so output stays byte-identical with the
  CUDA driver's. The plan is a small trait (`next_end(state) -> position`) so an adaptive plan that targets a batch size can be
  added without touching the driver.
- **Marker byte**: encoding skipped positions (3 to 255) belongs to the output side (`formats`), not the driver.

`Search::first_hit(groups, plan)` is a preset that builds the above; the name follows the rule, not the domain, because a hit is
only a *probable* prime.

## 6. Orchestration (gap hunting and similar)

The library owns the search and the per-device tests. An application owns the pipeline around them:

```
CandidateSource (sieve) -> Groups -> Search::step over FermatTest -> Outcome -> Sink (gap logic, persistence)
```

The sieve runs ahead of the GPU on the CPU. Multi-GPU is one `FermatTest` per device, each pulling batches from the
application's queue. `CandidateSource` and `Sink` are small traits in the library so the integration suite can show a worked
pipeline; the sieve, gap arithmetic, queue and persistence live in the application (`gapsieve`). Whether the library should also
ship a generic `Pipeline` is not decided; the default is no.

## 7. Performance rules (what must not be lost)

The split must not cost throughput or control. Each is a requirement with a test.

| Control | Requirement |
|---|---|
| Config override | `KernelConfig` (TPI, workgroup, squaring, scheme) is a public input to every constructor; the tuner is a default |
| Batch size | capacity is a parameter; `WavePlan` is pluggable; the tuner times candidates on the engine's batch size |
| Overlap | the runner splits *submit* from *collect*, so several dispatches can be in flight. **Probably new: as far as I have read, `fermat_async` awaits each dispatch to completion; to be confirmed against `runner.rs` before the plan** |
| Host prep | accept packed limb words as input, not only `BigUint`, so a pipeline that already holds limbs skips the conversion (1% to 2.7% of wall today) |
| Buffers | the runner keeps its buffers across calls |
| Dispatch | no virtual calls in the hot path (enum dispatch; concrete types inside) |
| Measurement | `RunStats` per dispatch stays public |
| Multi-GPU | no process-wide mutable state; one value per device |

## 8. Migration, in order

Each step is its own PR, keeps `cargo test --workspace` green, and keeps every output byte identical.

1. Tuner to core: async `candidates`, `measure`, `self_test`, `autotune`; a `Cache` trait; the cache value format. CLI keeps a file cache.
2. `formats` to core as `&[u8]` parsing and `Write` output; the CLI does the I/O.
3. Engine to core with `Observer`; fault injection behind a feature.
4. `bigmont-oracle` and `bigmont-cpu` become modules (the old crates re-export for one release); CPU single-thread path on wasm32.
5. `Search` split (`Groups`, `WavePlan`, `RetireRule`) with the step API; `schedule` and `run_waves` become thin wrappers.
6. Public surface: `prelude`, named algorithms and factories, tiers documented; submit/collect split in the runner.
7. Shells thin: CLI and wasm hold no logic; the integration suite runs the real binary and the web pages.

Every step touching `bigmont-core` also needs the wasm32 compile check and, per AGENTS.md, the browser pages run in a WebGPU
browser.

## 9. Open questions

- Crate name: keep `bigmont-core` (it is now the whole library) or rename to `bigmont`.
- `RetireRule` as a trait now, or a plain `FirstHit` function until a second rule exists. Default: the function.
- `AnyFermat` enum against boxed futures: the spec says enum; confirm when writing the checked-mode option on `GpuFermat`.
- Whether `reference` and `cpu` get cargo features so a browser build can leave `cpu` out. Default: always compiled; the
  wasm32 single-thread path makes it harmless.
- A generic `Pipeline` in the library (section 6). Default: no.
- Sequencing against the open performance work (the symmetric squaring branch and the tuner fix touch `tune` and `kernel`): step 1
  should start from `main` after that PR merges.
