# One library, thin shells: structure and public API (draft spec, 2026-11-27)

Status: **draft for review, nothing implemented.** Decisions below were made in conversation; section 9 lists what is still open.

## 1. Goal

All of the project's logic lives in one library, `bigmont-core`, usable natively and in a browser. The command-line binary, the
wasm shim and the web pages become thin shells over it. They own only what a library should not: argument parsing, files and
locks, printing, and browser storage. They double as the integration suite: they show how an application uses the library.

The structure must take further probable-prime tests (Miller-Rabin, Lucas, BPSW) without reshaping, although none is built here.

Non-goals: new algorithms, a new GPU backend, changing any result byte, or changing what `bigmont` does as a drop-in for the CUDA
driver.

## 2. Layout: organised by algorithm

```
crates/bigmont-core/src/        the library
  reference/      exact big-int arithmetic: the oracle every test is checked against (today's bigmont-oracle)
  montgomery/     shared arithmetic: limb schemes, the 64-bit CPU arithmetic (today's bigmont-cpu core),
                  KernelConfig, Squaring, the WGSL generators (private), Gpu and device handling
  tune/           kernel configuration choice, shared by every algorithm: candidates, measure, self_test, autotune, Cache
  algorithm/
    fermat/       verdict logic, the CPU ladder, the GPU kernel mode and runner, the Gerbicz-checked mode, tests
    miller_rabin/ (later) same shape
  executor/       Executor (where a batch runs) and Hybrid (a policy over two executors)
  search/         Groups, WavePlan, RetireRule, Search (step API): runs any test over groups
  formats/        G2C1 / G2V1 / G2S1 as bytes <-> types, no filesystem
  canary/         expected-verdict checks
crates/bigmont-cli/             bin only: args, files, lock, printing
crates/bigmont-wasm/            wasm-bindgen glue, browser storage
web/                            pages over bigmont-wasm
integration tests               run the real binary and the web pages
```

Dependencies point down only: `search` over `executor` over `algorithm` over `tune` over `montgomery` over `reference`. No module
knows a shell exists. `bigmont-oracle` and `bigmont-cpu` stop being crates.

`reference` and the CPU path in `montgomery` are both CPU code and both stay: one is ground truth (exact `BigUint` arithmetic), the
other the production CPU path (Montgomery limbs). The tuner's self-test and the re-check of GPU verdicts use the second; tests
check both against the first.

**Rule: the convenience layer uses only public pieces.** The tuned factories and `Hybrid` are built entirely from public parts.
There is no private back door, so anything they do, an application can do (section 7).

## 3. Algorithms, executors and policies

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

On the GPU a ladder is never run one operation at a time from the host (that would be one dispatch per squaring): each algorithm
is a kernel mode of the generator, as `Fermat` and `Mul` are today, with its host-side verdict logic. Miller-Rabin's mode adds a
comparison after every squaring. So adding an algorithm costs one CPU implementation plus one kernel mode; the arithmetic, the
tuner, the executors and the search are shared.

**An `Executor` says where a batch runs:** `Executor::Cpu { threads }` or `Executor::Gpu(KernelConfig)`. It is a value, not a type,
which is what the tuner chooses. An algorithm is constructed on an executor:

- **Concrete factory:** `Fermat::on(Executor::Cpu { threads: 0 })`, `Fermat::on(Executor::Gpu(config))`.
- **Tuned factory:** `Fermat::tuned(&gpu, bits, &cache)` picks the executor and kernel configuration for this adapter and size,
  using the tuner. The CLI and the web client call this. The tuner's choices (scheme, TPI, workgroup, squaring structure) are
  shared by every algorithm that uses the same arithmetic.

**`Hybrid` is a policy, not an executor.** It is today's `Engine`: use the GPU when the batch is big enough (`min_gpu_batch`),
re-check probable primes and disagreements on the CPU, fall back to the CPU on device loss. It holds two instances of an algorithm
and decides between them, so it works for any algorithm.

**Decorator or mode?** A feature is a decorator (a wrapper over `ProbablePrimeTest`) only if it can be written knowing nothing but
the inputs and the returned verdicts. `Hybrid`, device-loss retry and verification sampling qualify. The Gerbicz-Li check does not:
it needs the ladder's state at every 40-iteration block boundary (the running products `A` and `B`, the previous block's `y_prev`,
the Montgomery multiply kernel, and a doubled recomputation of the tail iterations), none of which leave the runner. So "checked"
is an option of the `Fermat` algorithm on both executors, and its only visible effect is the `Retest` verdict.

`async fn` in traits is not object-safe, so "whichever algorithm or executor" is an enum (`AnyTest`), not a boxed trait object.
The set is closed, the enum is zero-cost, and it works on wasm.

**Verdicts.** One `Verdict` enum for the whole library, replacing today's three overlapping sets (the engine's `Verdict`, the CPU
backend's `Checked`, and raw `u8`): `Composite` (0, certain: a failed Fermat or Miller-Rabin test is a proof), `ProbablePrime` (1,
passed, not proven), `Retest` (2, a Gerbicz alarm). It is `#[non_exhaustive]`. The reference arithmetic returns the same enum, and
differential tests compare `passes()`, not full equality. A skipped candidate is not a verdict; it carries the output marker byte
(section 5).

**Deferred: a proven `Prime`.** No test here proves primality, and at 2,029 bits that needs a different algorithm (ECPP or a
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
| Threads | `std::thread` inside the CPU batch; traps on wasm32 (issue #48) | the CPU path runs single-threaded on wasm32 and threaded elsewhere, gated by `cfg(target_arch)` as AGENTS.md requires, not by a feature |
| Environment | `AdapterSelector::from_env` in core | the shell reads env vars and passes values |
| Fault injection | `Options::inject_*` | a cargo feature, for tests |
| Locking (`--lock`) | the CLI | stays in the CLI: it is process-level |

## 5. Search

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
  has been tested to its end. Whether this is a trait or a plain function is open (section 9).
- **`WavePlan`**: which positions to test next. A wave is a range of positions applied to every group still alive. Today's plan is
  the stateless boundaries from `schedule(chunk, growth, s_max)` and is the default, so output stays byte-identical with the
  CUDA driver's. The plan is a small trait (`next_end(state) -> position`) so an adaptive plan that targets a batch size can be
  added without touching the driver.
- **Marker byte**: encoding skipped positions (3 to 255) belongs to the output side (`formats`), not the driver.

`Search::first_hit(groups, plan)` is a preset that builds the above; the name follows the rule, not the domain, because a hit is
only a *probable* prime.

## 6. Orchestration (gap hunting and similar)

The library owns the search, the tests and the executors. An application owns the pipeline around them:

```
CandidateSource (sieve) -> Groups -> Search::step over a test -> Outcome -> Sink (gap logic, persistence)
```

The sieve runs ahead of the GPU on the CPU. Multi-GPU is one test instance per device, each pulling batches from the
application's queue. `CandidateSource` and `Sink` are small traits in the library so the integration suite can show a worked
pipeline; the sieve, gap arithmetic, queue and persistence live in the application (`gapsieve`). Whether the library should also
ship a generic `Pipeline` is not decided; the default is no.

## 7. Performance rules (what must not be lost)

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

## 8. Migration, in order

Each step is its own PR, keeps `cargo test --workspace` green, and keeps every output byte identical.

1. Tuner to core: async `candidates`, `measure`, `self_test`, `autotune`; a `Cache` trait; the cache value format. CLI keeps a file cache.
2. `formats` to core as `&[u8]` parsing and `Write` output; the CLI does the I/O.
3. One `Verdict` enum across the reference, the CPU path, the runner and the engine.
4. `bigmont-oracle` becomes `reference`; `bigmont-cpu`'s arithmetic joins `montgomery` and its ladder joins `algorithm/fermat` (the
   old crates re-export for one release); the CPU path gets its single-thread form on wasm32.
5. `algorithm/fermat` gathers today's runner, Fermat kernel mode and the Gerbicz-checked mode; `Executor` replaces the scattered
   backend choices; the engine to core as `Hybrid`, with `Observer` and fault injection behind a feature.
6. `Search` split (`Groups`, `WavePlan`, `RetireRule`) with the step API; `schedule` and `run_waves` become thin wrappers.
7. Public surface: `prelude`, the `ProbablePrimeTest` trait, factories, tiers documented; submit/collect split on the GPU path.
8. Shells thin: CLI and wasm hold no logic; the integration suite runs the real binary and the web pages.

Every step touching `bigmont-core` also needs the wasm32 compile check and, per AGENTS.md, the browser pages run in a WebGPU
browser.

## 9. Open questions

- Crate name: keep `bigmont-core` (it is now the whole library) or rename to `bigmont`.
- `RetireRule` as a trait now, or a plain `FirstHit` function until a second rule exists. Default: the function.
- `AnyTest` enum against boxed futures: the spec says enum; confirm when writing the checked-mode option on the GPU executor.
- Whether the CPU path and `reference` get cargo features so a browser build can leave the CPU path out. Default: always compiled;
  the wasm32 single-thread path makes it harmless.
- A generic `Pipeline` in the library (section 6). Default: no.
- Where `Gpu` and device handling live: inside `montgomery` as drawn, or a module of their own, since a future non-Montgomery
  algorithm would still need a device.
- Sequencing against the open performance work (the symmetric squaring branch and the tuner fix touch `tune` and `kernel`): step 1
  should start from `main` after that PR merges.
