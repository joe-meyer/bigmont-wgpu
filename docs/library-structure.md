# meganum: one library for algorithms on very large numbers (draft spec, 2026-11-27)

Status: **draft for review, nothing implemented.** Decisions below were made in conversation; section 15 lists what is still open.

## 1. Goal and scope

**Long-term goal.** A general Rust library for algorithms on very large numbers, from a few hundred digits to megaprime sizes (a
million digits and more), on GPUs and CPUs, natively and in a browser. Today it does batched base-2 Fermat testing at about 2,000
bits; the roadmap (section 13) grows it towards strong primality tests, proofs, NTT arithmetic, single-number megaprime runs and
factoring.

**This spec's scope is phase 1: consolidation.** All of today's logic moves into one library, `meganum` (today's `bigmont-core`).
The command-line binary becomes a thin shell over it; today's wasm shim and web pages become the browser integration suite. Phase 1
builds nothing new beyond the work named in sections 8, 9 and 11; it gives the later phases their structure.

**One library.** Later phases add modules to the same crate, never new user-facing crates. Optional parts (an NTT arithmetic,
megaprime checkpointing, serde support, anything with heavy dependencies) sit behind cargo features, so a small or browser build can
leave them out. Features gate code; they do not split the public structure. If build times ever hurt, the library may split into
private helper crates that `meganum` re-exports; users would not see it.

## 2. Principles

- **Correct and deterministic.** Every result is checked against an independent reference, and the same inputs give the same
  results on every run and every executor.
- **Designed, not inherited.** We draw on the good parts of established libraries and formats (GMP, FLINT, PARI, NumPy's `.npy`,
  Arrow, PNG chunks), but nothing is kept for compatibility with an upstream consumer. Byte layouts, the CLI and defaults may change;
  consumers such as Beacon adapt to the library.
- **Layers are compile-time structure, never run-time cost** (section 8).
- **The convenience layer uses only public pieces.** The tuned factories and `Hybrid` are built entirely from public parts, so
  anything they do, an application can do.

## 3. Name

`meganum` says "very large numbers" without claiming primes or a method. "bigmont" (big-integer Montgomery) names one arithmetic
inside the library, and "big" suggests a general bignum library (the space of `num-bigint`, `rug`, `crypto-bigint`), which this is
not: it is algorithms on very large numbers. No crate named exactly `meganum` was found in a web search; that is not proof, so
crates.io and GitHub must be checked before the rename. Nothing has been released yet, so the rename is cheapest now.

## 4. Layout

Three areas, named by what their modules are about. Paths are flat (`meganum::primality::Fermat`); the areas group the folders and
the documentation, and the prelude re-exports the common items.

```
meganum/src/
  math
    arith/          number representations and multiplication: montgomery/ (today's), ntt/ (later, feature-gated)
    ntheory/        number-theoretic functions built on arith: modexp (with the Gerbicz-checked mode), gcd, inverse
    primality/      the primality problem: its outcome type, its trait, Fermat (today), Miller-Rabin, BPSW, Proth, ECPP (later)
    factoring/      (later) the factoring problem: Pollard p-1, ECM
  runtime
    executor/       where work runs: cpu/ (threads; single-threaded on wasm32), gpu/ (the wgpu device, adapter selection,
                    health and device loss, buffers, dispatch)
    tune/           which implementation to use: candidates, measure, autotune, the Cache trait
    validate/       checking an executor against known answers: self_test (used by the tuner), canary (run by applications)
    policy/         decisions over executors: Hybrid (GPU first, CPU re-check, fallback on device loss)
    schedule/       the batched search driver: Search (step API), WavePlan, the RetireRule trait
  io
    batch/          Batch (numbers and optional groups) and Results, as bytes (section 11)
tests/
  support/          dev-only crate: exact BigUint ground truth, random generators, the helpers today's bigmont-oracle offers
  cli/              runs the real binary
  browser/          proves the library runs in a browser: pages plus an unpublished wasm-bindgen harness
shells/
  cli/              the binary: arguments, files, locks, printing
```

Dependencies point one way: problems over `ntheory` over `arith`; problems and `ntheory` run on `executor`; `tune`, `validate`,
`policy` and `schedule` sit over the problems; `io` depends on nothing but the types it encodes. Nothing in the library knows a
shell or a test exists.

**Where today's crates go.** `bigmont-core` becomes `meganum`. `bigmont-cpu`'s arithmetic joins `arith/montgomery` and its ladder
joins `primality` (through `ntheory::modexp`). `bigmont-oracle` splits: the production helpers (packing numbers into limbs, limb
counts, the `-n^-1` inverse, the Gerbicz block constants) are arithmetic and join `arith/montgomery`; the ground truth moves to
`tests/support`. The runtime still needs one trusted answer, for the tuner's self-test and the canary: `validate/` computes it with
`num-bigint`'s own `modpow`, which shares no code with ours. `bigmont-cli`'s logic joins the library; what is left is the shell.
`bigmont-wasm` and `web/` become `tests/browser`.

## 5. Levels: arithmetic, number-theoretic functions, problems

These are all algorithms; they differ by what they are about, in the vocabulary of the field (Brent and Zimmermann's *Modern
Computer Arithmetic*, Crandall and Pomerance's *Prime Numbers*, GMP's "number-theoretic functions"):

| Level | About | Examples |
|---|---|---|
| `arith` | representing numbers and multiplying them | Montgomery form and multiplication; NTT multiplication |
| `ntheory` | reusable procedures that know nothing about any problem | modular exponentiation, gcd, modular inverse, elliptic-curve point operations |
| a problem (`primality`, `factoring`) | a question, its answer type and the methods that solve it | Fermat, Miller-Rabin, BPSW, Proth; Pollard p-1, ECM |

**Sharing happens at `ntheory`.** Fermat, Miller-Rabin, Proth and Pollard p-1 are all "exponentiate, then check something", so they
all use `ntheory::modexp`. Anything that knows nothing about the question goes down to `ntheory`; anything that interprets the result
stays with its problem.

**The Gerbicz-Li check belongs to `modexp`.** It verifies an exponentiation ladder (running products at every 40-iteration block
boundary and a doubled recomputation of the tail), so it is an option of `modexp`, and every problem built on it gets the check.

**Arithmetic is chosen by size,** as GMP chooses multiplication algorithms by measured crossover thresholds: schoolbook, Karatsuba or
Toom, then NTT or FFT. Here `tune` makes that choice per device and size, alongside kernel shapes. On the CPU arithmetic is a trait
(`ModArith`: enter and leave the representation, square, multiply, double, compare), sealed so only `arith` implements it; today's
`MontOps` trait is its seed. On the GPU each arithmetic is a generator of WGSL building blocks (section 8). Phase 1 moves today's
arithmetic under `arith/montgomery` and names the seam; extracting the trait is phase 2.

## 6. Problems: questions and their answers

Each problem module defines its question, its outcome type and a trait its methods implement. Outcomes are not shared across
problems, because the questions differ.

**Primality.** Probable-prime tests (Fermat, Miller-Rabin, Lucas, BPSW) answer "is n a probable prime?":

```
enum Verdict { Composite, ProbablePrime, Retest }   // #[non_exhaustive]
trait PrimalityTest { async fn verdicts(&mut self, ns: &[BigUint]) -> Result<Vec<Verdict>, Error>; }
```

`Composite` is certain (a failed Fermat or Miller-Rabin test is a proof), `ProbablePrime` is not proven, `Retest` is a Gerbicz alarm.
Differential tests compare `passes()`, not full equality. Proving methods answer a different question: Proth's test and LLR prove
primality for `k 2^n +- 1`, and ECPP proves it in general. They return a proof outcome (`Prime` with a certificate where the method
gives one, `Composite`, or `Unproven`), added with those methods.

**Factoring** (later) answers "what are the factors of n?" with its own outcome (factors found, not found within the bound).

| Method | Problem | Status |
|---|---|---|
| `Fermat` | primality (probable) | exists |
| `MillerRabin`, `Bpsw` | primality (probable, strong) | later |
| `Proth`, `Llr` | primality (proof, special forms) | later |
| `Ecpp` | primality (proof, general) | later |
| `PollardPm1`, `Ecm` | factoring | later |

The structure also takes executors and arithmetics we do not plan now. A quantum executor, for example, would be an `executor/`
(remote job submission fits the submit and collect split), its arithmetic a reversible-circuit generator in `arith/`, and Shor's
algorithm a method of `factoring`.

## 7. Executors, factories and policies

**An executor says where work runs:** `Executor::Cpu { threads }` or `Executor::Gpu(KernelConfig)`. It is a value, which is what the
tuner chooses. `executor/gpu` owns the device; arithmetics only generate kernels and run them on the executor they are given.

- **Concrete factory:** `Fermat::on(Executor::Cpu { threads: 0 })`, `Fermat::on(Executor::Gpu(config))`.
- **Tuned factory:** `Fermat::tuned(&gpu, bits, &cache)` picks the executor, arithmetic and kernel shape for this adapter and size.
  The CLI and the browser suite call this.

**Policies decide between executors.** `Hybrid` is today's `Engine`: use the GPU when the batch is big enough, re-check probable
primes and disagreements on the CPU, fall back to the CPU on device loss. It holds two instances of a method and decides between
them, so it works for any method of any problem whose answers can be re-checked.

**Decorator or mode?** A feature is a decorator (a wrapper over a problem's trait) only if it can be written knowing nothing but the
inputs and the returned answers: `Hybrid`, device-loss retry and verification sampling qualify. A feature that needs the internals of
a computation is a mode of the thing it inspects: the Gerbicz check is a mode of `modexp`.

`async fn` in traits is not object-safe, so "whichever method or executor" is an enum per problem (`AnyPrimalityTest`), not a boxed
trait object. The set is closed, the enum is zero-cost, and it works on wasm.

## 8. Layering and performance rules

The layers are compile-time structure. They must never become run-time calls per arithmetic operation.

- **GPU:** layers are code generators. `arith` emits WGSL fragments, `ntheory::modexp` assembles them into a ladder, and the problem
  adds its check. The result is one fused kernel per batch. A host-side per-operation API (`gpu.square(x)` in a loop, one dispatch
  per squaring) must not exist.
- **CPU:** generic traits monomorphise; `modexp<A: ModArith>` compiles to a specialised loop with the arithmetic inlined. No `dyn`
  in inner loops.
- **The rule:** the executor boundary is crossed once per batch or ladder, never per arithmetic operation.

**Enforcement** (the access-control layer):

- **Visibility:** GPU arithmetic exposes only generator building blocks to `ntheory` and problems; there is no callable per-operation
  API to misuse.
- **Sealed traits:** problems use `ModArith` but cannot implement it; implementations stay in `arith`.
- **An architecture test:** asserts the dependency direction of section 4 (nothing points upward), in the way ArchUnit does for Java.
- **Performance guards:** a benchmark example per method records throughput per device; a drop beyond a set margin is reported.

**Controls that must stay open:**

| Control | Requirement |
|---|---|
| Configuration | `KernelConfig` (TPI, workgroup, squaring structure, limb scheme) is a public input to `Executor::Gpu`; the tuner is a default |
| Batch size | capacity is a parameter; `WavePlan` is pluggable; the tuner times candidates on the batch size the runtime uses |
| Overlap | the GPU executor splits *submit* from *collect*, so several dispatches can be in flight. **Probably new: as far as I have read, `fermat_async` awaits each dispatch to completion; to be confirmed against `runner.rs`** |
| Phases | preparing a batch (conversion, upload) is separate from running it, so an application can hold a per-device lock only around the GPU phase while sibling processes prepare |
| Device-resident waves | a search uploads its numbers once and each wave selects positions by index. **New: today each wave's numbers are converted and uploaded again** |
| Host preparation | batches accept numbers already in limb layout (section 11), so loading skips the conversion |
| Buffers | the GPU executor keeps its buffers across calls |
| Measurement | per-dispatch statistics (`RunStats`) stay public |
| Multi-GPU | no process-wide mutable state; one value per device |

## 9. Seams

| Seam | Today | After |
|---|---|---|
| Cache | a path inside `Options`; `std::fs` and a process-wide map in `tune` | a `Cache` trait (lookup, store); an in-memory one in the library, a file one in the CLI, localStorage in the browser suite. No process-wide mutable state |
| Files | `formats` reads paths | `io` encodes to and decodes from bytes; shells do the I/O |
| Logging | `eprintln!` in the engine and the tuner | an `Observer` trait with events (tuned, device lost, recheck, alarm) |
| Blocking | `pollster` inside the tuner | everything `async`; shells block once at the top |
| Threads | `std::thread` inside the CPU batch; traps on wasm32 (issue #48) | `executor/cpu` runs single-threaded on wasm32 and threaded elsewhere, by `cfg(target_arch)` |
| Environment | `AdapterSelector::from_env` in the library | shells read environment variables and pass values |
| Fault injection | `Options::inject_*` | a cargo feature, for tests |
| Locking | the CLI's `--lock` | stays in shells: it is process-level |

## 10. Scheduling

`schedule::Search` is a pull-based driver, not an `Iterator`: a batch of numbers from many groups per dispatch is the unit of work, so
a per-item iterator or stream would starve the GPU.

```
let mut search = Search::new(batch, WavePlan::default(), primality::FirstHit);
while let Some(wave) = search.step(&mut test).await? { /* progress, cancel, yield to the page */ }
let outcome = search.finish();
```

- **The driver** knows nothing about any problem: it asks the plan which positions come next, runs the method, and hands the answers
  to the retire rule.
- **`RetireRule`** comes from the problem: per group still alive, after each wave, retire or keep. `primality::FirstHit` retires a
  group at the first `ProbablePrime` or `Retest`, or when the group is exhausted. A factoring rule would retire at the first factor.
- **`WavePlan`** says which positions to test next across all live groups. The default is whichever plan measures best; an adaptive
  plan that fills a target batch size is the expected default, with today's fixed boundaries (chunk and growth) available.
- **Skipped positions** are a status of the results (section 11), not a byte convention.

## 11. Data: batches and results

Two encoded types, as bytes (no filesystem), shared by shells, the browser suite and applications. Ideas from NumPy's `.npy` (a small
header, then raw data in the layout the consumer wants), PNG chunks (self-describing, extensible) and Arrow (columns that load without
conversion).

- **`Batch`**: the numbers and, optionally, their group boundaries, in one value and one encoding (no sidecar to drift out of step).
  Numbers are stored as little-endian limb words in the arithmetic's layout, so loading onto a device is close to a straight copy.
- **`Results`**: one outcome per number, linked to its batch by the batch's hash, so results can never be paired with the wrong input.
  It records the problem and codebook that produced it (for example `primality/v1`). `Skipped` and `Failed` are statuses alongside the
  problem's own outcomes.
- **Header:** a readable magic (`MEGANUM`), the format version, the record kind, the number width, the count and a checksum.
- **Each problem supplies its codebook** (its outcomes to and from a code). Problems whose answers do not fit one code per number, such
  as factoring, define their own record kind in the same container.
- **Optional serde support** (a feature) for JSON and similar, for small or human-readable exchange; the binary encoding is the fast
  path.

Applications that keep databases, ledgers or queues build them on these types; the library does not.

## 12. Orchestration (gap hunting and similar)

The library owns the methods, executors, policies and scheduling. An application owns the pipeline around them:

```
candidate source (sieve) -> Batch -> Search::step over a method -> Results -> sink (gap logic, persistence)
```

The sieve runs ahead of the GPU on the CPU. Multi-GPU is one method instance per device, each pulling batches from the application's
queue. `CandidateSource` and `Sink` are small traits in the library so the integration suite can show a worked pipeline. Whether the
library should also ship a generic `Pipeline` is open; the default is no.

What Beacon (the prime-gap pipeline) needs was read from its code and is reflected above: device-resident waves, locking only the GPU
phase, many inputs per process, strong tests for its notary and gap walk, proofs for gap endpoints, and sizes beyond 32,768 bits. Its
byte formats and CLI conventions are not requirements.

## 13. Roadmap (each phase gets its own spec and plan)

1. **Consolidation** (this spec).
2. **Arithmetic trait:** extract `ModArith`, generalise the generator interface, let the tuner choose between arithmetics.
3. **Strong probable-prime tests:** `MillerRabin`, `Bpsw` on today's arithmetic.
4. **NTT arithmetic:** integer NTT first (WGSL and WebGPU have no f64); f64 FFT later where a native backend offers it.
5. **Large-number execution:** one number per device in device memory, long runs with checkpoints and resume, the Gerbicz check on
   by default.
6. **Proofs:** `Proth`, `Llr`, then `Ecpp`.
7. **Factoring:** `PollardPm1`, `Ecm`.
8. **JavaScript package:** wasm-bindgen bindings for JavaScript and TypeScript callers, with no logic of their own.

Phases 2 to 8 are ordered by dependency, not by priority.

## 14. Phase 1 migration, in order

Each step is its own PR. Every step keeps the suite green and results correct and deterministic against the reference; byte layouts
and the CLI may change, and each such change is listed in the changelog.

1. **Rename** `bigmont-core` to `meganum` (once the name is confirmed free), so later moves happen under the final names.
2. **Tuner into the library:** async `candidates`, `measure`, `autotune`; a `Cache` trait; self-test moves to `validate/`.
3. **Primality outcome:** one `Verdict` across the CPU path, the GPU path and the engine, defined in `primality/`.
4. **Reference split:** production helpers into `arith/montgomery`; ground truth into `tests/support`; `validate/` computes its
   trusted answers with `num-bigint`.
5. **Executors and methods:** `executor/cpu` (with the wasm32 single-thread path) and `executor/gpu` (the device and today's runner);
   `ntheory::modexp` with the Gerbicz mode; `primality::Fermat` built on them; the engine becomes `policy::Hybrid` with `Observer`,
   and fault injection moves behind a feature.
6. **Scheduling:** `schedule::Search` with the step API, `WavePlan`, `RetireRule` and `primality::FirstHit`; numbers uploaded once and
   waves selected by index; measure the plans and pick the default.
7. **Data:** `io::Batch` and `io::Results` with the new encoding; the CLI reads and writes them.
8. **Public surface:** the prelude, factories, documented areas; submit and collect, and the separate preparation phase, on the GPU
   executor; the architecture test and the performance guards.
9. **Shells and suites:** the CLI holds no logic; `tests/cli` runs the real binary; `tests/browser` replaces `web/` and `bigmont-wasm`,
   with kernels generated at test time instead of committed. The browser suite covers what `index.html`, `fermat.html` and
   `rust.html` cover today.

Every step touching the library also needs the wasm32 compile check and the browser suite run in a WebGPU browser (AGENTS.md).

## 15. Open questions

- **Name:** confirm `meganum` is free on crates.io and GitHub; rename the repository (today `bigmont-wgpu`) and the binary too?
- **`RetireRule`:** a trait from the start (section 10 assumes so, since primality and factoring need different rules) or a function
  until factoring exists.
- **Outcome enums:** `AnyPrimalityTest` as an enum rather than boxed futures; confirm when building the Gerbicz mode on the GPU.
- **CPU executor in browser builds:** a feature to leave it out, or always compiled (the wasm32 single-thread path makes it
  harmless). Default: always compiled.
- **A generic `Pipeline`** in the library (section 12). Default: no.
- **Batch encoding details:** limb width per arithmetic in one container or one container per width; the hash function; whether
  results store only outcomes or also timing.
- **Sequencing:** the open symmetric-squaring and tuner work touches `tune` and the kernels; step 1 starts from `main` after that
  PR merges.
