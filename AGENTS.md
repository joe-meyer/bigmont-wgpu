# AGENTS.md

Guidance for AI coding assistants (Claude Code, Copilot, Codex, Cursor and similar) working in this repo.
Humans: see `CONTRIBUTING.md` and `AI_POLICY.md`.

## What this project is

`bigmont-wgpu` runs batched base-2 Fermat probable-prime tests on large integers
(tested up to about 2,000 bits) on any GPU supported by `wgpu` (Vulkan, DX12, Metal).
The kernels are generated WGSL (cooperative Montgomery arithmetic), there is an
autotuner, a pure-Rust CPU fallback, and a 13-bit-limb scheme that also runs under
WebGPU in a browser. Status: **alpha / experimental**.

It is a Cargo workspace (edition 2024):

| Path | What |
|---|---|
| `crates/bigmont-oracle` | CPU reference arithmetic for both limb schemes (tests, self-test) |
| `crates/bigmont-core` | the wgpu engine: kernel generators (`unrolled.rs`, `limb32.rs`), `Runner`, device handling |
| `crates/bigmont-cpu` | pure-Rust 64-bit-limb Montgomery Fermat test (CPU backend) |
| `crates/bigmont-cli` | the `bigmont` binary: file formats, autotuner and cache, engine |
| `crates/bigmont-wasm` | browser (wasm32) entry points over `bigmont-core`; empty on native targets |
| `web/` | browser test pages and generated kernels |
| `docs/superpowers/` | design, plans, measurements and notes |

## Commands

```sh
cargo fmt --all                                                      # format
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --no-fail-fast -- --test-threads=1            # unit + differential + GPU tests
cargo doc --workspace --no-deps --all-features                       # docs must build warning-free
```

Run fmt, clippy and test before proposing any change. There is no `benches/`
directory; performance work uses the `bigmont tune` command and the examples
under `crates/*/examples`.

GPU tests are **strict**: a missing adapter or capability fails the test.
`BIGMONT_ALLOW_SKIP=1` turns that into a loud skip (used on CI runners without a
GPU). `BIGMONT_ADAPTER=SUBSTR` and `BIGMONT_BACKEND=Vulkan|Dx12|...` pick the
device. Always run the suite with `--test-threads=1`.

## Ground rules

- **Correctness beats speed.** A faster kernel that returns a wrong
  composite/prime verdict is a regression. Any kernel change must be checked
  against the CPU reference (`bigmont-oracle` / `bigmont-cpu`) on the same
  inputs (differential test), including edge cases: small moduli, values near
  limb boundaries, carry-overflow paths.
- **WGSL has no add-with-carry, no extended multiply, and no 64-bit integers
  on WebGPU.** The 32-bit-limb kernels require the optional `SHADER_INT64`
  feature and are native-only; the 13-bit kernels must not need it, because they
  also target browsers. Check device features before using optional ones, and
  keep both limb schemes working.
- **Limb scheme, TPI and autotune changes are welcome, but results vary by
  vendor.** Correctness and speed can differ between GPU vendors, architectures
  and drivers (and between native backends and WebGPU), so a change that is
  right on one device may be wrong or slower on another. Don't assume your
  hardware is representative.
- **PRs that can change behaviour state what they were tested on and what they
  were not.** CI (`tested-on.yml`) requires this when a PR touches
  `crates/*/src`, Cargo manifests or non-markdown files in `web/`; docs-only and
  CI-only PRs are exempt, and a PR in those paths that cannot change behaviour
  can say `No hardware impact: <reason>` instead. List the
  GPUs/vendors, drivers and OS/backends you ran the tests (and benchmarks, if
  any) on, and say plainly which vendors (NVIDIA, AMD, Intel, Apple, software
  rasterisers, browsers) were *not* tested. This lets other contributors pick up
  the gaps. Fill in the "Tested on" section of the PR template; "untested" is an
  acceptable answer, a silent omission is not. Never claim coverage you didn't run.
- **Test in a browser whenever the change can reach one.** A change to `bigmont-core`,
  the 13-bit kernel generators (`unrolled.rs`), `bigmont-wasm`, anything in `web/`, or the
  pinned wgpu can behave differently under WebGPU, and CI's `wasm-check` only compiles for
  wasm32, it never runs anything. Before opening the PR, build and serve as described in
  `web/README.md` ("The Rust engine in the browser") and run `rust.html` (the real Rust
  runner against a BigInt oracle) in a WebGPU browser with subgroup support, plus
  `fermat.html` when kernels changed. The in-app browser pane or Claude in Chrome will do if
  your session has one. Put the browser, version and OS in the "Browser" row of "Tested on"
  and say what you ran. If you cannot run one, mark it "not tested", never leave it
  implied. Mark it "n/a" only when the change truly cannot reach a browser (for example
  the native-only 32-bit scheme, or the CLI alone).
- **`bigmont-core` must keep compiling for wasm32 (CI `wasm-check`).** Its real work is in
  `async fn`s (`Gpu::open_async`, `Runner::fermat_async`, `run_async`, ...); the blocking forms are
  native-only wrappers over them. Don't add blocking waits, threads, `Instant::now()` from `std`
  (use `web_time`) or other native-only calls outside `#[cfg(not(target_arch = "wasm32"))]`.
- **Handle device loss.** GPU calls can fail or reset mid-run; return errors,
  never panic or hang. The `devloss` tests cover this.
- **No unsafe** unless unavoidable; if added, document the invariant on the line above.
- **Never commit machine-identifying details** (hostnames, usernames, serials,
  local paths) in results or docs. Describe hardware by model only. History was
  scrubbed of these once already; don't reintroduce them.
- **Don't invent performance numbers.** Only cite benchmarks you actually ran,
  with the hardware and driver named.

## Style

- Idiomatic Rust (edition 2024); `rustfmt` defaults; no clippy warnings in code you touch.
- Public items need doc comments; examples should compile as doctests.
- Keep WGSL readable: comment the arithmetic invariant (e.g. limb ranges)
  above each non-obvious block.
- Small, focused commits using Conventional Commits (`feat:`, `fix:`, `perf:`,
  `docs:`, `test:`, `chore:`). Breaking changes: `feat!:` or a `BREAKING CHANGE:` footer.

## Versioning and releases

- SemVer. While `0.y.z`, anything may change; bump `y` for breaking changes,
  `z` for fixes. Planned line: `0.1.0-alpha.N`.
- The version lives once, in `[workspace.package]` in the root `Cargo.toml`, and
  every crate inherits it.
- Do **not** bump versions, edit `CHANGELOG.md` release sections, or create tags
  unless asked. Releases are cut by maintainers by pushing a `vX.Y.Z[-pre]` tag;
  `.github/workflows/release.yml` does the rest.
- Add user-visible changes under `## [Unreleased]` in `CHANGELOG.md`.

## Don't touch without asking

- `.github/workflows/release.yml` and release-related config
- `Cargo.toml` version / license settings
- `LICENSE`

## Commits by AI

When an AI tool materially authored a commit, add the trailer
`Assisted-by: <tool/model>` (see `AI_POLICY.md`). A human is always the author
of record and is responsible for the change.
