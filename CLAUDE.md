# CLAUDE.md

Guidance for AI coding assistants (Claude Code and similar) working in this repo.
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
- **Don't change the limb scheme, TPI, or autotune heuristics** without
  benchmark evidence from at least two GPU vendors, recorded in the PR.
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
  all four crates inherit it.
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
