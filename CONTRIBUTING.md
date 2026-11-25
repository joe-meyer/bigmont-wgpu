# Contributing

Thanks for helping out! `bigmont-wgpu` is alpha software, so expect churn.

## Getting started

```sh
git clone https://github.com/joe-meyer/bigmont-wgpu
cd bigmont-wgpu
cargo test --workspace --no-fail-fast -- --test-threads=1
```

A GPU with a Vulkan, DX12 or Metal driver is needed for the full suite. The GPU
tests are strict: a missing adapter or capability fails the test. Set
`BIGMONT_ALLOW_SKIP=1` to turn that into a loud skip, and `BIGMONT_ADAPTER=SUBSTR`
or `BIGMONT_BACKEND=Vulkan|Dx12|...` to pick a device. On Linux,
`mesa-vulkan-drivers` (lavapipe) provides a software Vulkan device.

## Workflow

1. Open or pick an issue first for anything non-trivial.
2. Branch from `main`; keep changes focused.
3. Before pushing: `cargo fmt --all`, `cargo clippy --workspace --all-targets --all-features -- -D warnings`, `cargo test --workspace -- --test-threads=1`.
4. Use [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `perf:`, `docs:`, `test:`, `chore:`).
5. Add a line under `## [Unreleased]` in `CHANGELOG.md` for user-visible changes.
6. Open a PR and complete the checklist.

## Performance and correctness

Kernel changes need a differential test against the CPU reference. Performance
claims need numbers: name the GPU, driver and OS, and include the command you
ran (for example `bigmont tune`). Don't include hostnames, usernames or other
personal details.

## AI-assisted contributions

Allowed and welcome, with a few rules. Read [`AI_POLICY.md`](AI_POLICY.md).
AI agents should also read [`AGENTS.md`](AGENTS.md) (`CLAUDE.md` just imports it).

## Versioning

[Semantic Versioning](https://semver.org). Pre-1.0, minor bumps may break the
API. Pre-releases look like `0.1.0-alpha.1`. Maintainers cut releases by tag.

## Licence

The project is [MIT](LICENSE) licensed. By contributing you agree your work is
licensed under the same terms.
