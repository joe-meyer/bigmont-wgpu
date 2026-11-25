# AI Contribution Guidelines

AI tools (Claude, Copilot, Cursor, ChatGPT, etc.) are welcome here. They're a
good fit for boilerplate, tests, docs, refactors and exploring ideas. These
rules keep quality and accountability clear.

## The short version

1. **You own it.** If you submit it, you've read it, understood it, and can
   explain and defend every line. "The AI wrote it" is not a review response.
2. **Test it.** Run `cargo fmt`, `cargo clippy`, and `cargo test` yourself.
   For kernel or arithmetic changes, show differential results against the CPU
   reference. Don't submit code you haven't run.
3. **Disclose meaningful assistance.** Add an `Assisted-by: <tool/model>`
   trailer to commits where AI wrote or substantially shaped the change, and
   tick the box in the PR template. Trivial autocomplete needs no disclosure.
4. **Keep PRs reviewable.** One focused change per PR. No large, unreviewed
   AI-generated dumps; maintainers may close them without detailed review.
5. **Mind licensing.** Don't submit output that reproduces code you know to be
   under an incompatible licence. You're certifying the contribution under the
   project's licence (see `LICENSE`, MIT).
6. **No secrets or private data** in prompts, commits, or results. Never paste
   credentials, or machine-identifying details, into tools or the repo.
7. **Don't fabricate.** Benchmarks, citations, issue references and test
   output must be real and reproducible. Verify anything an AI told you about
   an API, a GPU, or a driver.

## For maintainers and reviewers

- Review AI-assisted PRs exactly like any other: on correctness, clarity, and fit.
- Ask the author to explain non-obvious sections. Inability to do so is
  grounds to request changes or close.
- Disclosure is for transparency, not penalty. Undisclosed substantial AI use
  that surfaces later is treated as a process issue, not a ban.

## Using AI agents on this repo

Agents may open PRs on a human's behalf. The human operator is responsible for
the result. Agents should follow `CLAUDE.md`, work on branches, and never push
to `main`, tag releases, or change release workflows without explicit direction.

## Changes to this policy

Open an issue or PR. This policy will evolve as the project and tooling do.
