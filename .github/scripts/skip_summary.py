#!/usr/bin/env python3
"""Summarise the GPU-dependent tests that skipped in a CI run.

GPU tests are strict by default; CI sets BIGMONT_ALLOW_SKIP=1 because hosted runners have no
usable GPU, so a test that cannot run prints "SKIP <why>" and passes. A passing test run therefore
says nothing about how much was actually tested. The test helpers also append one tab-separated
line per skip (binary, test, reason) to the file named by BIGMONT_SKIP_LOG; this script turns that
file into a job summary and a warning annotation, so reduced coverage is visible on the PR.

With --fail-on REGEX it also fails the job when any skip reason matches. The software-GPU job
uses that to stay honest: a skip because there is no adapter or no subgroup support means the
job no longer runs the real wgpu path, and that must not pass silently.

Usage: skip_summary.py LOG TITLE [--fail-on REGEX]
  LOG       the BIGMONT_SKIP_LOG file (a missing or empty file means nothing skipped)
  TITLE     names the job in the summary, for example "ubuntu-latest"

Writes the summary to $GITHUB_STEP_SUMMARY (stdout when that is unset).
Exit status 0 = ok, 1 = a skip matched --fail-on.
"""

import argparse
import os
import re
import sys
from collections import Counter
from typing import List, NamedTuple


class Skip(NamedTuple):
    binary: str
    test: str
    reason: str


def parse(text: str) -> List[Skip]:
    """The skips in a log. A line that is not `binary<TAB>test<TAB>reason` is ignored."""
    skips = []
    for line in text.splitlines():
        parts = line.split("\t", 2)
        if len(parts) == 3 and parts[2].strip():
            skips.append(Skip(parts[0], parts[1], parts[2].strip()))
    return skips


def kind(reason: str) -> str:
    """`reason` without the parts that differ between tests, so equal causes group together: a
    leading kernel configuration or test label, and the adapter selector's debug text."""
    reason = re.sub(r"^KernelConfig \{[^}]*\}: ", "", reason)
    reason = re.sub(r"^[a-z0-9_]+: ", "", reason)
    return re.sub(r"AdapterSelector \{[^}]*\}", "the selector", reason)


def cell(text: str) -> str:
    """`text` made safe for a Markdown table cell."""
    return text.replace("|", "\\|").replace("`", "'")


def summary(skips: List[Skip], title: str) -> str:
    """The Markdown job summary."""
    out = [f"### GPU-dependent tests skipped: {title}", ""]
    if not skips:
        out.append("None: every GPU-dependent test ran.")
        return "\n".join(out) + "\n"
    tests = sorted({(s.binary, s.test) for s in skips})
    out.append(
        f"{len(skips)} skip(s) in {len(tests)} test(s). The code paths they cover "
        "were **not tested** on this runner."
    )
    out += ["", "| Skips | Reason |", "|---:|---|"]
    for reason, count in Counter(kind(s.reason) for s in skips).most_common():
        out.append(f"| {count} | {cell(reason)} |")
    out += ["", "<details><summary>Skipped tests</summary>", "", "```"]
    out += [f"{binary}::{test}" for binary, test in tests]
    out += ["```", "", "</details>"]
    return "\n".join(out) + "\n"


def forbidden(skips: List[Skip], pattern: str) -> List[Skip]:
    """The skips whose reason matches `pattern`."""
    rx = re.compile(pattern)
    return [s for s in skips if rx.search(s.reason)]


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("log")
    ap.add_argument("title")
    ap.add_argument("--fail-on", default="", help="fail when a skip reason matches this regex")
    args = ap.parse_args(argv)

    try:
        with open(args.log, encoding="utf-8", errors="replace") as f:
            skips = parse(f.read())
    except FileNotFoundError:
        skips = []

    text = summary(skips, args.title)
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(text)
    else:
        sys.stdout.write(text)

    if skips:
        print(
            f"::warning title=GPU tests skipped ({args.title})::"
            f"{len(skips)} skip(s): the GPU paths they cover were not tested on this runner."
        )
    bad = forbidden(skips, args.fail_on) if args.fail_on else []
    if bad:
        print(
            f"::error title=Unexpected GPU test skips ({args.title})::"
            f"{len(bad)} skip(s) match /{args.fail_on}/, so this job is not running the GPU path it exists for."
        )
        for s in bad:
            print(f"  {s.binary}::{s.test}: {s.reason}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
