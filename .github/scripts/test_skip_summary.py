#!/usr/bin/env python3
"""Tests for skip_summary.py. Run: python3 -m unittest discover -s .github/scripts"""

import contextlib
import io
import os
import re
import tempfile
import unittest

from skip_summary import Skip, forbidden, kind, main, parse, summary

NO_ADAPTER = "no adapter: no adapter matches AdapterSelector"
LOG = (
    f"devloss\ta_lost_device\t{NO_ADAPTER}\n"
    f"devloss\ta_poll_timeout\t{NO_ADAPTER}\n"
    "engine\tthe_32_bit_scheme\tthe adapter lacks SHADER_INT64\n"
    "engine\tretarget\ttpi 16: subgroup size does not fit\n"
)


class Parse(unittest.TestCase):
    def test_reads_binary_test_and_reason(self):
        self.assertEqual(
            parse("cli\tsome_test\tno adapter: x\n"), [Skip("cli", "some_test", "no adapter: x")]
        )

    def test_a_reason_may_contain_anything_after_the_second_tab(self):
        self.assertEqual(parse("a\tb\tc\td\n")[0].reason, "c\td")

    def test_ignores_malformed_and_empty_lines(self):
        self.assertEqual(parse("\nno tabs here\nonly\tone\na\tb\t \n"), [])

    def test_empty_log(self):
        self.assertEqual(parse(""), [])


class Summary(unittest.TestCase):
    def test_no_skips_says_everything_ran(self):
        text = summary([], "ubuntu-latest")
        self.assertIn("ubuntu-latest", text)
        self.assertIn("every GPU-dependent test ran", text)
        self.assertNotIn("not tested", text)

    def test_counts_reasons_and_lists_tests_once(self):
        text = summary(parse(LOG), "ubuntu-latest")
        self.assertIn("4 skip(s) in 4 test(s)", text)
        self.assertIn(f"| 2 | {NO_ADAPTER} |", text)
        self.assertIn("devloss::a_lost_device", text)
        self.assertIn("not tested", text)

    def test_a_test_that_skips_twice_is_listed_once_but_counted_twice(self):
        text = summary(parse("a\tt\tr1\na\tt\tr2\n"), "x")
        self.assertIn("2 skip(s) in 1 test(s)", text)
        self.assertEqual(text.count("a::t"), 1)

    def test_a_reason_cannot_break_the_table(self):
        text = summary([Skip("a", "t", "x | y `z`")], "x")
        self.assertIn("| 1 | x \\| y 'z' |", text)


class Kind(unittest.TestCase):
    def test_strips_the_kernel_configuration_a_test_label_and_the_selector(self):
        sel = "AdapterSelector { name: Some(\"x\"), backend: None, index: None }"
        for reason in [
            f"no adapter: no adapter matches {sel}",
            f"gec: no adapter: no adapter matches {sel}",
            f"KernelConfig {{ tpi: 8, wg: 64, scheme: Limb13 }}: no adapter: no adapter matches {sel}",
        ]:
            self.assertEqual(kind(reason), "no adapter: no adapter matches the selector")

    def test_keeps_distinct_causes_apart(self):
        self.assertNotEqual(kind("tpi 8: subgroup size does not fit"), kind("tpi 16: subgroup size does not fit"))
        self.assertEqual(kind("the adapter lacks SHADER_INT64"), "the adapter lacks SHADER_INT64")


class Forbidden(unittest.TestCase):
    PATTERN = "no (int64 )?adapter|lacks subgroup|SUBGROUP"

    def test_matches_a_missing_adapter_or_subgroups(self):
        skips = parse(
            LOG
            + "core\tx\tadapter lacks wgpu::Features::SUBGROUP\n"
            + "cli\ty\tno int64 adapter (found: None)\n"
        )
        self.assertEqual(
            [s.test for s in forbidden(skips, self.PATTERN)],
            ["a_lost_device", "a_poll_timeout", "x", "y"],
        )

    def test_a_subgroup_size_that_does_not_fit_is_allowed(self):
        skips = parse("engine\tretarget\ttpi 16: subgroup size does not fit\n")
        self.assertEqual(forbidden(skips, self.PATTERN), [])


HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))

# Every distinct way a GPU test skips, as (static part of the reason as written in the Rust
# sources, a realistic reason as logged, whether the software-GPU job must fail on it).
# lavapipe has an adapter, subgroups and SHADER_INT64, so those missing means the job no longer
# covers the GPU path; a subgroup size that does not fit a tpi is expected there (size 8).
SKIP_REASONS = [
    ("no adapter: ", "no adapter: no adapter matches AdapterSelector { name: None }", True),
    (
        "the adapter lacks subgroup support, so no tuned kernel can run",
        "the adapter lacks subgroup support, so no tuned kernel can run",
        True,
    ),
    ("no int64 adapter (found: ", "no int64 adapter (found: None)", True),
    ("the adapter lacks SHADER_INT64", "the adapter lacks SHADER_INT64", True),
    (
        "the adapter lacks SHADER_INT64 or subgroups",
        "the adapter lacks SHADER_INT64 or subgroups",
        True,
    ),
    (
        "adapter lacks wgpu::Features::SUBGROUP",
        "KernelConfig { tpi: 8 }: adapter lacks wgpu::Features::SUBGROUP",
        True,
    ),
    (
        "is not a multiple of tpi",
        "KernelConfig { tpi: 16 }: subgroup size 8..8 is not a multiple of tpi 16",
        False,
    ),
    ("subgroup size does not fit", "tpi 16: subgroup size does not fit", False),
]


def rust_sources():
    text = []
    for dirpath, dirnames, filenames in os.walk(os.path.join(REPO, "crates")):
        dirnames[:] = [d for d in dirnames if d != "target"]
        for name in filenames:
            if name.endswith(".rs"):
                with open(os.path.join(dirpath, name), encoding="utf-8") as f:
                    text.append(f.read())
    return "\n".join(text)


def ci_fail_on():
    """The --fail-on pattern the gpu-software job passes, read from the workflow itself."""
    with open(os.path.join(REPO, ".github", "workflows", "ci.yml"), encoding="utf-8") as f:
        found = re.findall(r"--fail-on\s+'([^']+)'", f.read())
    return found


class CiPatternMatchesTheTests(unittest.TestCase):
    """The workflow's --fail-on regex depends on the wording of skip reasons in the Rust tests.
    Pin both ends: the wording must still be in the sources, and the regex must classify it as
    intended, so rewording a reason cannot quietly stop the guard from matching."""

    def test_the_workflow_passes_one_fail_on_pattern(self):
        self.assertEqual(len(ci_fail_on()), 1)

    def test_each_reason_is_still_worded_that_way_in_the_sources(self):
        sources = rust_sources()
        for static, _, _ in SKIP_REASONS:
            with self.subTest(reason=static):
                self.assertIn(static, sources)

    def test_the_pattern_matches_exactly_the_reasons_that_should_fail_the_job(self):
        (pattern,) = ci_fail_on()
        for _, logged, must_fail in SKIP_REASONS:
            with self.subTest(reason=logged):
                skips = parse(f"bin\ttest\t{logged}\n")
                self.assertEqual(bool(forbidden(skips, pattern)), must_fail)


class Main(unittest.TestCase):
    def run_main(self, log_text, *extra):
        with tempfile.TemporaryDirectory() as d:
            log = os.path.join(d, "skips.log")
            if log_text is not None:
                with open(log, "w", encoding="utf-8") as f:
                    f.write(log_text)
            summary_file = os.path.join(d, "summary.md")
            out = io.StringIO()
            old = os.environ.get("GITHUB_STEP_SUMMARY")
            os.environ["GITHUB_STEP_SUMMARY"] = summary_file
            try:
                with contextlib.redirect_stdout(out):
                    code = main([log, "job", *extra])
            finally:
                if old is None:
                    del os.environ["GITHUB_STEP_SUMMARY"]
                else:
                    os.environ["GITHUB_STEP_SUMMARY"] = old
            with open(summary_file, encoding="utf-8") as f:
                return code, out.getvalue(), f.read()

    def test_a_missing_log_means_nothing_skipped(self):
        code, out, md = self.run_main(None)
        self.assertEqual(code, 0)
        self.assertNotIn("::warning", out)
        self.assertIn("every GPU-dependent test ran", md)

    def test_skips_warn_but_pass_by_default(self):
        code, out, md = self.run_main(LOG)
        self.assertEqual(code, 0)
        self.assertIn("::warning title=GPU tests skipped (job)", out)
        self.assertIn("4 skip(s)", md)

    def test_fail_on_fails_when_a_reason_matches(self):
        code, out, _ = self.run_main(LOG, "--fail-on", "no adapter")
        self.assertEqual(code, 1)
        self.assertIn("::error", out)
        self.assertIn("devloss::a_lost_device", out)

    def test_fail_on_passes_when_no_reason_matches(self):
        code, out, _ = self.run_main(
            "engine\tretarget\ttpi 16: subgroup size does not fit\n", "--fail-on", "no adapter"
        )
        self.assertEqual(code, 0)
        self.assertNotIn("::error", out)


if __name__ == "__main__":
    unittest.main()
