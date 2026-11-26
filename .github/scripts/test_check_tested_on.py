#!/usr/bin/env python3
"""Tests for check_tested_on.py. Run: python3 -m unittest discover -s .github/scripts"""

import os
import unittest

from check_tested_on import validate

ROWS = ["NVIDIA", "AMD", "Intel", "Apple", "Software (lavapipe, WARP)", "Browser (WebGPU)"]
PLACEHOLDER = "<GPU model, driver, OS / backend>"
HERE = os.path.dirname(os.path.abspath(__file__))
OTHER_SECTIONS = "## What and why\n\nwhy\n\n## Checklist\n\n- [ ] x\n\n## AI assistance\n\n- [ ] y\n"

SRC = ["crates/bigmont-cpu/src/lib.rs"]  # behaviour-affecting, does not run in a browser tooling-wise
CORE = ["crates/bigmont-core/src/runner.rs"]  # behaviour-affecting and reaches a browser
DOCS = ["README.md"]


def table(cells: dict[str, tuple[str, str]]) -> str:
    """The older table form; `cells` maps a vendor to (detail, result)."""
    lines = ["## Tested on", "", "| Vendor | GPU | Result |", "|---|---|---|"]
    for row in ROWS:
        detail, result = cells.get(row, ("", "not tested"))
        lines.append(f"| {row} | {detail} | {result} |")
    return "\n".join(lines) + "\n\n" + OTHER_SECTIONS


def blocks(choice: dict[str, str], default: str = "not tested") -> str:
    """The checklist form. `choice` maps a vendor to what to tick: "not tested", "n/a",
    "tested: details", or "" to leave every box unticked; vendors not listed get `default`."""
    lines = ["## Tested on", ""]
    for row in ROWS:
        pick = choice.get(row, default)
        lines.append(f"### {row}")
        tested = f"tested: {pick[len('tested: '):]}" if pick.startswith("tested:") else f"tested: {PLACEHOLDER}"
        lines.append(f"- [{'x' if pick.startswith('tested:') else ' '}] {tested}")
        lines.append(f"- [{'x' if pick == 'not tested' else ' '}] not tested")
        lines.append(f"- [{'x' if pick == 'n/a' else ' '}] n/a")
        lines.append("")
    return "\n".join(lines) + "\n" + OTHER_SECTIONS


AMD_ROW = {"AMD": ("RX 6700 XT, driver 1, Windows, Vulkan", "tested")}
AMD = {"AMD": "tested: RX 6700 XT, driver 1, Windows, Vulkan"}
ALL_NA = {r: "n/a" for r in ROWS}


def template() -> str:
    with open(os.path.join(HERE, "..", "pull_request_template.md"), encoding="utf-8") as f:
        return f.read()


class CheckTemplate(unittest.TestCase):
    def test_the_older_table_form_still_passes(self):
        self.assertEqual(validate(table(AMD_ROW), SRC), ([], []))

    def test_ticked_checklists_pass(self):
        self.assertEqual(validate(blocks(AMD), SRC), ([], []))

    def test_a_vendor_with_nothing_ticked_is_an_error(self):
        errors, _ = validate(blocks({**AMD, "NVIDIA": "", "Intel": ""}), SRC)
        self.assertEqual(len(errors), 2)
        self.assertTrue(all("tick exactly one" in e and "0 ticked" in e for e in errors), errors)

    def test_two_ticked_is_an_error(self):
        text = blocks(AMD).replace(
            f"### NVIDIA\n- [ ] tested: {PLACEHOLDER}\n- [x] not tested",
            "### NVIDIA\n- [x] tested: some GPU\n- [x] not tested",
        )
        errors, _ = validate(text, SRC)
        self.assertEqual(len(errors), 1)
        self.assertIn("NVIDIA", errors[0])
        self.assertIn("2 ticked", errors[0])

    def test_tested_needs_details_not_the_placeholder(self):
        errors, _ = validate(blocks({"AMD": f"tested: {PLACEHOLDER}"}), SRC)
        self.assertEqual(len(errors), 1)
        self.assertIn("say which GPU", errors[0])

    def test_a_missing_vendor_is_an_error(self):
        errors, _ = validate(blocks(AMD).replace("### Intel", "### Something else"), SRC)
        self.assertEqual(len(errors), 1)
        self.assertIn("Intel", errors[0])

    def test_template_comments_do_not_count(self):
        text = blocks(AMD).replace("## Tested on", "## Tested on\n<!-- ### NVIDIA\n- [x] tested: x -->")
        self.assertEqual(validate(text, SRC), ([], []))


class RequiredSections(unittest.TestCase):
    def test_every_template_section_must_be_kept_even_on_a_docs_only_pr(self):
        for name in ["What and why", "Tested on", "Checklist", "AI assistance"]:
            text = blocks(ALL_NA).replace(f"## {name}", f"## {name} (renamed)")
            errors, _ = validate(text, DOCS)
            self.assertTrue(any(f'"## {name}"' in e for e in errors), (name, errors))

    def test_a_free_form_description_fails_for_every_missing_section(self):
        errors, _ = validate("Did a thing.\n", DOCS)
        self.assertEqual(len(errors), 4, errors)

    def test_the_unfilled_template_is_rejected_for_every_vendor(self):
        errors, _ = validate(template(), DOCS)
        self.assertEqual(len(errors), 6, errors)
        self.assertTrue(all("tick exactly one" in e for e in errors), errors)

    def test_the_template_filled_in_by_ticking_passes(self):
        text = template().replace("- [ ] not tested", "- [x] not tested")  # "not tested" everywhere
        text = text.replace(
            f"### AMD\n- [ ] tested: {PLACEHOLDER}\n- [x] not tested",
            "### AMD\n- [x] tested: RX 6700 XT, driver 1, Windows, Vulkan\n- [ ] not tested",
        )
        self.assertEqual(validate(text, SRC), ([], []))


class NotApplicable(unittest.TestCase):
    def test_a_docs_only_pr_marks_everything_na_without_a_warning(self):
        self.assertEqual(validate(blocks(ALL_NA), DOCS), ([], []))

    def test_na_everywhere_on_a_behaviour_change_passes_with_a_maintainer_warning(self):
        errors, warnings = validate(blocks(ALL_NA), SRC)
        self.assertEqual(errors, [])
        self.assertEqual(len(warnings), 1)
        self.assertIn("scrutinize", warnings[0])
        self.assertIn("crates/bigmont-cpu/src/lib.rs", warnings[0])

    def test_nothing_tested_and_not_all_na_on_a_behaviour_change_is_an_error(self):
        errors, _ = validate(blocks({}), SRC)
        self.assertEqual(len(errors), 1)
        self.assertIn("Nothing is marked as tested", errors[0])
        self.assertIn("n/a", errors[0])

    def test_browser_na_on_code_a_browser_runs_is_flagged(self):
        errors, warnings = validate(blocks({**AMD, "Browser (WebGPU)": "n/a"}), CORE)
        self.assertEqual(errors, [])
        self.assertEqual(len(warnings), 1)
        self.assertIn("Browser is marked n/a", warnings[0])

    def test_browser_na_is_not_flagged_when_the_change_cannot_reach_one(self):
        self.assertEqual(validate(blocks({**AMD, "Browser (WebGPU)": "n/a"}), SRC), ([], []))

    def test_browser_tested_or_not_tested_is_not_flagged(self):
        for pick in ["not tested", "tested: Chrome 156, Windows 11"]:
            self.assertEqual(validate(blocks({**AMD, "Browser (WebGPU)": pick}), CORE), ([], []), pick)


if __name__ == "__main__":
    unittest.main()
