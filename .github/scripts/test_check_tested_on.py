#!/usr/bin/env python3
"""Tests for check_tested_on.py. Run: python3 -m unittest discover -s .github/scripts"""

import os
import unittest

from check_tested_on import validate

ROWS = ["NVIDIA", "AMD", "Intel", "Apple", "Software (lavapipe, WARP)", "Browser (WebGPU)"]
PLACEHOLDER = "<GPU model, driver, OS / backend>"
HERE = os.path.dirname(os.path.abspath(__file__))


def table(cells: dict[str, tuple[str, str]]) -> str:
    """The older table form; `cells` maps a vendor to (detail, result)."""
    lines = ["## Tested on", "", "| Vendor | GPU | Result |", "|---|---|---|"]
    for row in ROWS:
        detail, result = cells.get(row, ("", "not tested"))
        lines.append(f"| {row} | {detail} | {result} |")
    return "\n".join(lines) + "\n"


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
        if row.startswith("Browser"):
            lines.append(f"- [{'x' if pick == 'n/a' else ' '}] n/a")
        lines.append("")
    return "\n".join(lines)


AMD_ROW = {"AMD": ("RX 6700 XT, driver 1, Windows, Vulkan", "tested")}
AMD = {"AMD": "tested: RX 6700 XT, driver 1, Windows, Vulkan"}


def template() -> str:
    with open(os.path.join(HERE, "..", "pull_request_template.md"), encoding="utf-8") as f:
        return f.read()


class CheckTestedOn(unittest.TestCase):
    def test_the_older_table_form_still_passes(self):
        self.assertEqual(validate(table(AMD_ROW)), [])

    def test_ticked_checklists_pass(self):
        self.assertEqual(validate(blocks({**AMD, "Browser (WebGPU)": "n/a"})), [])

    def test_a_vendor_with_nothing_ticked_is_an_error(self):
        errors = validate(blocks({**AMD, "NVIDIA": "", "Intel": ""}))
        self.assertEqual(len(errors), 2)
        self.assertTrue(all("tick exactly one" in e and "0 ticked" in e for e in errors), errors)

    def test_two_ticked_is_an_error(self):
        text = blocks(AMD).replace(
            f"### NVIDIA\n- [ ] tested: {PLACEHOLDER}\n- [x] not tested",
            "### NVIDIA\n- [x] tested: some GPU\n- [x] not tested",
        )
        errors = validate(text)
        self.assertEqual(len(errors), 1)
        self.assertIn("NVIDIA", errors[0])
        self.assertIn("2 ticked", errors[0])

    def test_tested_needs_details_not_the_placeholder(self):
        errors = validate(blocks({"AMD": f"tested: {PLACEHOLDER}"}))
        self.assertEqual(len(errors), 1)
        self.assertIn("say which GPU", errors[0])

    def test_at_least_one_vendor_must_be_tested(self):
        errors = validate(blocks({}))
        self.assertEqual(len(errors), 1)
        self.assertIn("Nothing is marked as tested", errors[0])

    def test_a_missing_vendor_is_an_error(self):
        errors = validate(blocks(AMD).replace("### Intel", "### Something else"))
        self.assertEqual(len(errors), 1)
        self.assertIn("Intel", errors[0])

    def test_the_escape_hatch_and_a_missing_section(self):
        self.assertEqual(validate("No hardware impact: a doc comment change only"), [])
        self.assertIn('no "## Tested on"', validate("nothing here")[0])

    def test_template_comments_do_not_count(self):
        text = blocks(AMD).replace("## Tested on", "## Tested on\n<!-- ### NVIDIA\n- [x] tested: x -->")
        self.assertEqual(validate(text), [])

    def test_the_unfilled_pull_request_template_is_rejected_for_every_vendor(self):
        errors = validate(template())
        self.assertEqual(len(errors), 6, errors)
        self.assertTrue(all("tick exactly one" in e for e in errors), errors)

    def test_the_template_filled_in_by_ticking_passes(self):
        text = template().replace("- [ ] not tested", "- [x] not tested")  # "not tested" everywhere
        text = text.replace(
            f"### AMD\n- [ ] tested: {PLACEHOLDER}\n- [x] not tested",
            "### AMD\n- [x] tested: RX 6700 XT, driver 1, Windows, Vulkan\n- [ ] not tested",
        )
        self.assertEqual(validate(text), [])


if __name__ == "__main__":
    unittest.main()
