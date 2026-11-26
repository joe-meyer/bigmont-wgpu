#!/usr/bin/env python3
"""Tests for check_tested_on.py. Run: python3 -m unittest discover -s .github/scripts"""

import os
import unittest

from check_tested_on import validate

ROWS = ["NVIDIA", "AMD", "Intel", "Apple", "Software (lavapipe, WARP)", "Browser (WebGPU)"]


def body(cells: dict[str, tuple[str, str]], extra: str = "") -> str:
    """A PR body with a Tested on table; `cells` maps a vendor to (detail, result cell)."""
    lines = ["## Tested on", "", "| Vendor | GPU | Result |", "|---|---|---|"]
    for row in ROWS:
        detail, result = cells.get(row, ("", "not tested"))
        lines.append(f"| {row} | {detail} | {result} |")
    return "\n".join(lines) + "\n\n" + extra


AMD = {"AMD": ("RX 6700 XT, driver 1, Windows, Vulkan", "tested")}


class CheckTestedOn(unittest.TestCase):
    def test_plain_words_still_pass(self):
        self.assertEqual(validate(body(AMD)), [])

    def test_ticked_boxes_pass(self):
        cells = {r: ("", "[ ] tested  [x] not tested") for r in ROWS}
        cells["AMD"] = ("RX 6700 XT, driver 1, Windows, Vulkan", "[x] tested  [ ] not tested")
        cells["Browser (WebGPU)"] = ("", "[ ] tested  [ ] not tested  [X] n/a")
        self.assertEqual(validate(body(cells)), [])

    def test_a_row_with_nothing_ticked_is_an_error(self):
        cells = {r: ("", "[ ] tested  [ ] not tested") for r in ROWS}
        cells["AMD"] = ("RX 6700 XT, driver 1, Windows, Vulkan", "[x] tested  [ ] not tested")
        errors = validate(body(cells))
        self.assertEqual(len(errors), 5)
        self.assertTrue(all("tick exactly one" in e and "0 ticked" in e for e in errors), errors)

    def test_two_ticked_is_an_error(self):
        cells = dict(AMD)
        cells["Intel"] = ("", "[x] tested  [x] not tested")
        errors = validate(body(cells))
        self.assertEqual(len(errors), 1)
        self.assertIn("Intel", errors[0])
        self.assertIn("2 ticked", errors[0])

    def test_tested_needs_a_detail(self):
        errors = validate(body({"AMD": ("", "[x] tested  [ ] not tested")}))
        self.assertEqual(len(errors), 1)
        self.assertIn("say which GPU", errors[0])

    def test_at_least_one_row_must_be_tested(self):
        errors = validate(body({}))
        self.assertEqual(len(errors), 1)
        self.assertIn("Nothing is marked as tested", errors[0])

    def test_the_escape_hatch_and_a_missing_section(self):
        self.assertEqual(validate("No hardware impact: a doc comment change only"), [])
        self.assertIn("no \"## Tested on\"", validate("nothing here")[0])

    def test_template_comments_do_not_count(self):
        text = body(AMD).replace("## Tested on", "## Tested on\n<!-- | NVIDIA | x | [x] tested | -->")
        self.assertEqual(validate(text), [])

    def test_the_unfilled_pull_request_template_is_rejected(self):
        here = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(here, "..", "pull_request_template.md"), encoding="utf-8") as f:
            errors = validate(f.read())
        self.assertTrue(errors)
        self.assertTrue(all("tick exactly one" in e for e in errors[:6]), errors)


if __name__ == "__main__":
    unittest.main()
