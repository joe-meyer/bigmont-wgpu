#!/usr/bin/env python3
"""Check that a pull request documents which GPU vendors it was tested on.

Results differ between GPU vendors, architectures and drivers (see AGENTS.md), so
a PR that can change behaviour must fill in the "Tested on" table from the PR
template. Docs-only, CI-only and similar PRs are exempt.

Usage: check_tested_on.py BODY_FILE CHANGED_FILES_FILE
  BODY_FILE           the PR description
  CHANGED_FILES_FILE  one changed path per line

Exit status 0 = pass or not required, 1 = the section is missing or incomplete.
"""

import re
import sys

# Paths that can change behaviour on a GPU, and so need vendor coverage noted.
REQUIRED_PATTERNS = [
    r"^crates/[^/]+/src/",  # engine, kernels, autotuner, CPU backend
    r"^crates/[^/]+/Cargo\.toml$",  # dependency changes (for example wgpu)
    r"^Cargo\.(toml|lock)$",
]

VENDORS = ["NVIDIA", "AMD", "Intel", "Apple", "Software", "Browser"]
VALID_RESULTS = {"tested", "not tested", "untested", "n/a", "na"}

# The template offers one small checklist per vendor (GitHub renders "- [ ]" list items as clickable
# checkboxes, but not inside table cells):
#
#     ### NVIDIA
#     - [ ] tested: <GPU model, driver, OS / backend>
#     - [ ] not tested
#
# Tick exactly one item; "tested" needs the details after the colon. The older table form (a plain
# "tested", "not tested" or "n/a" in the Result column) is still accepted.
HEADING = re.compile(r"^###\s+(.+?)\s*$")
ITEM = re.compile(r"^\s*[-*]\s+\[([ xX]?)\]\s*(.*?)\s*$")
PLACEHOLDER = re.compile(r"<[^<>]*>")

# A reviewer-visible escape hatch for PRs touching the paths above that cannot
# change behaviour (a pure refactor, a doc comment fix, ...).
ESCAPE = re.compile(r"(?im)^\s*no hardware impact:\s*(\S.{9,})$")


def requires_section(path: str) -> bool:
    if any(re.search(p, path) for p in REQUIRED_PATTERNS):
        return True
    # Browser kernels and pages, but not the web README.
    return path.startswith("web/") and not path.endswith(".md")


def strip_comments(text: str) -> str:
    return re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)


def tested_on_section(body: str):
    match = re.search(r"(?im)^##\s+Tested on\s*$", body)
    if not match:
        return None
    rest = body[match.end():]
    nxt = re.search(r"(?m)^##\s", rest)
    return rest[: nxt.start()] if nxt else rest


def parse_rows(section: str):
    rows = []
    for line in section.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) != 3:
            continue
        if set("".join(cells)) <= set("-: "):  # the |---|---|---| separator
            continue
        if cells[0].lower() == "vendor":  # header
            continue
        rows.append(cells)
    return rows


def parse_blocks(section: str):
    """Vendor blocks of the checklist form: heading text -> [(ticked, label, detail)]."""
    blocks, current = {}, None
    for line in section.splitlines():
        heading = HEADING.match(line)
        if heading:
            current = blocks.setdefault(heading.group(1), [])
            continue
        item = ITEM.match(line)
        if item and current is not None:
            mark, text = item.groups()
            label, _, detail = text.partition(":")
            detail = PLACEHOLDER.sub("", detail).strip()
            current.append((mark.strip().lower() == "x", label.strip(), detail))
    return blocks


def block_result(items):
    """(result, detail, problem) for one vendor's checklist."""
    options = ", ".join(f'"{label}"' for _, label, _ in items)
    ticked = [(label, detail) for on, label, detail in items if on]
    if len(ticked) != 1:
        return "", "", f"tick exactly one of {options} (found {len(ticked)} ticked)"
    return ticked[0][0], ticked[0][1], None


def validate(body: str) -> list[str]:
    body = strip_comments(body)
    if ESCAPE.search(body):
        return []
    section = tested_on_section(body)
    if section is None:
        return ['The PR description has no "## Tested on" section (copy it from the PR template).']

    rows = parse_rows(section)
    errors = []
    tested_count = 0
    blocks = parse_blocks(section)
    for vendor in VENDORS:
        block = next((v for k, v in blocks.items() if k.lower().startswith(vendor.lower())), None)
        row = next((r for r in rows if r[0].lower().startswith(vendor.lower())), None)
        if block is not None:
            result, detail, problem = block_result(block)
        elif row is not None:
            _, detail, result = row
            problem = None
        else:
            errors.append(f"Missing the {vendor} entry in the Tested on section.")
            continue
        norm = result.lower()
        if problem:
            errors.append(f"{vendor}: {problem}.")
        elif norm not in VALID_RESULTS:
            errors.append(
                f'{vendor}: result must be one of "tested", "not tested" or "n/a", got "{result}".'
            )
        elif norm == "tested":
            if not detail:
                errors.append(f"{vendor}: say which GPU model, driver and OS/backend you tested on.")
            else:
                tested_count += 1
    if not errors and tested_count == 0:
        errors.append(
            "Nothing is marked as tested. Mark at least one vendor as tested, or, if this change "
            'cannot affect behaviour, add a line "No hardware impact: <reason>".'
        )
    return errors


def main() -> int:
    body_file, files_file = sys.argv[1], sys.argv[2]
    with open(body_file, encoding="utf-8") as f:
        body = f.read()
    with open(files_file, encoding="utf-8") as f:
        files = [line.strip() for line in f if line.strip()]

    needing = [p for p in files if requires_section(p)]
    if not needing:
        print("No behaviour-affecting files changed; the Tested on section is not required.")
        return 0

    print("Files that need vendor coverage noted:")
    for p in needing[:20]:
        print(f"  {p}")
    if len(needing) > 20:
        print(f"  ... and {len(needing) - 20} more")

    errors = validate(body)
    if not errors:
        print("Tested on section looks complete.")
        return 0
    for e in errors:
        print(f"::error title=Tested on::{e}")
    print(
        "\nFill in the Tested on table in the PR description (see CONTRIBUTING.md). "
        '"not tested" is an acceptable answer; leaving it blank is not.'
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
