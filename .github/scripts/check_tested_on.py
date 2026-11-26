#!/usr/bin/env python3
"""Check a pull request description against the PR template.

Every PR keeps the template's sections (see REQUIRED_SECTIONS) and answers the "Tested on"
section: one ticked box per vendor, "tested" (with the GPU model, driver and OS/backend),
"not tested" or "n/a".

Results differ between GPU vendors, architectures and drivers (see AGENTS.md), so a PR that
touches behaviour-affecting files must have at least one vendor "tested". If testing really
cannot matter for such a PR, the author marks every vendor "n/a": that passes, but it raises a
warning so a maintainer scrutinizes the claim. "n/a" for the browser on a PR that touches code
a browser runs is flagged the same way. PRs that touch no such files (docs, CI) can mark
everything "n/a" without a warning.

Usage: check_tested_on.py BODY_FILE CHANGED_FILES_FILE
  BODY_FILE           the PR description
  CHANGED_FILES_FILE  one changed path per line

Exit status 0 = pass (possibly with warnings), 1 = the description is incomplete.
"""

import os
import re
import sys

# Sections of the PR template every description keeps.
REQUIRED_SECTIONS = ["What and why", "Tested on", "Checklist", "AI assistance"]

# Paths that can change behaviour on a GPU, and so need vendor coverage noted.
REQUIRED_PATTERNS = [
    r"^crates/[^/]+/src/",  # engine, kernels, autotuner, CPU backend
    r"^crates/[^/]+/Cargo\.toml$",  # dependency changes (for example wgpu)
    r"^Cargo\.(toml|lock)$",
]

# Paths whose code, or whose dependencies, also run in a browser (WebGPU).
BROWSER_PATTERNS = [
    r"^crates/bigmont-(core|wasm|oracle)/",
    r"^Cargo\.(toml|lock)$",
]

VENDORS = ["NVIDIA", "AMD", "Intel", "Apple", "Software", "Browser"]
RESULTS = {"tested": "tested", "not tested": "not tested", "untested": "not tested", "n/a": "n/a", "na": "n/a"}

# The template offers one small checklist per vendor (GitHub renders "- [ ]" list items as clickable
# checkboxes, but not inside table cells):
#
#     ### NVIDIA
#     - [ ] tested: <GPU model, driver, OS / backend>
#     - [ ] not tested
#     - [ ] n/a
#
# Tick exactly one item; "tested" needs the details after the colon. The older table form (a plain
# "tested", "not tested" or "n/a" in the Result column) is still accepted.
HEADING = re.compile(r"^###\s+(.+?)\s*$")
ITEM = re.compile(r"^\s*[-*]\s+\[([ xX]?)\]\s*(.*?)\s*$")
PLACEHOLDER = re.compile(r"<[^<>]*>")


def requires_section(path: str) -> bool:
    if any(re.search(p, path) for p in REQUIRED_PATTERNS):
        return True
    # Browser kernels and pages, but not the web README.
    return path.startswith("web/") and not path.endswith(".md")


def reaches_browser(path: str) -> bool:
    if any(re.search(p, path) for p in BROWSER_PATTERNS):
        return True
    return path.startswith("web/") and not path.endswith(".md")


def strip_comments(text: str) -> str:
    return re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)


def has_section(body: str, name: str) -> bool:
    return re.search(rf"(?im)^##\s+{re.escape(name)}\s*$", body) is not None


def section_text(body: str, name: str):
    match = re.search(rf"(?im)^##\s+{re.escape(name)}\s*$", body)
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


def vendor_results(section: str, errors: list[str]):
    """vendor -> (normalised result, detail); problems are appended to `errors`."""
    rows, blocks = parse_rows(section), parse_blocks(section)
    out = {}
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
        norm = RESULTS.get(result.lower())
        if problem:
            errors.append(f"{vendor}: {problem}.")
        elif norm is None:
            errors.append(f'{vendor}: result must be one of "tested", "not tested" or "n/a", got "{result}".')
        elif norm == "tested" and not detail:
            errors.append(f"{vendor}: say which GPU model, driver and OS/backend you tested on.")
        else:
            out[vendor] = (norm, detail)
    return out


def validate(body: str, files: list[str]) -> tuple[list[str], list[str]]:
    """(errors, warnings) for a PR description and the paths the PR changes."""
    body = strip_comments(body)
    errors: list[str] = []
    warnings: list[str] = []

    for name in REQUIRED_SECTIONS:
        if not has_section(body, name):
            errors.append(
                f'The PR description has no "## {name}" section. Keep every section of the PR '
                "template (.github/pull_request_template.md); do not delete or rename them."
            )
    section = section_text(body, "Tested on")
    if section is None:
        return errors, warnings

    results = vendor_results(section, errors)
    if errors:
        return errors, warnings

    needing = [p for p in files if requires_section(p)]
    tested = [v for v, (r, _) in results.items() if r == "tested"]
    all_na = all(r == "n/a" for r, _ in results.values())
    if needing:
        if not tested and not all_na:
            errors.append(
                "Nothing is marked as tested. Mark at least one vendor as tested, or, if testing "
                'cannot matter for this change, mark every vendor "n/a" (a maintainer will be '
                "asked to look closer)."
            )
        elif all_na:
            shown = ", ".join(needing[:5]) + (f" and {len(needing) - 5} more" if len(needing) > 5 else "")
            warnings.append(
                f"Every vendor is marked n/a (no testing needed), but this PR changes behaviour-"
                f"affecting files ({shown}). Maintainers: scrutinize that claim."
            )
    if (
        not errors
        and not all_na
        and results.get("Browser", ("", ""))[0] == "n/a"
        and any(reaches_browser(p) for p in files)
    ):
        warnings.append(
            "Browser is marked n/a, but this PR changes code a browser runs "
            "(bigmont-core, bigmont-wasm, bigmont-oracle, web/ or a dependency). Maintainers: "
            "check that it cannot reach a browser, or ask for a browser run."
        )
    return errors, warnings


def main() -> int:
    body_file, files_file = sys.argv[1], sys.argv[2]
    with open(body_file, encoding="utf-8") as f:
        body = f.read()
    with open(files_file, encoding="utf-8") as f:
        files = [line.strip() for line in f if line.strip()]

    needing = [p for p in files if requires_section(p)]
    if needing:
        print("Files that need vendor coverage noted:")
        for p in needing[:20]:
            print(f"  {p}")
        if len(needing) > 20:
            print(f"  ... and {len(needing) - 20} more")
    else:
        print("No behaviour-affecting files changed; marking every vendor n/a is fine.")

    errors, warnings = validate(body, files)
    for w in warnings:
        print(f"::warning title=Tested on::{w}")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary and warnings:
        with open(summary, "a", encoding="utf-8") as f:
            f.write("### Maintainers: look closer\n\n" + "\n".join(f"- {w}" for w in warnings) + "\n")
    if not errors:
        print("The PR description follows the template.")
        return 0
    for e in errors:
        print(f"::error title=PR template::{e}")
    print(
        "\nFill in the PR template (see CONTRIBUTING.md). Every section stays; "
        '"not tested" and "n/a" are acceptable answers, leaving a vendor unticked is not.'
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
