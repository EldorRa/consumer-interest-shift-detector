"""Assert that the committed script export still matches the notebook.

The notebook is the artefact that was executed; notebooks/consumer_interest_pipeline.py
exists only so that code changes produce a readable diff. If the two drift
apart the diff stops being trustworthy, so CI compares them.

Only code lines are compared. The jupytext header carries a tool version and
markdown wrapping is cosmetic, so neither should fail a build.

Usage: python scripts/check_export.py <committed.py> <regenerated.py>
"""

import sys
from pathlib import Path


def code_lines(path: Path) -> list[str]:
    lines, keep = [], False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# %%"):
            keep = "[markdown]" not in line
            continue
        if keep and line.strip():
            lines.append(line.rstrip())
    return lines


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2

    committed, regenerated = code_lines(Path(argv[1])), code_lines(Path(argv[2]))

    if committed == regenerated:
        print(f"export in step: {len(committed)} code lines")
        return 0

    print("ERROR: the committed export is stale. Run `make export` and commit.")
    for i, (a, b) in enumerate(zip(committed, regenerated)):
        if a != b:
            print(f"  first difference at code line {i + 1}")
            print(f"    committed   : {a!r}")
            print(f"    regenerated : {b!r}")
            break
    else:
        print(f"  length differs: {len(committed)} committed, {len(regenerated)} regenerated")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
