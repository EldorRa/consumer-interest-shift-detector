"""Assert that the analysis notebook is well-formed, and report its shape.

Run directly or through `make validate`. Used by the CI workflow.
"""

import sys
from pathlib import Path

import nbformat

NOTEBOOK = Path(__file__).resolve().parent.parent / "notebooks" / "consumer_interest_pipeline.ipynb"


def main() -> int:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)

    code = [c for c in notebook.cells if c.cell_type == "code"]
    markdown = [c for c in notebook.cells if c.cell_type == "markdown"]
    figures = sum(
        1
        for c in code
        for o in c.get("outputs", [])
        if "image/png" in o.get("data", {})
    )
    executed = sum(1 for c in code if c.get("outputs"))

    print(f"notebook ok      : {NOTEBOOK.name}")
    print(f"cells            : {len(notebook.cells)}")
    print(f"  code           : {len(code)} ({executed} carrying output)")
    print(f"  markdown       : {len(markdown)}")
    print(f"figures embedded : {figures}")

    if executed == 0:
        print("ERROR: no cell carries output, so the notebook evidences nothing")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
