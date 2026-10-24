"""Build the Colab notebooks (.ipynb) from their percent-format sources (.py).

The .py files are the reviewable source of truth; run this after editing one:

    python notebooks/build_notebook.py            # every notebook
    python notebooks/build_notebook.py peekaboo_backbone2_colab
"""

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NOTEBOOKS = ("peekaboo_h20_colab", "peekaboo_backbone2_colab")


def cells(text: str):
    parts = re.split(r"^# %%(.*)$", text, flags=re.M)
    for marker, body in zip(parts[1::2], parts[2::2]):
        body = body.strip("\n")
        if "[markdown]" in marker:
            lines = [re.sub(r"^# ?", "", ln) for ln in body.splitlines()]
            yield {"cell_type": "markdown", "metadata": {}, "source": _src(lines)}
        elif body.strip():
            yield {"cell_type": "code", "metadata": {}, "execution_count": None,
                   "outputs": [], "source": _src(body.splitlines())}


def _src(lines):
    return [ln + "\n" for ln in lines[:-1]] + lines[-1:]


def build(name: str) -> None:
    src, dst = HERE / f"{name}.py", HERE / f"{name}.ipynb"
    nb_cells = list(cells(src.read_text()))
    for i, c in enumerate(nb_cells):
        c["id"] = f"cell-{i:02d}"
    nb = {
        "cells": nb_cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "machine_shape": "hm", "gpuType": "A100"},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    dst.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print(f"{len(nb['cells'])} cells -> {dst}", file=sys.stderr)


def main() -> None:
    for name in sys.argv[1:] or NOTEBOOKS:
        build(name)


if __name__ == "__main__":
    main()
