#!/usr/bin/env python3
"""Regenerate the wall coordinates embedded in ``docs/tikz/nozzle_geometry.tex``.

The geometry figure draws the *actual* contour the solver uses rather than a
freehand sketch, so it stays honest when the contour families change.  Run this
after editing :mod:`dgnozzle.geometry`, then recompile the figure::

    python docs/make_tikz.py
    cd docs/tikz && pdflatex nozzle_geometry.tex

Only the two coordinate lists and the throat marker are rewritten; the rest of
the TikZ source is hand-maintained and left alone.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from dgnozzle.geometry import NozzleGeometry  # noqa: E402

SCALE = 12.0
TARGET = ROOT / "docs" / "tikz" / "nozzle_geometry.tex"


def coordinate_lists(geom: NozzleGeometry, n: int = 61) -> tuple[str, str]:
    x = np.linspace(0.0, geom.length, n)
    y = np.asarray(geom.wall(x))
    up = " ".join(f"({a * SCALE:.4f},{b * SCALE:.4f})" for a, b in zip(x, y, strict=True))
    lo = " ".join(f"({a * SCALE:.4f},{-b * SCALE:.4f})" for a, b in zip(x, y, strict=True))
    return up, lo


def main() -> int:
    geom = NozzleGeometry(contour="bell", area_ratio=2.5019, throat_x=0.1388)
    up, lo = coordinate_lists(geom)
    text = TARGET.read_text()

    # the upper list appears twice (shading and wall), the lower once
    n_up = len(re.findall(r"plot coordinates \{\(0\.0000,", text))
    print(f"contour: {geom.describe()}")
    print(f"rewriting {n_up} upper and 1 lower coordinate list(s) in {TARGET.name}")

    def replace_nth(src: str, new: str, wanted_sign: str) -> str:
        pattern = re.compile(r"plot coordinates \{\(0\.0000," + wanted_sign + r"[^}]*\}")
        return pattern.sub("plot coordinates {" + new + "}", src)

    text = replace_nth(text, up, r"1")
    text = replace_nth(text, lo, r"-1")
    TARGET.write_text(text)
    print("done -- recompile with: cd docs/tikz && pdflatex nozzle_geometry.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
