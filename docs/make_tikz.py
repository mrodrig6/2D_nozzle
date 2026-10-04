#!/usr/bin/env python3
"""Regenerate the wall coordinates embedded in ``docs/tikz/nozzle_geometry.tex``.

The geometry figure draws the *actual* contour the solver uses rather than a
freehand sketch, so it stays honest when the contour families change.  Run this
after editing :mod:`src.geometry`, then recompile the figure::

    python docs/make_tikz.py
    python docs/render_figures.py

Only the two coordinate lists and the throat marker are rewritten; the rest of
the TikZ source is hand-maintained and left alone.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# the package sits at the repository root, so a clone is importable as it stands
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.geometry import NozzleGeometry  # noqa: E402

SCALE = 12.0
TARGET = ROOT / "docs" / "tikz" / "nozzle_geometry.tex"


def coordinate_lists(geom: NozzleGeometry, n: int = 61) -> tuple[str, str]:
    x = np.linspace(0.0, geom.length, n)
    y = np.asarray(geom.wall(x))
    up = " ".join(f"({a * SCALE:.4f},{b * SCALE:.4f})" for a, b in zip(x, y, strict=True))
    lo = " ".join(f"({a * SCALE:.4f},{-b * SCALE:.4f})" for a, b in zip(x, y, strict=True))
    return up, lo


#: How many coordinate lists the figure is expected to contain.
#: Two upper (the shaded half-domain and the upper wall) and one lower wall.
EXPECTED_UPPER = 2
EXPECTED_LOWER = 1


def _substitute(text: str, new_coords: str, sign: str, expected: int, what: str) -> str:
    """Replace every ``plot coordinates`` list of one sign, verifying the count.

    The count check is the important part.  The lists are matched by the sign of
    their first ``y`` value, which is the only thing distinguishing the upper
    wall from its mirror image in the source.  If a contour change moved that
    value so the pattern stopped matching, a silent no-op would leave the figure
    stale while ``git diff`` stayed clean -- so the CI job that regenerates and
    diffs this file would pass on an out-of-date figure.  Failing loudly here is
    what makes that check trustworthy.
    """
    pattern = re.compile(r"plot coordinates \{\(0\.0000," + sign + r"[0-9][^}]*\}")
    text, count = pattern.subn("plot coordinates {" + new_coords + "}", text)
    if count != expected:
        raise SystemExit(
            f"make_tikz: expected {expected} {what} coordinate list(s) in "
            f"{TARGET.name}, matched {count}. The figure source has changed "
            f"shape; update EXPECTED_{what.upper()} and the pattern together."
        )
    return text


def main() -> int:
    geom = NozzleGeometry(contour="bell", area_ratio=2.5019, throat_x=0.1388)
    upper, lower = coordinate_lists(geom)
    text = TARGET.read_text(encoding="utf-8")

    print(f"contour: {geom.describe()}")
    text = _substitute(text, upper, r"", EXPECTED_UPPER, "upper")
    text = _substitute(text, lower, r"-", EXPECTED_LOWER, "lower")
    TARGET.write_text(text, encoding="utf-8")
    print(
        f"rewrote {EXPECTED_UPPER} upper and {EXPECTED_LOWER} lower "
        f"coordinate list(s) in {TARGET.name}"
    )
    print("done -- re-render with: python docs/render_figures.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
