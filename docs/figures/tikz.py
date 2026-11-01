#!/usr/bin/env python3
"""Keep the TikZ figures in step with the solver, and build their PNGs.

The figures are *authored* in TikZ, which is the source of truth, and Markdown
cannot typeset TikZ, so the Markdown links to rasterised PNGs.  Two steps:

``coordinates``  Rewrite the wall coordinates in ``nozzle_geometry.tex`` from
                 the solver's own contour code, so the drawn nozzle is the one
                 the solver uses.  Run it after editing ``src.geometry``.
``render``       Compile each ``.tex`` in ``docs/figures/tikz/`` with pdflatex
                 and rasterise it to a PNG beside it.

Run both after a geometry change::

    python docs/figures/tikz.py coordinates
    python docs/figures/tikz.py render

``render`` needs ``pdflatex`` (TeX Live: ``texlive-latex-base``,
``texlive-pictures``, ``texlive-latex-extra`` for ``standalone.cls``) and
``pymupdf`` for the rasterisation step::

    pip install pymupdf

Both are build-time only.  Nobody reading the documentation needs either, which
is the point of committing the PNGs.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
# the package sits at the repository root, so a clone is importable as it stands
sys.path.insert(0, str(ROOT))

from src.geometry import NozzleGeometry  # noqa: E402

TIKZ = Path(__file__).resolve().parent / "tikz"

#: Rendering resolution.  The figures are vector line art at roughly 16 cm
#: wide; 200 dpi keeps the labels legible when GitHub scales them down to the
#: README column width without making the files large.
DPI = 200

SCALE = 12.0
TARGET = TIKZ / "nozzle_geometry.tex"

#: How many coordinate lists the figure is expected to contain.
#: Two upper (the shaded half-domain and the upper wall) and one lower wall.
EXPECTED_UPPER = 2
EXPECTED_LOWER = 1


def coordinate_lists(geom: NozzleGeometry, n: int = 61) -> tuple[str, str]:
    x = np.linspace(0.0, geom.length, n)
    y = np.asarray(geom.wall(x))
    up = " ".join(f"({a * SCALE:.4f},{b * SCALE:.4f})" for a, b in zip(x, y, strict=True))
    lo = " ".join(f"({a * SCALE:.4f},{-b * SCALE:.4f})" for a, b in zip(x, y, strict=True))
    return up, lo


def _substitute(text: str, new_coords: str, sign: str, expected: int, what: str) -> str:
    """Replace every ``plot coordinates`` list of one sign, verifying the count.

    The count check is the important part.  The lists are matched by the sign of
    their first ``y`` value, which is the only thing distinguishing the upper
    wall from its mirror image in the source.  If a contour change moved that
    value so the pattern stopped matching, a silent no-op would leave the figure
    stale while ``git diff`` stayed clean -- so regenerating and diffing this
    file would look clean on an out-of-date figure.  Failing loudly here is what
    makes running this script a real check rather than a reassuring no-op.
    """
    pattern = re.compile(r"plot coordinates \{\(0\.0000," + sign + r"[0-9][^}]*\}")
    text, count = pattern.subn("plot coordinates {" + new_coords + "}", text)
    if count != expected:
        raise SystemExit(
            f"tikz.py coordinates: expected {expected} {what} coordinate list(s) in "
            f"{TARGET.name}, matched {count}. The figure source has changed "
            f"shape; update EXPECTED_{what.upper()} and the pattern together."
        )
    return text


def coordinates() -> int:
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
    print("done -- now run: python docs/figures/tikz.py render")
    return 0


def _compile(source: Path, workdir: Path) -> Path:
    """Compile one standalone TikZ file, returning the PDF."""
    subprocess.run(
        [
            "pdflatex",
            "-interaction=nonstopmode",
            "-halt-on-error",
            f"-output-directory={workdir}",
            str(source),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return workdir / (source.stem + ".pdf")


def _rasterise(pdf: Path, png: Path, dpi: int = DPI) -> None:
    import pymupdf

    with pymupdf.open(pdf) as doc:
        page = doc[0]
        page.get_pixmap(dpi=dpi).save(png)


def render(dpi: int = DPI) -> int:
    if shutil.which("pdflatex") is None:
        print("pdflatex not found -- install TeX Live first", file=sys.stderr)
        return 2
    try:
        import pymupdf  # noqa: F401
    except ImportError:
        print("pymupdf not found -- pip install pymupdf", file=sys.stderr)
        return 2

    sources = sorted(TIKZ.glob("*.tex"))
    if not sources:
        print(f"no TikZ sources in {TIKZ}", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        for source in sources:
            png = TIKZ / (source.stem + ".png")
            try:
                pdf = _compile(source, workdir)
            except subprocess.CalledProcessError as exc:
                print(f"{source.name}: pdflatex failed\n{exc.stdout[-2000:]}", file=sys.stderr)
                return 1
            _rasterise(pdf, png, dpi)
            print(f"{source.name} -> {png.relative_to(ROOT)}  ({png.stat().st_size // 1024} kB)")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="step", required=True)
    sub.add_parser("coordinates", help="rewrite the nozzle wall coordinates from src.geometry")
    render_ap = sub.add_parser("render", help="compile and rasterise the TikZ figures")
    render_ap.add_argument("--dpi", type=int, default=DPI)
    args = ap.parse_args(argv)
    if args.step == "coordinates":
        return coordinates()
    return render(args.dpi)


if __name__ == "__main__":
    raise SystemExit(main())
