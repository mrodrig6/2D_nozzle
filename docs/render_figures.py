#!/usr/bin/env python3
"""Rasterise the TikZ figures so ``theory.md`` can show them.

``docs/theory.md`` is Markdown, which has no way to typeset TikZ.  The figures
are still *authored* in TikZ -- that is the source of truth, and the nozzle wall
in ``nozzle_geometry.tex`` is generated from the solver's own contour code by
``make_tikz.py`` -- so this script compiles each one and rasterises the result
into ``docs/figures/tikz/``, which is what the Markdown links to.

Run:  python docs/render_figures.py

Needs ``pdflatex`` (TeX Live: ``texlive-latex-base``, ``texlive-pictures``,
``texlive-latex-extra`` for ``standalone.cls``) and ``pymupdf`` for the
rasterisation step::

    pip install pymupdf

Both are build-time only.  Nobody reading the documentation needs either, which
is the point of committing the PNGs.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

DOCS = Path(__file__).resolve().parent
TIKZ = DOCS / "tikz"
OUT = DOCS / "figures" / "tikz"

#: Rendering resolution.  The figures are vector line art at roughly 16 cm
#: wide; 200 dpi keeps the labels legible when GitHub scales them down to the
#: README column width without making the files large.
DPI = 200


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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dpi", type=int, default=DPI)
    args = ap.parse_args(argv)

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

    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        for source in sources:
            png = OUT / (source.stem + ".png")
            try:
                pdf = _compile(source, workdir)
            except subprocess.CalledProcessError as exc:
                print(f"{source.name}: pdflatex failed\n{exc.stdout[-2000:]}", file=sys.stderr)
                return 1
            _rasterise(pdf, png, args.dpi)
            print(
                f"{source.name} -> {png.relative_to(DOCS.parent)}"
                f"  ({png.stat().st_size // 1024} kB)"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
