# Figures

This inventory records image sources and rebuild commands. Run commands from the repository root unless stated otherwise.

## Why the figures are TikZ *and* PNG

The figures are authored in TikZ, because that is the only way to keep the
nozzle wall in the geometry figure generated from the solver's own contour code
(`make_tikz.py`) rather than drawn by hand. Markdown cannot typeset TikZ, so
the committed PNGs in `figures/tikz/` are what `theory.md` actually shows.
The `.tex` files remain the source of truth; the PNGs are build products that
happen to be committed, so that reading the documentation needs no TeX at all.

## Rebuilding the figures

After changing a TikZ source — or the contour families it draws:

```bash
python docs/make_tikz.py        # only if the contour code changed
python docs/render_figures.py   # needs pdflatex and pymupdf
```

`render_figures.py` needs `texlive-latex-base`, `texlive-pictures`,
`texlive-latex-recommended` and `texlive-latex-extra` (for the `standalone`
class), plus `pip install pymupdf` for the rasterisation step. Both are
build-time only.

Any single figure as a vector PDF, if you want one for print:

```bash
cd docs/figures/tikz && pdflatex nozzle_geometry.tex
```

## The TikZ figures

| Source | Shows |
|---|---|
| `nozzle_geometry.tex` | the nozzle, every design variable, and all four boundary conditions |
| `reference_elements.tex` | reference triangle and square, node ordering, edge numbering |
| `dg_coupling.tex` | how two elements communicate through the numerical flux |
| `mesh_map.tex` | the fixed logical grid and the design-dependent map onto it |

The wall in `nozzle_geometry.tex` is the **actual** contour the solver
produces, sampled from `src.geometry` — not a freehand sketch. `make_tikz.py`
fails loudly rather than silently doing nothing if the figure source changes
shape.

The site build uses committed PNGs and does not require TeX. Regenerate and inspect the image after changing its source; CI validates links but does not compare PNG bytes.

## Current solver figures

The adjacent [rebuild.py](../figures/rebuild.py) is the complete source recipe for the current illustrations. To refresh images in place, omit `--output`; use a temporary directory first to review changes:

```bash
# Rebuild the current figures without overwriting committed images.
python docs/figures/rebuild.py --output /tmp/nozzle-figures
```

| Image | Source / configuration |
|---|---|
| `overview_design.png`, `mesh.png`, `stitched.png` | Smooth AR=2.5, p=1, Q=2; field/mesh/wall/exit overview, mesh and DG/quasi-1D plotting functions |
| `contours.png` | Six `NozzleGeometry` families |
| `area_ratio_sweep.png`, `area_ratio_sweep.csv` | Smooth AR=2–4 at refine=1; every point must converge |
| `choked_mass_flow.png` | Smooth AR=2.5, back pressure 0.05–0.4 |
| `jet_cells.png`, `jet_fields.png`, `nozzle_plume.png` | p=2 nozzle at back pressure 0.064, external ambient ratio 0.03 |
| `tikz/*.png` | Adjacent `.tex` sources, rebuilt by `docs/render_figures.py` |

The recipe generates current illustrations, not byte-identical historical exports. The worked sweep extends to AR=4.5 and records stalled points at AR=4.25 and 4.5 on its refine=1 grid; the illustration recipe uses the verified range and asserts convergence.
