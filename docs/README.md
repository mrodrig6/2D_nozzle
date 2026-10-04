# Documentation

| File | Contents |
|---|---|
| [`usage.md`](usage.md) | **Driving the code.** Launcher reference, the Python API, every design variable, and how to choose a resolution. |
| [`workflows.md`](workflows.md) | **What to do with it.** Parameter sweeps, sensitivity analysis, shape optimisation, and the external flow outside the exit. |
| [`performance.md`](performance.md) | Backends, the time step, and what made it fast. |
| [`verification.md`](verification.md) | What is checked, and what to do when a run misbehaves. |
| [`theory.md`](theory.md) | **The formulation and geometry definition.** Governing equations, DG weak form, Roe flux, boundary conditions, limiters, quasi-1D theory, performance metrics, verification evidence, and the adjoint. Markdown with LaTeX equations — it renders on GitHub, so there is nothing to build to read it. |
| [`lab_guide.md`](lab_guide.md) | Student-facing exercises. |
| [`tikz/`](tikz/) | TikZ sources for every figure in `theory.md`. |
| [`render_figures.py`](render_figures.py) | Compiles the TikZ sources and rasterises them into `figures/tikz/`, which is what `theory.md` links to. |
| [`make_tikz.py`](make_tikz.py) | Regenerates the geometry figure's coordinates from the solver's own contour code. |
| `figures/` | The PNGs the documents show. `figures/tikz/` is generated from `tikz/`; the rest come from the solver via `plotting`. |

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
cd docs/tikz && pdflatex nozzle_geometry.tex
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
shape, which is what makes the CI staleness check on it worth having.
