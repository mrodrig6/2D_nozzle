# Documentation

| File | Contents |
|---|---|
| [`theory.tex`](theory.tex) → `theory.pdf` | **The formulation and geometry definition.** Governing equations, DG weak form, Roe flux, boundary conditions, limiters, quasi-1D theory, performance metrics, verification evidence, and the adjoint. All equations in LaTeX; all figures in TikZ. |
| [`lab_guide.md`](lab_guide.md) | Student-facing exercises. |
| [`tikz/`](tikz/) | TikZ sources. Each compiles standalone *and* embeds in `theory.tex`. |
| [`make_tikz.py`](make_tikz.py) | Regenerates the geometry figure's coordinates from the solver's own contour code. |
| `figures/` | PNGs used by the top-level README. |

## Building

```bash
pdflatex theory.tex && pdflatex theory.tex     # twice, for the contents page
```

Needs `texlive-latex-base`, `texlive-pictures`, `texlive-latex-recommended` and
`texlive-latex-extra` (for the `standalone` class).

Any single figure on its own:

```bash
cd tikz && pdflatex nozzle_geometry.tex
```

## The TikZ figures

| Source | Shows |
|---|---|
| `nozzle_geometry.tex` | the nozzle, every design variable, and all four boundary conditions |
| `reference_elements.tex` | reference triangle and square, node ordering, edge numbering |
| `dg_coupling.tex` | how two elements communicate through the numerical flux |
| `mesh_map.tex` | the fixed logical grid and the design-dependent map onto it |

Each guards its `standalone` wrapper with `\ifdefined\DGNOZZLEEMBEDDED`, so the
same file serves both as a standalone PDF and as an `\input` inside
`theory.tex`.

The wall in `nozzle_geometry.tex` is the **actual** contour the solver
produces, sampled from `dgnozzle.geometry` — not a freehand sketch. After
changing the contour families:

```bash
python make_tikz.py
cd tikz && pdflatex nozzle_geometry.tex
```
