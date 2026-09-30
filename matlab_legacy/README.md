# Legacy MATLAB sources

The original 2D Euler DG nozzle solver, preserved unmodified for reference.

**These files are not used by `dgnozzle` and are not maintained.** They are kept
so that the rewrite can be compared against its origin, and so the defects
listed below can be inspected in context.

## Defects found during the rewrite

| File | Defect |
|---|---|
| `makenozzle.m` | The `'smooth'` diverging branch evaluates its rescale constants at the wrong parameter values (`y0_ex` at `t = π` instead of `2π`), giving `0.16` where `0.35` was intended. The denominator collapses to `0.01`, and the wall reaches **4.34 m** at the exit instead of 0.350 m — passing through **negative** values just downstream of the throat. |
| `residual_calc.m` | The outflow branch always extrapolates, so `p_back_ratio` never reaches the flow. No shocked operating point is computable. |
| `residual_calc.m` | The subsonic-inflow quadratic always takes `(-b + √disc)/2a`, which is the physical root only while `a > 0`. |
| `residual_calc.m` | A negative Roe-averaged sound speed calls `error`, aborting a recoverable transient. |
| `postcalc.m` | Throat height hard-coded as `h = .13989434`, so the thrust coefficient is mis-normalised for any other geometry. The wall integral also runs over the symmetry axis, which shares the `-2` tag. |
| `extrapolate.m` | Reads `resdata.p` as the *old* order, but `main.m` sets it to the *new* one before calling; and the mesh is rebuilt at a different refinement between calls, so element counts disagree. |
| `postprocess.m` | Declared with no arguments but called as `postprocess(resdata, U)`. Loads `.mat` files (`p0Q1ref2.mat`) that `main.m` never writes, and uses undefined variables (`n`, `t`, `e`, `tmax`, `p`, `Q`, `ref`). |
| `quad2d.m` | The degree-3 Dunavant rule has a negative weight (`-0.28125`). |
| `rk4.m` | Convergence is tested on the fourth RK stage's rate against an unnormalised absolute threshold, and there is no iteration cap. |
| `massmatrix.m` | Superseded by `prealloc.m`; forms a dense `inv()` of a sparse global matrix. |

All are fixed in the Python implementation; see the corresponding sections of
`docs/theory.pdf` and the table at the end of the top-level `README.md`.
