# Verification and troubleshooting

What is checked, and what to do when a run misbehaves.

---

## Verification

Checks that hold to machine precision, all asserted in the test suite:

- **Freestream preservation** (the discrete geometric conservation law) to 1e-17
  — for triangles and quads, `Q=1` and `Q=2`, `p=0,1,2`.
- **Discrete divergence theorem**: `Σ n·ds = 0` per element to 1e-12.
- **Flux consistency and conservation**: `F̂(U,U,n) = F·n` and
  `F̂(L,R,n) = −F̂(R,L,−n)`.
- **Inflow BC** reproduces a uniform isentropic state to 1e-15.
- **Backend agreement** to a relative 4e-15 on the residual.
- **Limiter conservation**: cell averages bit-identical before and after.

Observed orders of accuracy (entropy error, `Q=2`, shock-free):

| contour | `p` | rates |
|---|---|---|
| `analytic` | 1 | 1.93, 1.96 |
| `analytic` | 2 | 2.70 |
| `smooth` | 1 | 1.97, 1.95 |
| `smooth` | 2 | 2.59 |
| `bell` | 1 | 2.30, **0.01 — stalls** |

The C¹ contours reach their design rates. `bell` stalls because its throat
corner is a genuine singularity in the exact solution — physics, not a solver
defect, and the reason to use `smooth` for convergence work.

---


## Troubleshooting

| Symptom | What it means |
|---|---|
| `No module named src` | You ran `python -m src` from somewhere other than the clone. Either `cd` into it, use `./dg2d.sh`, which works from anywhere, or `./dg2d.sh install`. |
| `dg2d.sh: Permission denied` | `chmod +x dg2d.sh`, or run it as `bash dg2d.sh ...`. |
| `dg2d.sh: no python3 on PATH` | Load your Python environment first, or set `DG2D_PYTHON` to the interpreter you want. |
| *"the residual has stalled … That is a limit cycle"* | The mesh cannot resolve a shock or strong expansion. **`refine=+1`** usually fixes it. |
| *"residual converged but the solution is not physical"* | Residual convergence to negative pressure. Enable a limiter, or refine. |
| *"residual became non-finite"* | Reduce `cfl` (try `0.5`), or `scheme='ssprk3'`. |
| *"N cell-average repairs"* | `cfl` is too large; the average went non-physical and had to be floored. |
| *"contour has a non-positive wall height"* | The geometry is invalid. Run `check_contour(geom)` before solving. |
| *"the diverging section is not monotone"* (warning) | The wall bulges — a legitimate but unusual design. Reduce `theta_initial_deg`. |
| A convergence study plateaus | You are probably using `bell`. Use `smooth`. |

Every failure message names both the cause and the fix. The solver detects a
limit cycle and stops rather than burning the whole iteration budget.

---
