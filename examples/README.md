# Examples

Run them from the repository root, in order.  Each is self-contained and prints
what it is doing.

| Script | What it shows | Needs | Roughly |
|---|---|---|---|
| `01_first_solve.py` | one design point, compared against quasi-1D theory | — | 5 s |
| `02_parameter_sweep.py` | sweeping the area ratio; finding the thrust optimum | — | 1 min |
| `03_sensitivity.py` | adjoint gradients, verified against finite differences | JAX | 2 min |
| `04_shape_optimisation.py` | L-BFGS-B on a Bézier wall using adjoint gradients | JAX, SciPy | 5 min |
| `05_convergence_study.py` | observed order of accuracy, and why the contour matters | — | 5 min |
| `06_shock_in_nozzle.py` | back-pressure sweep across all four operating regimes | — | 5 min |

```bash
python examples/01_first_solve.py
```

Each writes a PNG next to wherever you ran it.
