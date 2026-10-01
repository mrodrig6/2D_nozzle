# Examples

Run them through the launcher, which needs no install and works from any
directory:

```bash
./dg2d.sh run sweep
```

`./dg2d.sh list` prints the same table. The number is the reading order; `run`
matches on the name, so `./dg2d.sh run 02` works too.

| `run` | Script | What it shows | Needs | Roughly |
|---|---|---|---|---|
| `solve` | `01_solve.py` | one design point, compared against quasi-1D theory | — | 5 s |
| **`sweep`** | `02_sweep.py` | **sweeping the area ratio; finding the thrust optimum** | — | 1 min |
| **`sensitivity`** | `03_sensitivity.py` | **adjoint gradients, verified against finite differences** | JAX | 2 min |
| **`optimise`** | `04_optimise.py` | **L-BFGS-B on a Bézier wall using adjoint gradients** | JAX, SciPy | 5 min |
| `convergence` | `05_convergence.py` | observed order of accuracy, and why the contour matters | — | 5 min |
| `shock` | `06_shock.py` | back-pressure sweep across all four operating regimes | — | 5 min |

The three in bold are the workflows the course is built around — parameter
sweeps, sensitivity analysis and shape optimisation. They are the ones to read
first and the ones the exercises in
[`../docs/lab_guide.md`](../docs/lab_guide.md) extend.

Each is self-contained, prints what it is doing, and writes a PNG into the
directory you ran it from. They are meant to be copied and edited: changing a
geometry in one of these is the normal way to start a study.

Running them directly works too, if you have the package installed or `src/` on
your `PYTHONPATH`:

```bash
python examples/02_sweep.py
```
