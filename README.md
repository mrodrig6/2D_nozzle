# 2D Discontinuous Galerkin nozzle code

A 2D discontinuous Galerkin solver for the compressible Euler equations, built
as a **teaching code for nozzle design studies**.

You change the geometry, sweep it, and take sensitivities. You do not write the
solver — it is already written, verified, and fast.

![Mach field, mesh, wall and exit profiles at the design point](docs/figures/overview_design.png)

```python
from src import solve_nozzle, performance

result = solve_nozzle(area_ratio=3.0, back_pressure_ratio=0.12, order=1)
print(performance(result).summary())
```

---

## Quick start

You need Python 3.10+ and nothing else — there is no install step and nothing
to compile.

```bash
git clone https://github.com/mrodrig6/dg.git
cd dg
./dg2d.sh check                 # what is available
./dg2d.sh solve area_ratio=3.0 back_pressure_ratio=0.12 order=1
```

`check` reports a `NO` for each optional piece you are missing; `./dg2d.sh
install` adds them all. A solve prints its convergence history and then:

```
thrust        0.054007  (c_F = 0.193026, 92.24% of ideal)
  wall form   0.053504  (imbalance 9.31e-03)
mass flow     in 0.301234, out 0.301633  (imbalance 1.32e-03)
exit          M = 2.7188, p/p_t = 0.05041
entropy error 1.8657e-02
```

Run a worked case, or list them all:

```bash
./dg2d.sh list
./dg2d.sh run sweep             # examples/02_sweep.py
```

→ Full launcher reference and the Python API: **[`docs/usage.md`](docs/usage.md)**

---

## What it does

**Solves.** Nodal DG on triangles or quads, polynomial order 0–4, curved
elements, marched to steady state. Two interface fluxes: Roe with the
Harten–Hyman entropy fix (the default) and HLLC with Batten's wave speeds.
→ [`docs/theory.md`](docs/theory.md)

**Designs.** Five contour families (conical, smooth, bell, method-of-
characteristics, Bézier), each parameterised so you can sweep or optimise it.
→ [`docs/usage.md`](docs/usage.md#design-variables)

**Differentiates.** Adjoint gradients through the whole solve via JAX, verified
against finite differences, so shape optimisation costs one adjoint rather than
one solve per variable. → [`docs/workflows.md`](docs/workflows.md)

**Looks outside the nozzle.** A nozzle designed to be shock free inside has its
whole wave system *outside* the exit, so `src/external/` computes that in closed
form — the lip wave and the repeating shock-cell pattern downstream.
→ [`docs/workflows.md`](docs/workflows.md)

![Shock cells outside the nozzle](docs/figures/jet_fields.png)

**Runs fast.** A Numba backend with fused Runge–Kutta stages, a NumPy reference
backend, and JAX for gradients. → [`docs/performance.md`](docs/performance.md)

---

## Where to read next

| If you want to | Read |
|---|---|
| Drive the code — launcher, Python API, design variables, resolution | [`docs/usage.md`](docs/usage.md) |
| Do something with it — sweeps, sensitivity, optimisation, external flow | [`docs/workflows.md`](docs/workflows.md) |
| Understand the formulation — equations, DG weak form, fluxes, limiters, adjoint | [`docs/theory.md`](docs/theory.md) |
| Set exercises for students | [`docs/lab_guide.md`](docs/lab_guide.md) |
| Know how fast it is and why | [`docs/performance.md`](docs/performance.md) |
| Check it is right, or fix a misbehaving run | [`docs/verification.md`](docs/verification.md) |
| Rebuild the figures | [`docs/README.md`](docs/README.md) |

Worked, runnable cases live in [`examples/`](examples/) — start with
[`examples/01_solve.py`](examples/01_solve.py).

---

## One limitation worth knowing first

**The solver refuses operating points that put a normal shock inside the
diverging section.** Between the second and first critical pressure ratios the
pseudo-time march does not reach a steady state, so rather than return numbers
that look like an answer, `solve_nozzle` raises and tells you the band.

This is not a restriction on nozzle design — it is most of the point of it.
Sizing a nozzle means *avoiding* a shock in the diverging section, and the
interesting waves (oblique shocks when over-expanded, a Prandtl–Meyer fan when
under-expanded) form outside the exit plane, which is exactly what
`src/external/` draws. Pass `allow_shock_in_nozzle=True` to opt back in; the
run will report `converged=False`.

→ [`docs/theory.md`](docs/theory.md#a-limitation-shocked-operating-points)

---

## Licence

This project is licensed under the MIT License — see the [`LICENSE`](LICENSE)
file for details.

Copyright (c) 2026 dgnozzle contributors
