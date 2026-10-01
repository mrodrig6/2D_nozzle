# Nozzle design lab

A student-facing guide. You will not write solver code; you will **engineer the
geometry** and explain what the flow does in response.

Read [`theory.md`](theory.md) alongside this — it defines every quantity used
below, and the geometry figure in §3 names every design variable.

---

## Before you start

Run this and make sure it works:

```bash
python examples/01_first_solve.py
```

Then get comfortable with the one function you need:

```python
from dgnozzle import solve_nozzle, performance

result = solve_nozzle(area_ratio=2.5, back_pressure_ratio=0.15, order=1)
assert result.converged, result.message
print(performance(result).summary())
```

**Always assert `result.converged`.** A solver that has not converged will still
hand you numbers, and they will be wrong. When it fails, `result.message` tells
you both the cause and the fix.

---

## The physics you are working with

A converging–diverging nozzle accelerates flow to supersonic speed. Two things
control it:

- **Area ratio** sets *how fast the flow can go*. Through the area–Mach
  relation, a larger exit-to-throat ratio means a higher design exit Mach number.
- **Back pressure ratio** sets *what the flow actually does*. The same nozzle
  behaves completely differently at different `p_back/p_t`.

Before running anything, ask the solver what regimes exist for your geometry:

```python
from dgnozzle import critical_ratios
print(critical_ratios(2.5).describe())
# first=0.9609 (choking), second=0.4345 (shock at exit), third=0.0639 (design)
```

| `back_pressure_ratio` | What happens |
|---|---|
| above `first` | not choked — subsonic everywhere; mass flow still rises as you lower `p_back` |
| `second` … `first` | **choked**, with a normal shock standing in the diverging section |
| `third` … `second` | shock-free inside; **over-expanded** (shocks form outside) |
| at `third` | **design point** — perfectly expanded |
| below `third` | **under-expanded** (expansion fans outside) |

Once choked, the mass flow stops responding to back pressure entirely. That is
worth confirming yourself — see Exercise 2.

---

## Exercise 1 — Find the thrust optimum

**Question.** At a fixed back pressure, which area ratio produces the most
thrust? Why is it not simply the largest one?

```python
import numpy as np
from dgnozzle import sweep

table = sweep(area_ratio=np.linspace(2.0, 4.5, 11),
              contour="smooth", order=1, refine=1,
              back_pressure_ratio=0.15)
print(table.table())
```

**What to report.** The optimum area ratio, and the two competing effects that
produce it. Plot `thrust_coefficient` and `exit_pressure_ratio` against area
ratio on the same axes — the crossing tells the story.

**Check yourself.** Compare `exit_mach` against `quasi1d_exit_mach` in the same
table. Where do they start to disagree, and why?

---

## Exercise 2 — Confirm choking

**Question.** Show that once the nozzle is choked, mass flow is independent of
back pressure.

Sweep only the **shock-free** range — below the second critical ratio — because
shocked points do not converge (see *Limitations* at the end of this guide):

```python
from dgnozzle import critical_ratios
crit = critical_ratios(2.5)
table = sweep(back_pressure_ratio=np.linspace(0.05, crit.second * 0.95, 12),
              area_ratio=2.5, contour="smooth", order=1)
print(table.table(("mass_flow_in", "exit_mach", "thrust_coefficient")))
```

**What to report.** `mass_flow_in` against `back_pressure_ratio`. Every point
here is choked, so the curve should be *flat to within discretisation error* —
quantify that error and say whether the flatness is convincing.

**Then extend it with theory.** `solve_quasi1d` covers the whole range
including the shocked part, so use it to show where the mass flow *would* start
to respond:

```python
from dgnozzle import solve_quasi1d, NozzleGeometry, FlowConditions
for pb in (0.999, 0.99, 0.97, 0.9, 0.5, 0.15):
    s = solve_quasi1d(NozzleGeometry(contour="smooth", area_ratio=2.5),
                      FlowConditions(back_pressure_ratio=pb))
    print(f"{pb:.3f}  mdot={s.mass_flow:.6f}  {s.regime.value}")
```

Compare the flat part against your DG numbers.

---

## Exercise 3 — Track a shock, and find out why the solver will not

**Question.** Where does a normal shock stand, and how does its position respond
to back pressure?

Quasi-1D theory answers this exactly:

```python
from dgnozzle import solve_quasi1d, NozzleGeometry, FlowConditions
geom = NozzleGeometry(contour="smooth", area_ratio=2.5)
for pb in (0.9, 0.7, 0.5):
    s = solve_quasi1d(geom, FlowConditions(back_pressure_ratio=pb))
    print(f"pb/pt={pb}: shock at x={s.shock_x:.4f}, M1={s.shock_mach:.3f}, "
          f"M_exit={s.exit_mach:.3f}")
```

**What to report.** Shock position and upstream Mach number against back
pressure, and the total-pressure loss across the shock. Explain why a nozzle is
never *designed* to run in this regime.

**Now try the DG solver on the same point** and watch it fail:

```python
r = solve_nozzle(back_pressure_ratio=0.70, order=0, refine=1,
                 contour="smooth", max_iterations=40000)
print(r.converged, r.message)
import matplotlib.pyplot as plt
from dgnozzle.plotting import plot_convergence
plot_convergence(r)     # the residual falls, then parks
plt.show()
```

**What to report.** The residual history, and the level it parks at for
`refine=0`, `1` and `2`. The floor is essentially mesh-independent — so this is
*not* a resolution problem. Two-dimensionality is the thing quasi-1D theory
cannot see: a normal shock in a diverging duct is not obviously a steady 2D
structure. Take a position on whether the solver is failing to find a steady
solution, or whether there is no steady solution to find. Say what evidence
would settle it.

This is a real open issue in the code, not a contrived exercise. See
*Limitations*.

---

## Exercise 4 — Does the contour shape matter?

**Question.** At identical area ratio and throat location, do different wall
shapes give different thrust?

```python
for contour in ("conical", "smooth", "bell", "moc", "bezier"):
    r = solve_nozzle(contour=contour, area_ratio=2.5, order=1,
                     refine=1, geometry_order=2, verbose=False)
    print(f"{contour:9s} c_F = {performance(r).thrust_coefficient:.5f}")
```

**What to report.** Rank them, and explain the ranking using the **exit
profile** (`plot_exit_profile`) rather than the thrust number alone. A nozzle
that leaves the flow with transverse velocity has wasted momentum — that is
divergence loss, and you can see it directly.

---

## Exercise 5 — Convergence, done properly

**Question.** Does the solver achieve its design order of accuracy, and what
limits it?

```python
python examples/05_convergence_study.py
```

**What to report.** Observed rates for `p=1` and `p=2` with `contour='smooth'`,
and the rate for `contour='bell'`.

**The point of this exercise.** `bell` *stalls*. Its throat has a deliberate
slope discontinuity — the sharp-corner expansion of classical minimum-length
nozzle design — which launches a Prandtl–Meyer fan. That is a genuine
singularity in the exact solution, and no amount of refinement recovers the
asymptotic rate. It is physics, not a solver bug.

Two consequences you should internalise:
1. A convergence study must use a smooth contour (`'smooth'` or `'analytic'`).
2. Geometry order matters. With `geometry_order=1` the straight-sided wall
   caps the rate near 1.6 regardless of `p`.

---

## Exercise 6 — Sensitivity analysis

**Question.** Which design variable does thrust care about most?

```python
from dgnozzle import differentiable_case, check_gradient

dc = differentiable_case(contour="bezier", order=1,
                         back_pressure_ratio=0.15, tolerance=1e-11)
check_gradient(dc, "thrust", names=("area_ratio", "throat_x"))   # do this first

value, grad, _ = dc.value_and_gradient(
    "thrust", names=("area_ratio", "throat_x", "bezier_w1", "bezier_w2"))
```

**Verify before you trust.** Run `check_gradient` once and confirm the adjoint
matches finite differences. It should agree to ~1e-6 or better.

**What to report.** The gradients, *non-dimensionalised* so they are comparable
(a derivative with respect to an area ratio and one with respect to a length are
not the same kind of number — scale each by its variable's magnitude). Rank the
variables by influence.

**Something to explain.** `d(thrust)/d(back_pressure)` comes out exactly zero at
this operating point. Why? (Hint: look at the exit Mach number.)

---

## Exercise 7 — Optimise a shape

```python
python examples/04_shape_optimisation.py
```

**What to report.** The optimised wall against the initial one, the thrust
improvement, and — importantly — **whether the optimum moves when you refine the
mesh**. It will, a little. That movement is discretisation error, and reporting
it is the difference between an optimisation result and an optimisation claim.

---

## Choosing resolution

| Purpose | Settings | Cost |
|---|---|---|
| exploring, sweeping | `order=1, refine=0` | ~0.5 s |
| a number for a report | `order=2, geometry_order=2, refine=1` | ~10 s |
| a convergence study | `contour='smooth', geometry_order=2, refine=0,1,2` | minutes |
| a shocked operating point | not currently possible — see *Limitations* | — |

Sweeps warm-start automatically, so a 20-point sweep costs far less than 20
solves.

---

## When it fails

Read the message. Every failure names both cause and remedy.

| Message contains | Do this |
|---|---|
| "stalled … limit cycle" | `refine=+1` — the mesh cannot resolve a shock or expansion |
| "not physical" | enable a limiter, or refine |
| "non-finite" | `cfl=0.5`, or `scheme='ssprk3'` |
| "cell-average repairs" | `cfl` is too large |
| "non-positive wall height" | your geometry is invalid — run `check_contour` first |
| "not monotone" (warning) | the wall bulges; reduce `theta_initial_deg` |

---

## Limitations

**Shocked operating points do not converge.** For `back_pressure_ratio` between
the second and first critical ratios, the residual falls about an order of
magnitude and then parks:

| settings | residual floor (scaled) |
|---|---|
| `p=0`, `refine=0/1/2` | 6.1e-2 / 4.9e-2 / 5.6e-2 |
| `p=1` + `barth-jespersen` + `ssprk3`, `refine=0/1` | 1.5 / 2.6 |

The `p=0` floor is mesh-independent, and `p>=1` with a limiter is *worse* than
`p=0`. Localising the residual shows 85% of it in the single axial band
`x ∈ [0.2, 0.3)` — where the shock sits — and only 0.1% at the outflow, which
rules out the boundary conditions. The exit plane also shows reverse flow
(minimum normal Mach −0.22): a 2D recirculation quasi-1D theory cannot
represent. The solver reports these as `converged=False` and does not pass the
numbers off as trustworthy, but it cannot currently compute them. Use
`solve_quasi1d` for shock physics.

Everything shock-free is verified and converges cleanly: the design point,
over-expanded and under-expanded operation, the whole area-ratio design space,
and all sensitivity and optimisation work.

---

## Reporting checklist

- [ ] Every result came from a run where `converged` was `True`
- [ ] Resolution stated (`order`, `geometry_order`, `refine`, element count)
- [ ] At least one quantity compared against quasi-1D theory
- [ ] `thrust_imbalance` reported — it is your discretisation-error estimate
- [ ] Any gradient verified against finite differences at least once
- [ ] Claims about *optimality* checked on at least two mesh resolutions
