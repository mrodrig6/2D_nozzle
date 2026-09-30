# Nozzle design lab

A student-facing guide. You will not write solver code; you will **engineer the
geometry** and explain what the flow does in response.

Read [`theory.pdf`](theory.tex) alongside this — it defines every quantity used
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

```python
table = sweep(back_pressure_ratio=np.linspace(0.05, 0.99, 20),
              area_ratio=2.5, contour="smooth", order=1,
              limiter="barth-jespersen", scheme="ssprk3")
```

**What to report.** `mass_flow_in` against `back_pressure_ratio`, with the three
critical ratios marked. Identify the exact point where the curve goes flat and
compare it to `critical_ratios(...).first`.

**Watch out.** Shocked cases need `limiter='barth-jespersen'` and
`scheme='ssprk3'`. Without a limiter the `p=1` solution overshoots into negative
pressure at the shock, and the solver will (correctly) refuse to call that
converged.

---

## Exercise 3 — Track a shock

**Question.** Where does the normal shock stand, and how does its position
respond to back pressure?

```python
from dgnozzle.plotting import plot_field, plot_centreline
result = solve_nozzle(back_pressure_ratio=0.70, order=1, refine=1,
                      contour="smooth", limiter="barth-jespersen",
                      scheme="ssprk3")
plot_field(result, "mach")
plot_centreline(result)     # DG against quasi-1D, with the theoretical shock marked
```

**What to report.** The shock position from your DG solution against the
quasi-1D prediction, for three back pressures. Quasi-1D assumes a *normal* shock
spanning the channel. Look at the Mach field — is the computed shock actually
normal? What does it look like near the wall?

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
| anything with a shock | `+ limiter='barth-jespersen', scheme='ssprk3'` | slower |

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

## Reporting checklist

- [ ] Every result came from a run where `converged` was `True`
- [ ] Resolution stated (`order`, `geometry_order`, `refine`, element count)
- [ ] At least one quantity compared against quasi-1D theory
- [ ] `thrust_imbalance` reported — it is your discretisation-error estimate
- [ ] Any gradient verified against finite differences at least once
- [ ] Claims about *optimality* checked on at least two mesh resolutions
