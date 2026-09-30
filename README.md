# 2D Discontinuous Galerkin nozzle code

A 2D discontinuous Galerkin solver for the compressible Euler equations, built
for **nozzle design studies**.

You change the geometry, sweep it, and take sensitivities. You do not write the
solver — it is already written, verified, and fast.

![Mach field, mesh, wall and exit profiles at the design point](docs/figures/overview_design.png)

```python
from dgnozzle import solve_nozzle, performance

result = solve_nozzle(area_ratio=3.0, back_pressure_ratio=0.12, order=1)
print(performance(result).summary())
```

---

## Contents

- [Install](#install)
- [Run it](#run-it)
- [Documentation](#documentation)
- [Design variables](#design-variables)
- [What students do with it](#what-students-do-with-it)
  - [1. Parameter sweeps](#1-parameter-sweeps)
  - [2. Sensitivity analysis](#2-sensitivity-analysis)
  - [3. Shape optimisation](#3-shape-optimisation)
- [Command line](#command-line)
- [Choosing resolution](#choosing-resolution)
- [Backends and performance](#backends-and-performance)
- [Choking, as a check on the solver](#choking-as-a-check-on-the-solver)
- [Known limitation: shocked operating points](#known-limitation-shocked-operating-points-do-not-converge)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)
- [Relationship to the original MATLAB code](#relationship-to-the-original-matlab-code)

---

## Install

```bash
git clone https://github.com/mrodrig6/dg.git
cd dg
pip install -e ".[all]"
```

`[all]` pulls in Numba (speed), JAX (gradients) and matplotlib (figures). The
solver itself needs only NumPy and SciPy; everything optional degrades
gracefully with a clear message if it is missing.

Check the install:

```bash
python -m dgnozzle solve --order 1
pytest -q -m "not slow"   # 276 unit tests, under a minute
pytest -q                 # plus the end-to-end solves and adjoint checks
```

The `slow` marker covers the tests that run a real flow solve — including the
finite-difference gradient checks, which are two extra solves per design
variable. Run those before trusting a change to the physics; the fast suite is
enough while iterating.

There is **nothing to compile and no binary to download**. Numba compiles the
kernels on first use (a few seconds, cached afterwards).

---

## Run it

Everything goes through one function. Give it the design you want; leave the
rest alone.

```python
from dgnozzle import solve_nozzle, performance

result = solve_nozzle(
    contour="smooth",           # wall family
    area_ratio=2.5,             # exit / throat
    throat_x=0.14,              # throat location, fraction of length
    back_pressure_ratio=0.15,   # p_back / p_total — sets the operating point
    order=1,                    # polynomial order p
    refine=0,                   # mesh refinement level
)

print(result.summary())
print(performance(result).summary())
```

```
converged in 1701 iterations (0.63 s, numba): residual 9.169e-07 scaled
  (1.80e-06 of initial); min rho 3.0174e-01, min p 5.1730e-02

thrust        0.045825  (c_F = 0.163785, 90.09% of ideal)
  wall form   0.045344  (imbalance 1.05e-02)
mass flow     in 0.301380, out 0.301775  (imbalance 1.31e-03)
exit          M = 2.4490, p/p_t = 0.06661
entropy error 3.7307e-03
```

Always check `result.converged` before trusting the numbers. When it is
`False`, `result.message` says what went wrong and what to do about it.

> **On convergence.** The residual is measured against the problem's own
> physical scale `ρ_t a_t / L`, not against the first iteration's residual. A
> relative criterion would demand a tighter absolute residual the better your
> initial guess is — making a good starting field look *slower* than a bad one,
> and making two runs started differently incomparable.

### Plotting

```python
from dgnozzle.plotting import overview, plot_field, plot_centreline
import matplotlib.pyplot as plt

overview(result)              # four-panel summary
plot_field(result, "mach")    # any of: mach, pressure, density, temperature,
                              #         u, v, velocity, entropy
plot_centreline(result)       # axial profile against quasi-1D theory
plt.show()
```

---

## Documentation

| Document | What is in it |
|---|---|
| **[`docs/theory.pdf`](docs/theory.tex)** | **The formulation and the geometry definition.** Governing equations, the DG weak form, the Roe flux, every boundary condition, the limiters, quasi-1D theory, the thrust and entropy-error definitions, and the verification evidence — all in LaTeX, with TikZ figures. Start here. |
| [`docs/lab_guide.md`](docs/lab_guide.md) | The student-facing lab: exercises, what to look for, what to report. |
| [`docs/tikz/`](docs/tikz/) | TikZ sources for every figure. Each compiles standalone *and* embeds in `theory.tex`. |
| Docstrings | Every module carries its own derivation and rationale. `help(dgnozzle.physics)` is worth reading. |

Build the theory document:

```bash
cd docs && pdflatex theory.tex && pdflatex theory.tex
```

The geometry figure is generated from the *actual* contour the solver uses, not
sketched by hand. Regenerate it after changing the contour families:

```bash
python docs/make_tikz.py
cd docs/tikz && pdflatex nozzle_geometry.tex
```

---

## Design variables

The nozzle is **planar** (a 2D channel of unit depth), so the one-dimensional
area is the channel height and the area ratio is the height ratio —
isentropic tables apply directly.

![Contour families](docs/figures/contours.png)

| Keyword | Meaning | Default |
|---|---|---|
| `contour` | wall family, see below | `'bell'` |
| `area_ratio` | exit-to-throat area ratio; sets the design Mach number | `2.5019` |
| `throat_x` | throat location as a fraction of length | `0.1388` |
| `throat_half_height` | throat half-height [m]; scales the whole nozzle | `0.13989434` |
| `inlet_half_height` | inlet half-height [m] | `0.15` |
| `length` | axial length [m] | `1.0` |
| `theta_initial_deg` | wall angle just past the throat | auto (monotone) |
| `theta_exit_deg` | wall angle at the exit plane | `0.0` |
| `bezier_w1`, `bezier_w2` | Bézier shape weights | `0.55`, `0.90` |
| `back_pressure_ratio` | `p_back / p_total`; sets the operating point | `0.15` |

### Contour families

| `contour` | Shape | Use it for |
|---|---|---|
| `'bell'` | cubic Hermite with prescribed wall angles | the general-purpose design family |
| `'smooth'` | `'bell'` with a zero throat angle, so the wall is C¹ | **convergence studies** |
| `'conical'` | straight diverging wall | the simplest baseline |
| `'moc'` | `'bell'` at the classical Rao angles (30°, 0°) | textbook comparison |
| `'bezier'` | cubic Bézier, two shape weights | **shape optimisation** |
| `'analytic'` | the fixed contour of the original MATLAB code | reproducing legacy results |

> **A deliberate corner.** With `theta_initial_deg > 0` the wall slope jumps at
> the throat. That sharp-corner expansion is the classical minimum-length
> idealisation, not a mistake — but it puts a Prandtl–Meyer singularity in the
> exact solution, which **caps the achievable order of accuracy**. A convergence
> study must use `'smooth'` or `'analytic'`. See
> [Verification](#verification).

### Operating point

`back_pressure_ratio` is what makes the nozzle interesting. Three critical
values divide the map — ask for them before you run anything:

```python
from dgnozzle import critical_ratios
print(critical_ratios(2.5019).describe())
# first=0.9609 (choking), second=0.4345 (shock at exit), third=0.0639 (design, M_exit=2.444)
```

| `back_pressure_ratio` | Regime |
|---|---|
| above `first` | not choked; subsonic throughout |
| between `second` and `first` | **choked, normal shock in the diverging section** |
| between `third` and `second` | shock-free, over-expanded |
| at `third` | design point, perfectly expanded |
| below `third` | under-expanded |

**Shocked cases (between the second and first critical ratios) do not currently converge** — see [Known limitation](#known-limitation-shocked-operating-points-do-not-converge).

---

## What students do with it

### 1. Parameter sweeps

```python
import numpy as np
from dgnozzle import sweep

table = sweep(
    area_ratio=np.linspace(2.0, 4.0, 9),   # iterable  -> a sweep axis
    order=1, contour="smooth",             # scalar    -> a fixed setting
)
print(table.table())
table.to_csv("sweep.csv")
```

![Area-ratio sweep: thrust coefficient and exit Mach against quasi-1D theory](docs/figures/area_ratio_sweep.png)

Thrust **peaks and then falls** — past the matched condition the nozzle
over-expands and the extra area costs more than the extra exit Mach number buys.
On this sweep the optimum is at `area_ratio = 3.25` (`c_F = 0.176`), and thrust
efficiency drops from 0.956 to 0.869 across the range. That trade is the point of
Exercise 1 in the lab guide.

Sweeps **warm start** from the previous point automatically. Measured on a
7-point area-ratio sweep that is 40% fewer iterations and 1.8× less wall time.
Two parameters give a full grid:

```python
grid = sweep(
    area_ratio=np.linspace(2.0, 4.0, 7),
    back_pressure_ratio=np.linspace(0.08, 0.40, 6),
    order=1,
)
from dgnozzle.plotting import plot_sweep
plot_sweep(grid, "thrust_coefficient")     # 2D contour map
```

Every point records 19 metrics (`dgnozzle.METRICS`), including the quasi-1D
reference values so you can see where two-dimensionality starts to matter.
Points that fail are recorded, not hidden:

```python
for point, why in table.failures():
    print(point, "→", why)
```

### 2. Sensitivity analysis

Exact derivatives by the **discrete adjoint** — one solve, any number of design
variables:

```python
from dgnozzle import differentiable_case

dc = differentiable_case(order=1, contour="bezier", back_pressure_ratio=0.15)

value, grad, _ = dc.value_and_gradient(
    "thrust",
    names=("area_ratio", "throat_x", "bezier_w1", "bezier_w2", "theta_exit"),
)
print(value, grad)
```

Verify it against finite differences — and make your students do this once:

```python
from dgnozzle import check_gradient
check_gradient(dc, "thrust", names=("area_ratio", "throat_x"))
```

```
objective thrust = 0.0458513361
parameter                     adjoint    finite diff    rel err
area_ratio               9.772082e-03   9.772082e-03   2.12e-08
throat_x                -1.490300e-02  -1.490297e-02   2.01e-06
```

Objectives: `thrust`, `thrust_coefficient`, `exit_mach`, `mass_flow`,
`exit_pressure`. Design variables: every geometric one, plus `back_pressure`.

> The gradient is of the **discrete** problem, which is what optimisation needs.
> Use a shock-free operating point: a limiter switching on and off introduces
> kinks, and a shock's position is only piecewise differentiable.

### 3. Shape optimisation

The gradient plugs straight into SciPy:

```python
import numpy as np
from scipy.optimize import minimize
from dgnozzle import differentiable_case

dc = differentiable_case(order=1, contour="bezier")
names = ("area_ratio", "bezier_w1", "bezier_w2")

def negative_thrust(x):
    params = dict(zip(names, x))
    value, grad, _ = dc.value_and_gradient("thrust", params, names=names)
    return -value, -np.array([grad[n] for n in names])

x0 = np.array([dc.default_params()[n] for n in names])
best = minimize(negative_thrust, x0, jac=True, method="L-BFGS-B",
                bounds=[(1.5, 5.0), (0.0, 1.0), (0.0, 1.0)])
```

Worked versions of all three live in [`examples/`](examples/).

---

## Command line

```bash
python -m dgnozzle solve --area-ratio 3.0 --back-pressure-ratio 0.12 -p 1 \
                         --figure result.png
python -m dgnozzle sweep area_ratio 2.0 4.0 9 -p 1 --csv sweep.csv
python -m dgnozzle geometry --contour bezier --bezier-w1 0.7 --figure wall.png
python -m dgnozzle bench -p 1 --refine 1
```

`geometry` runs no flow solve — use it to check a contour before committing to a
simulation.

---

## Choosing resolution

![Meshes](docs/figures/mesh.png)

| Knob | Meaning |
|---|---|
| `order` (`p`) | solution polynomial order. `0` is finite volume; `1` and `2` are 2nd and 3rd order. |
| `geometry_order` (`Q`) | `1` straight-sided, `2` curved. `Q=2` represents the curved wall ~125× more accurately at the same element count. |
| `refine` | uniform refinement; each level multiplies elements by 4. |
| `element` | `'tri'` (default) or `'quad'`. |

Timings on 4 cores, converged to a relative residual of 1e-6:

| `p` | `refine` | elements | DOF | iterations | time |
|---|---|---|---|---|---|
| 0 | 0 | 140 | 140 | 851 | **0.20 s** |
| 1 | 0 | 140 | 420 | 1751 | **0.59 s** |
| 2 | 0 | 140 | 840 | 3301 | **1.8 s** |
| 1 | 1 | 560 | 1680 | 2501 | **1.9 s** |
| 2 | 1 | 560 | 3360 | 6301 | **7.3 s** |
| 1 | 2 | 2240 | 6720 | 6051 | **12 s** |

For comparison, the original MATLAB code documented ~30 s for the first row and
"several minutes" for `p=1, refine=1`.

**Start with `p=1, refine=0`** for design exploration — it is under a second and
already within a few percent on thrust. Move to `p=2, geometry_order=2,
refine=1` for numbers you will put in a report.

---

## Backends and performance

Same equations, three execution strategies:

| `backend` | Role |
|---|---|
| `'numba'` | compiled nopython kernels — **the default and the fastest** |
| `'numpy'` | vectorised reference; no compile step, easiest to read |
| `'jax'` | the same vectorised source under `jax.numpy`; **differentiable** |

```bash
python -m dgnozzle bench -p 1 --refine 1
```

They agree to a relative 1e-10, which the test suite enforces.

Two things make it fast, and two make it robust. None of them change the answer
— and the distinction is measured, not assumed.

**Faster:**

- **Vectorised, gather-only assembly.** No Python loop over elements or edges,
  and no scatter-add — so the loops parallelise without atomics and the same
  source runs under all three backends.
- **A limiter fast path.** On a smooth solution nothing violates positivity, so
  the pressure bisection is skipped entirely. This is not a micro-optimisation:
  the limiter runs once per Runge–Kutta stage, and in naive vectorised form it
  cost ~20× a residual evaluation and dominated the whole solve. Moving it into
  a kernel with a zero-allocation fast path took one RK4 step from 3.92 ms to
  0.34 ms.

**More robust, at a small cost:**

- **Quasi-1D initial condition** (on by default). The shock is roughly in place
  and the nozzle already choked before iteration one. It saves only ~15% of the
  iterations at `p=0` and ~3% at `p=1` — but at `p=2` a uniform start *diverges*
  where this converges.
- **`p`-continuation** (off by default). Solve at `p=0` and re-project upward.
  Re-projection is exact, so it cannot change the answer — but it costs 2–28%
  *more* wall time than a direct solve, because the quasi-1D start has already
  removed the transient it exists to remove. Turn it on as a fallback: it
  converges `p=2` from a uniform initial condition, which a direct solve does
  not.

> Both of the last two were documented here as large speed-ups until they were
> actually measured. The iteration counts that seemed to support that were an
> artefact of two defects since fixed: convergence measured *relative to the
> first residual* (which tightens the target as the guess improves), and
> `p`-continuation reporting only its final stage's cost.

---

### Choking, as a check on the solver

![Mass flow across the choked range](docs/figures/choked_mass_flow.png)

Once the nozzle is choked, mass flow must not depend on back pressure at all.
Across the whole shock-free range the computed spread is **2.1e-8** — a check the
solver was never tuned to pass, and a good one to have students reproduce.

---

### Known limitation: shocked operating points do not converge

Between the **first** and **second** critical pressure ratios — a normal shock
standing in the diverging section — the pseudo-time march does **not** reach a
steady state. The residual falls by roughly an order of magnitude and then
parks:

| settings | residual floor (scaled) |
|---|---|
| `p=0`, `refine=0` | 6.1e-2 |
| `p=0`, `refine=1` | 4.9e-2 |
| `p=0`, `refine=2` | 5.6e-2 |
| `p=1`, `refine=0`, `barth-jespersen` + `ssprk3` | 1.5 |
| `p=1`, `refine=1`, `barth-jespersen` + `ssprk3` | 2.6 |

The `p=0` floor is **mesh-independent**, so this is not shock under-resolution.
The solver reports these runs as `converged=False` with a message, and does not
present the numbers as trustworthy — but it cannot currently produce a converged
shock-in-nozzle solution, and `p>=1` with the Barth–Jespersen limiter is worse
than `p=0` rather than better.

**What the residual is doing.** Localising it at `p=0`, `refine=1`,
`p_b/p_t = 0.70` (12,000 steps, no limiter):

| where | share of squared residual |
|---|---|
| `x ∈ [0.2, 0.3)` | **85.1%** |
| `x ∈ [0.3, 0.4)` | 8.3% |
| everywhere else | < 2% per band |
| outflow elements | **0.1%** |
| inflow elements | 0.0% |

This **rules out the boundary conditions.** The outflow plane carries a tenth of
a percent of the residual, and its normal Mach number is subsonic at every
quadrature point (max 0.75) with none near sonic — so the subsonic/supersonic
branch switch in the outflow condition is not chattering. The residual instead
concentrates in one narrow axial band, spread across the *full channel height*,
which is where the captured shock sits.

Two further observations point the same way. The DG shock settles near
`x ≈ 0.25` while quasi-1D theory puts it at `x = 0.508`; and the exit plane shows
**reverse flow** (minimum normal Mach −0.22), a recirculation that a
one-dimensional model cannot represent at all. A normal shock in a diverging
duct is not obviously a steady two-dimensional structure — in inviscid flow it
tends to bifurcate — so it is quite possible there is no steady solution here to
converge to. Settling that needs a time-accurate computation, which this solver
does not do.

**What still works:** everything shock-free — the design point, over-expanded
and under-expanded operation (`back_pressure_ratio` below the second critical
ratio), the whole area-ratio design space, and all sensitivity and optimisation
work. Those are the cases the solver is verified on.

**For shock physics**, use `dgnozzle.solve_quasi1d`, which solves the
one-dimensional problem exactly, including the shock position:

```python
from dgnozzle import solve_quasi1d, NozzleGeometry, FlowConditions
sol = solve_quasi1d(NozzleGeometry(contour="smooth"),
                    FlowConditions(back_pressure_ratio=0.70))
print(sol.summary())
# quasi-1D: shock-in-nozzle, shock x = 0.5084 m (M1 = 1.929), M_exit = 0.3270
```

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

## Relationship to the original MATLAB code

This is a rewrite, not a translation. The MATLAB sources are preserved in
[`matlab_legacy/`](matlab_legacy/) for reference. Defects found and fixed:

| Defect | Consequence |
|---|---|
| Diverging-contour rescale used constants evaluated at the wrong parameter values | The default contour reached an exit half-height of **4.34 m instead of 0.350 m**, and went **negative** just past the throat — inverted elements |
| Outflow boundary always extrapolated | `p_back_ratio` was dead; **no shocked operating point was reachable** |
| Inflow quadratic took the wrong root when the leading coefficient changed sign | Negative Mach number, meaningless boundary state |
| Throat height hard-coded in the thrust normalisation | Wrong `c_F` for any geometry but the original |
| Wall integral included the symmetry axis | Harmless for a horizontal axis, wrong in general |
| `extrapolate.m` read the new order as the old one, and the mesh changed between calls | `p`-continuation indexed past the end of its input |
| `postprocess()` took no arguments but was called with two; loaded `.mat` files never written | Post-processing could not run at all |
| Dunavant degree-3 rule (weight `−0.28125`) | Risked an indefinite mass matrix and a negative sum-of-squares |
| Unnormalised convergence test on the 4th RK stage; no iteration cap | "Converged" meant different things on different meshes; a diverging run never stopped |
| Negative Roe sound speed raised an error | A recoverable transient aborted the whole run |

Beyond the fixes: quadrilateral elements, curved (`Q=2`) geometry, a working
back-pressure boundary condition, positivity and slope limiters, quasi-1D
theory, adjoint sensitivities, sweeps, and a test suite.

---

## Licence

MIT. See [`LICENSE`](LICENSE).
