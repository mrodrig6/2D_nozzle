# Backends and performance

How the solver is made fast, what was measured, and the time step.

---

## Backends and performance

Same equations, three execution strategies:

| `backend` | Role |
|---|---|
| `'numba'` | compiled nopython kernels — **the default and the fastest** |
| `'numpy'` | vectorised reference; no compile step, easiest to read |
| `'jax'` | the same vectorised source under `jax.numpy`; **differentiable** |

```bash
./dg2d.sh bench p=1 refine=1
```

They agree to a relative 1e-10, which the test suite enforces. Measured on one
RK4 step, `p=1`:

| elements | `numba` | `numpy` | ratio |
|---|---|---|---|
| 140 | 0.17 ms | 3.8 ms | 22× |
| 560 | 0.34 ms | 8.9 ms | 26× |
| 2240 | 0.83 ms | 34 ms | 41× |

Only the Numba path carries the fused stages and the preallocated working set;
`numpy` is the readable reference and is not meant to be fast.

### Where the time goes

One RK4 step at `p=1, refine=1` (560 elements, 1680 DOF), by component. The step
measured 0.34 ms in this run; repeat runs on the same machine spread about ±10%,
so read the shares rather than the absolute times:

| Component | Per step | Share |
|---|---|---|
| rate: edge pass + element pass + inverse mass, × 4 | 0.22 ms | 65% |
| positivity limiter × 4 (screened out, nothing limited) | 0.029 ms | 8% |
| stage arithmetic (3 axpy + the final combination) | 0.021 ms | 6% |
| residual norm + local time step | 0.008 ms | 2% |
| allocation and dispatch | — | ~18% |

Thread scaling is still the weak point at the sizes used for design work. All
four Numba kernels are `prange`-parallel, but at 140 elements there are only 35
elements per thread per parallel region, so the launch overhead swamps the work.
Throughput, on the other hand, now runs from 10 M unknown-updates/s at the
smallest size to 32 M at the largest.

### What made it faster

Three changes, measured in two groups. Running at `cfl=1.0` — the old default,
and still the same time step under the same definition — separates the groups:
the iteration count then matches the old table to the digit, so whatever wall
time has moved is the kernel work and the rest is the larger default step. The
two kernel changes landed together and are not separated from each other here.

| `p` | `refine` | before | kernels only | time step as well |
|---|---|---|---|---|
| 0 | 0 | 851 it, 0.19 s | 2.30× | **2.50×** |
| 1 | 0 | 1751 it, 0.56 s | 2.10× | **3.42×** |
| 2 | 0 | 3301 it, 1.64 s | 1.73× | **3.57×** |
| 1 | 1 | 2501 it, 1.64 s | 2.16× | **3.83×** |
| 2 | 1 | 6301 it, 6.96 s | 1.79× | **3.03×** |
| 1 | 2 | 6051 it, 9.68 s | 1.71× | **3.40×** |

**The kernels (1.7–2.3×).** Two things, both of which were measured as small and
turned out not to be:

- *Everything is preallocated and the stages are fused.* A four-stage step
  evaluated the residual four times, and each evaluation allocated and zeroed an
  edge flux table and a residual array; the stage arithmetic (`U + 0.5*dt*F0` and
  friends) allocated a full state array six more times, and the residual norm
  allocated one more for `A*A`. The inverse mass solve was a separate parallel
  pass over the residual, so every step wrote `R` to memory and read it straight
  back. All of that is gone: the backend owns its working set, the mass solve is
  folded into the element pass, and the stage arithmetic is kernels.
- *The limiter's inactive path is screened.* On a smooth solution nothing
  violates positivity, but proving it still swept every probe point. The basis is
  a partition of unity, so a cell's mean plus `Λ·max|U_i − Ū|` bounds every probe
  value, with `Λ = max_x Σ|φ_i(x)|` a constant of the element. That costs one pass
  over the `nbf` coefficients instead of `nbf` times the probe count, and it is
  sufficient — when it fails, the exact probe still runs, so the limiter's output
  is unchanged. The inactive limiter went from 14% of a step to 8%.

**The time step (a further 1.1–2.1×).** See below.

### The time step

`cfl` means what it always means — a multiplier on the order-dependent stable
step:

```
dt_e = cfl / (2p + 1) * 2 A_e / sum_f s_f l_f
```

with `1/(2p+1)` the standard restriction for explicit DG. What changed is the
*default*, because that restriction is a bound and measurement says it is much
tighter than it needs to be above `p=0`. Bisecting the largest `cfl` at which
the march still converges, over the `bell` and `smooth` contours at refinement
levels 0 and 1 (the two agreeing to within a bisection step at every order):

| `p` | RK4 | SSP-RK3 | `cfl=1` as a fraction of the RK4 limit |
|---|---|---|---|
| 0 | 1.625 | 1.437 | 62% |
| 1 | 2.625 | 2.337 | 38% |
| 2 | 2.500 | 2.240 | 40% |

So one fixed `cfl` carries a different safety margin at every order, and the old
default of `1.0` left most of a factor of two unused above `p=0`. **The default
is now order-dependent** — 70% of the measured limit, so the *margin* is a
constant 30% while the definition stays conventional:

| `p` | default `cfl`, RK4 | default `cfl`, SSP-RK3 |
|---|---|---|
| 0 | 1.14 | 1.01 |
| 1 | 1.84 | 1.64 |
| 2 | 1.75 | 1.57 |

Pass `cfl` explicitly and it means exactly what the formula says — nothing
rescales it:

```bash
./dg2d.sh solve p=1 cfl=2.2          # 84% of the measured limit, your call
./dg2d.sh solve p=1 cfl=1.0          # the old default, if you want to compare
```

Read as a coefficient on the geometric step rather than as a `cfl` number, the
stable value falls as `1 : 0.54 : 0.31` across `p = 0, 1, 2`, which `1/(p+1)`
fits (`1 : 0.5 : 0.33`) and `1/(2p+1)` does not (`1 : 0.33 : 0.2`). That is the
scaling used beyond the measured orders.

> **A measurement, not a proof.** Two contours at one back pressure is a scan,
> not a stability analysis, which is why the default keeps 30% in hand rather
> than sitting on the limit. The test suite marches 400 steps at each tabulated
> limit, so a change that moves the real limit fails the build instead of
> quietly invalidating the table.

### What makes it fast, and what makes it robust

None of these change the answer — and the distinction is measured, not assumed.

**Faster:**

- **Vectorised, gather-only assembly.** No Python loop over elements or edges,
  and no scatter-add — so the loops parallelise without atomics and the same
  source runs under all three backends.
- **Fused stages and a preallocated working set**, and **a screened limiter
  fast path** — both above, together 1.7–2.3×.
- **A calibrated time step** — above, a further 1.1–2.1×.

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

> **On measuring these honestly.** Three things in this section were documented
> as large speed-ups before anyone measured them — the quasi-1D start,
> `p`-continuation and warm-started sweeps — and all three were corrected. The
> two that *did* pay, the kernel work and the time step, were estimated at
> 25–40% and 60–90% and came in at 1.7–2.3× and 1.1–2.1×. The estimates were
> wrong in both directions, which is the argument for measuring rather than for
> estimating better.

---

### Choking, as a check on the solver

![Mass flow across the choked range](figures/choked_mass_flow.png)

Once the nozzle is choked, mass flow must not depend on back pressure at all.
Across the whole shock-free range the computed spread is **2.1e-8** — a check the
solver was never tuned to pass, and a good one to have students reproduce.

---

### Known limitation: shocked operating points

Between the **first** and **second** critical pressure ratios a normal shock
stands in the diverging section, and the pseudo-time march generally does not
reach a steady state there: `p=0` stalls (and sometimes converges), `p>=1`
diverges. The solver reports those runs as `converged=False` and does not
present the numbers as trustworthy. Use `p=0` and check `converged`, or
`solve_quasi1d` for the shock physics itself.

Everything shock-free is verified and converges: the design point,
over-expanded and under-expanded operation, the whole area-ratio design space,
and all sensitivity and optimisation work.

**The solver tells you before the run.** `shock_free_range(area_ratio)` returns
the two usable intervals, `is_shock_free(area_ratio, pb)` classifies a point,
and `solve_nozzle` warns at setup if the operating point puts a shock in the
diverging section — which saves finding out minutes later. For `AR=2.5` the
usable set is `p_b/p_t < 0.4348` (choked, supersonic exit) or `>= 0.9608`
(unchoked). That covers what a nozzle-design exercise wants: the over-expanded
and under-expanded wave structure forms **outside** the exit plane, downstream
of the computed domain.
