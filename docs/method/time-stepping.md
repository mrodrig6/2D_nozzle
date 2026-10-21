# Time stepping and convergence

The solver marches in pseudo-time to a steady state with local steps. The scheme, convergence norm and stability restrictions below determine when it stops.

## Time integration

The semi-discrete system is marched in pseudo-time to steady state with an
element-local step,

```math
\Delta t_e
= \frac{\mathrm{CFL}}{2p+1}\,
  \frac{2 A_e}{\displaystyle\sum_{f \in \partial\Omega_e} s_f\, \ell_f},
```

where $A_e$ is the element area, $\ell_f$ the length of face $f$ and $s_f$ the
maximum signal speed there. The $1/(2p+1)$ factor is the standard
order-dependent restriction for explicit DG, and $\mathrm{CFL}$ means what it
usually does: a multiplier on that stable step.

### What CFL is safe

The textbook restriction is a bound, not an equality, and measurement says it is
much tighter than it needs to be above $p = 0$. Bisecting the largest
$\mathrm{CFL}$ at which the march still converges — over the `bell` and
`smooth` contours at refinement levels 0 and 1, the two agreeing to within a
bisection step at every order — gives

| $p$ | SSP-RK3 | Euler | $\mathrm{CFL} = 1$ as a fraction of the SSP-RK3 limit |
|---|---|---|---|
| 0 | 1.437 | 1.169 | 70% |
| 1 | 2.337 | 1.337 | 43% |
| 2 | 2.240 | 0.775 | 45% |

The two columns were not measured over the same set of cases: SSP-RK3 over
four contour/refinement combinations, Euler over `smooth` at refinement 0
only. The 30% margin below covers the difference — across the SSP-RK3 cases
the most permissive contour sat 6–10% above the tightest.

So a single fixed $\mathrm{CFL}$ carries a different safety margin at every
order. The solver's default is therefore order-dependent — 70% of the measured
limit, so the *margin* is constant at 30% while the definition stays
conventional. Pass $\mathrm{CFL}$ explicitly and it means exactly what the
formula above says.

Read as a coefficient on the geometric step rather than as a $\mathrm{CFL}$
number, SSP-RK3's stable value falls as $1 : 0.54 : 0.31$ across
$p = 0, 1, 2$, which $1/(p+1)$ fits ($1 : 0.5 : 0.33$) and $1/(2p+1)$ does not
($1 : 0.33 : 0.2$). That is the scaling used beyond the measured orders.

**Euler does not follow that rule**, and the table above is the reason to say
so explicitly: its limit *falls* with order, to $0.775$ at $p = 2$ — below
where it started. So beyond $p = 2$ it is held flat rather than extrapolated
upward, which the growth rule would otherwise do in exactly the regime where
it is already weakest. Euler is the cheap scheme per step and the expensive
one per unit of pseudo-time at high order.

> **A measurement, not a proof.** Two contours at one back pressure is a scan,
> not a stability analysis, which is why the default keeps 30% in hand rather
> than sitting on the limit. `tests/test_timestep.py` marches 400 steps at each
> tabulated limit, so a change to the flux, the quadrature or the limiter that
> moves the real limit fails the suite instead of silently invalidating the
> table.

Two schemes are available. The default is Euler at `p=0` and SSP-RK3 above it, selected by `src.timestepping.step.recommended_scheme`. `'ssprk3'` is the three-stage
strong-stability-preserving method, with the local step frozen at the first
stage — standard for a pseudo-time march, since the intermediate stages are
not meant to be time-accurate, and also what makes every stage a convex
combination of forward-Euler steps with the *same* $\Delta t$:

```math
\begin{aligned}
  \mathbf{U}^{(1)} &= \mathbf{U}^{n}
    + \Delta t\, \mathbf{L}(\mathbf{U}^{n}),\\
  \mathbf{U}^{(2)} &= \tfrac{3}{4}\mathbf{U}^{n} + \tfrac{1}{4}
    \left(\mathbf{U}^{(1)}
      + \Delta t\, \mathbf{L}(\mathbf{U}^{(1)})\right),\\
  \mathbf{U}^{n+1} &= \tfrac{1}{3}\mathbf{U}^{n} + \tfrac{2}{3}
    \left(\mathbf{U}^{(2)}
      + \Delta t\, \mathbf{L}(\mathbf{U}^{(2)})\right),
\end{aligned}
\qquad
\mathbf{L} = -\mathbf{M}^{-1}\mathbf{R}.
```

Each SSP stage is a convex combination of forward-Euler updates, which is
the stage structure required by positivity arguments. A complete guarantee also requires a positivity-preserving first-order flux and the corresponding CFL bound. The default Roe flux and calibrated stability step do not establish that guarantee.

`'euler'` is the single-stage method, $\mathbf{U}^{n+1} = \mathbf{U}^{n} +
\Delta t\,\mathbf{L}(\mathbf{U}^{n})$: one residual evaluation per step
instead of three, a smaller stable step, and — being one forward-Euler stage —
the same positivity argument applies to it. At $p = 0$ it is exactly the
scheme the companion finite-volume airfoil code marches with, which is what
makes the two codes comparable at the bottom of the order range.

Classical `'rk4'` was removed. It cost a fourth residual evaluation per step,
bought no time accuracy in a march whose step is already element-local, and
had no strong-stability property to justify the expense.

### Convergence measure

The march stops when

```math
\frac{\left\|\dot{\mathbf{U}}\right\|_{\mathrm{rms}}}{\rho_t a_t / L}
< \texttt{tolerance},
\qquad
\left\|\dot{\mathbf{U}}\right\|_{\mathrm{rms}}
= \sqrt{\frac{1}{4N}\sum_{e,i,k} \left(w_k(\mathbf{M}^{-1}\mathbf{R})_{e,i,k}\right)^{2}} .
```

The choice of denominator matters more than it looks. Testing the
*unnormalised* residual against a fixed threshold makes "converged" mean
different things on different meshes and at different operating points. But
testing it *relative to the first iteration* is no better: it demands a tighter
absolute residual the better the initial guess is, so improving the initial
condition makes the solver appear slower and two runs started differently cannot
be compared at all. A fixed physical scale, $\rho_t a_t / L$, is independent of
both the mesh and the starting field.

One further trap: the residual must be measured at a state the solver actually
holds, not at an intermediate Runge–Kutta stage, whose rate is not the residual
anywhere.

Here `N` counts scalar solution nodes and `w = (1, 1/a_t, 1/a_t, 1/a_t²)`. `src.backends.base.component_weights` applies these weights before the RMS. Each weighted rate has the same reservoir scale; the unweighted norm would mix different powers of sound speed.
