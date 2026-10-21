# Limiters

The nozzle limiters reduce polynomial deviations while preserving an admissible cell average. They cannot turn every shocked operating point into a converged solve.

## Limiting

At $p = 0$ the solution is constant per element and cannot oscillate. At
$p \ge 1$ it can, and near a shock it does: the polynomial overshoots, pressure
or density goes negative at a quadrature point, and the Roe flux fails.

Both limiters act by shrinking the deviation from the cell average,

```math
\mathbf{U}_{e,i}^{\mathrm{lim}} = \bar{\mathbf{U}}_e
  + \theta_e \left(\mathbf{U}_{e,i} - \bar{\mathbf{U}}_e\right),
\qquad \theta_e \in [0,1],
```

which leaves the cell average untouched and is therefore *conservative*.

**Positivity (Zhang–Shu).** $\theta_e$ is the largest value keeping $\rho$ and
$p$ above small floors at every point the scheme evaluates — the volume
quadrature points and the element's own face quadrature points. The density
constraint is linear in $\theta$ and solved in closed form; the pressure
constraint is quadratic and bracketed by bisection. On a smooth solution
$\theta_e = 1$ everywhere and the limiter is exactly inactive, so full accuracy
is retained.

**Superbee (Roe).** A TVD slope limiter: it additionally bounds the element's
own increment toward each neighbour by what the jump in cell averages across
that face admits, damping the oscillations themselves rather than only their
consequences. Writing $d_f$ for the element's increment toward face $f$ and
$a_f$ for the jump in cell averages across it,

```math
\theta_e = \min_{f,\,s} \min\!\left(1,\
  \frac{\mathrm{superbee}(a_{f,s},\, d_{f,s})}{d_{f,s}}\right),
\qquad
\mathrm{superbee}(a, b) = \mathrm{maxmod}\bigl(
  \mathrm{minmod}(2a, b),\ \mathrm{minmod}(a, 2b)\bigr),
```

equivalently $\phi(r) = \max\bigl(0, \min(2r, 1), \min(r, 2)\bigr)$ with
$r = a/b$. Of the second-order TVD limiters Superbee is the most
*compressive* — it sits on the upper edge of Sweby's TVD region, where `minmod`
sits on the lower one, permitting twice the neighbour jump where `minmod`
permits one. That keeps a captured shock sharp, and it also makes Superbee the
most willing of the family to switch on.

Three details decide whether it is usable.

*One scalar factor, not one per component.* The minimum runs over the four
conserved components $s$ as well as the faces, so a single number scales the
whole element. That is a matter of admissibility, not tidiness. The set
$\{\rho > 0,\ p > 0\}$ is **convex**, so
$\bar{\mathbf{U}}_e + \theta(\mathbf{U}_{e,i} - \bar{\mathbf{U}}_e)$ is a convex
combination of two admissible states and is itself admissible for every
$\theta \in [0,1]$. A per-component factor leaves that segment — it pairs a
strongly limited density with a barely limited energy — and can land outside the
set even though both endpoints are inside it. Measured on the shocked case at
`cfl=0.3`, a per-component factor drove the *cell average* non-physical 10,246
times and the march to `NaN`; the scalar factor needed no repairs at all and
held the minimum pressure positive throughout.


*Boundary faces are not limited against.* An element is its own neighbour across
a domain boundary, so the jump there is identically zero; limiting against it
would drive $\theta_e$ to zero in exactly the elements carrying the wall and the
exit plane, which is where thrust is integrated.

*The TVB threshold.* Cockburn and Shu call this constant $M$. **This document does not**, because $M$ is the Mach number everywhere else in it and in the code; it is written $K_{\mathrm{TVB}}$ here and `tvb_constant` in the API. A face whose increment is small enough to be smooth data
rather than an oscillation is left alone, following Cockburn and Shu. Their
threshold is $K_{\mathrm{TVB}} h^2$, with $K_{\mathrm{TVB}}$ a bound on a second derivative; this code uses the
dimensionless equivalent

```math
|d_{f,s}| \le K_{\mathrm{TVB}}\, \frac{A_e}{A_\Omega}\, \max_{e'} |\bar{u}_{e',s}|
\quad\Longrightarrow\quad
\text{component } s \text{ is not limited across } f ,
```

so that one `tvb_constant` $= K_{\mathrm{TVB}}$ covers any geometry, any non-dimensionalisation
and any refinement level. The element's share of the domain area stands in for
$(h/L)^2$, so the threshold still shrinks as $h^2$ under refinement, which is
what makes the limiter vanish in the limit and recovers the TVB argument. The
state scale is the largest cell average of that component *anywhere in the
domain*, not the element's own — the transverse momentum passes through zero on
the nozzle axis, and a local scale would therefore drive the threshold to zero on
precisely the elements where the field is smoothest.

The threshold is not optional. Without it a TVD limiter stays marginally active
at every smooth extremum, clipping it on some iterations and not others, and a
limiter that switches on and off between iterations parks the residual at a fixed
level instead of converging. Measured at the shock-free design point, $K_{\mathrm{TVB}} = 0$
leaves 77 of 140 elements of an *already converged* field being clipped and the
residual parks at $3.1$ instead of reaching $10^{-6}$. This is the same failure
mode that made the Barth–Jespersen limiter this replaced *worse than no slope
limiter at all*, by more than an order of magnitude on the shocked case.

The shipped default is $K_{\mathrm{TVB}} = 50$, and it is the smallest value that works.
Scanned over the design, over-expanded and under-expanded points, at
$(p, \texttt{refine})$ of $(1,0)$, $(1,1)$ and $(2,0)$: $K_{\mathrm{TVB}} = 10$ fails all nine,
$K_{\mathrm{TVB}} = 20$ fails the three at $\texttt{refine}=1$, and $K_{\mathrm{TVB}} = 50$ converges all nine.

It is worth being honest about what that costs, because it is the central
difficulty of this whole section. At $K_{\mathrm{TVB}} = 50$ the slope limiter is nearly
inactive *even at a shock*: 200 steps into a shocked run it touches 0 of 140
elements at $p_b/p_t = 0.50$ and 2 of 140 at $0.70$, where $K_{\mathrm{TVB}} = 0$ touches 72 and
92. For this problem the window in which one TVB constant both leaves a smooth
steady solution alone and still bites on a shock is **empty**. Lower $K_{\mathrm{TVB}}$ and the
limiter chatters and the residual parks; raise it and the limiter stops acting.
That is a property of asking a TVD limiter — designed for a time-accurate march,
where a little chatter is harmless — to coexist with a *steady* pseudo-time march
whose whole purpose is to drive a residual to zero. It is the reason shocked
operating points do not converge at $p \ge 1$, and it is not something a
different choice of TVD limiter function would fix. A shock-capturing method that
does converge in steady state — artificial viscosity with a smooth sensor, or a
subcell WENO reconstruction — would be a change of method, not a change of
limiter.

**When a limiter cannot help.** If the *cell average* itself becomes
non-physical, no scaling can repair it — at $\theta = 0$ the solution *is* that
average. The code then floors the average minimally, counts the repair, and
reports it; a run needing repairs should be rerun with a smaller `cfl`.
Separately, a limiter that switches on and off between iterations parks the
residual at a fixed level instead of converging. The solver detects that limit
cycle, stops, and says so.
