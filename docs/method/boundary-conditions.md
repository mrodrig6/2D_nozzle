# Boundary conditions

Boundary conditions impose only the information carried into the domain. This page describes the wall and characteristic branches implemented by the solver.

## Boundary conditions

The number of conditions that may be imposed equals the number of *incoming*
characteristics, read off the flux-Jacobian eigenvalues with $\mathbf{n}$ the
outward normal.

### Inviscid wall and symmetry axis

Both are impermeable: $\mathbf{v}\cdot\mathbf{n} = 0$. Removing the normal
velocity and recovering the pressure from the remaining energy gives a flux
that transmits pressure but no mass or energy,

```math
p_b = (\gamma - 1)\left(\rho E
      - \tfrac{1}{2}\rho\,|\mathbf{v}_{t}|^{2}\right),
\qquad
\hat{\mathbf{F}} = \bigl(0,\; p_b n_x,\; p_b n_y,\; 0\bigr)^{T}.
```

The contoured wall and the symmetry axis use the same flux but carry different
tags, so that thrust can be integrated over the wall alone.

### Subsonic stagnation inflow

Three characteristics enter, one leaves. The outgoing one carries the interior
Riemann invariant

```math
J^{+} = v_n + \frac{2a}{\gamma - 1}.
```

Combining it with the isentropic stagnation relations and a prescribed flow
direction $\hat{\mathbf{d}} = (\cos\alpha, \sin\alpha)$ gives a quadratic for
the inflow Mach number,

```math
\underbrace{\left(\tfrac{\gamma-1}{2}\beta - n_d^{2}\right)}_{A} M^{2}
\underbrace{-\,\tfrac{4 n_d}{\gamma-1}}_{B} M
+ \underbrace{\left(\beta
  - \left(\tfrac{2}{\gamma-1}\right)^{2}\right)}_{C}
= 0,
\qquad
\beta = \left(\frac{J^{+}}{a_t}\right)^{2},\quad
n_d = \mathbf{n}\cdot\hat{\mathbf{d}}.
```

The *smallest non-negative* root is the physical branch; the boundary state then
follows from the isentropic relations and total enthalpy
$H_b = a_t^{2}/(\gamma-1)$.

> **Root selection.** Taking $(-B + \sqrt{B^{2}-4AC})/2A$ unconditionally is
> correct only while $A > 0$. When the leading coefficient changes sign — which
> happens for weak inflow — that root is negative and the boundary state becomes
> meaningless. The smallest non-negative root is the physical branch in every
> case.

### Outflow

If $v_n/a \ge 1$ the outflow is supersonic: no characteristic enters, nothing
may be imposed, and the interior state is extrapolated. If it is subsonic,
exactly one condition is admissible. Static pressure is set to $p_b$, while
entropy and the outgoing invariant are carried from the interior and the
tangential velocity is unchanged:

```math
\rho_b = \rho^{+}\left(\frac{p_b}{p^{+}}\right)^{1/\gamma},
\qquad
v_{n,b} = v_n^{+} + \frac{2}{\gamma-1}\left(a^{+} - a_b\right),
\qquad
\mathbf{v}_{t,b} = \mathbf{v}_t^{+} .
```

The two branches are selected by the local normal Mach number, so a shock
crossing the exit plane during the transient is handled without special cases.

> **Both branches are needed.** Extrapolating unconditionally would leave the
> back pressure with no effect on the flow at all, and with it every operating
> point whose exit is subsonic — which is to say every shocked one.
