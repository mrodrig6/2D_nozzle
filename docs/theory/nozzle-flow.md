# Quasi-one-dimensional nozzle flow

Quasi-one-dimensional theory gives a reference nozzle solution and initial field. Its area–Mach and shock relations separate the operating regimes.

## Quasi-one-dimensional reference solution

Quasi-1D theory serves two purposes: it is an exact reference to compare
against, and it is the initial condition. As a starting field it saves a modest
number of iterations (about 15% at $p = 0$, 3% at $p = 1$ on the reference
case); its real value is robustness, since at $p = 2$ without $p$-continuation a
uniform start diverges where the quasi-1D start converges.

The area–Mach relation is

```math
\frac{A}{A^{*}} = \frac{1}{M}
\left[\frac{2}{\gamma+1}\left(1 + \frac{\gamma-1}{2}M^{2}\right)
\right]^{\frac{\gamma+1}{2(\gamma-1)}},
```

monotone on each of the subsonic and supersonic branches and therefore
invertible by bracketed bisection. The isentropic relations give

```math
\frac{p}{p_t} = \left(1 + \tfrac{\gamma-1}{2}M^{2}\right)^{-\frac{\gamma}{\gamma-1}},
\qquad
\frac{\rho}{\rho_t} = \left(1 + \tfrac{\gamma-1}{2}M^{2}\right)^{-\frac{1}{\gamma-1}},
\qquad
\frac{T}{T_t} = \left(1 + \tfrac{\gamma-1}{2}M^{2}\right)^{-1}.
```

Across a normal shock,

```math
M_2^{2} = \frac{1 + \frac{\gamma-1}{2}M_1^{2}}
               {\gamma M_1^{2} - \frac{\gamma-1}{2}},
\qquad
\frac{p_2}{p_1} = \frac{2\gamma M_1^{2} - (\gamma-1)}{\gamma+1},
\qquad
\frac{A_2^{*}}{A_1^{*}} = \frac{p_{t1}}{p_{t2}} .
```

### Operating regimes

Three critical back-pressure ratios divide the operating map:

| Ratio | Condition | Regime below it |
|---|---|---|
| $(p_b/p_t)_1$ | $M = 1$ at the throat, subsonic exit | choked, shock in the diverging section |
| $(p_b/p_t)_2$ | normal shock at the exit plane | shock-free in the nozzle, over-expanded |
| $(p_b/p_t)_3$ | shock-free supersonic (design) | under-expanded |

Above $(p_b/p_t)_1$ the nozzle is not choked and the mass flow still depends on
back pressure. For the reference geometry ($\mathrm{AR} = 2.50189$,
$\gamma = 1.4$) the three ratios are $0.9609$, $0.4345$ and $0.06390$, with a
design exit Mach number of $2.4436$.

When a shock stands in the diverging section its position is found by bisection:
$p_{\mathrm{exit}}$ falls monotonically as the shock moves downstream, so there
is exactly one position at which it equals $p_b$.
