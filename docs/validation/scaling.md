# Reference-scale invariance

These tests rescale the reference quantities and compare converged nozzle solutions. The weighted residual is part of the invariance being checked.

## Is it actually non-dimensional? Measured, not asserted

Invariance is the property being claimed, so it is tested rather than stated.
`tests/test_nondimensional.py` rescales the references and compares converged
solves:

| Rescaling | Dimensionless outputs | Dimensional outputs | Iterations |
|---|---|---|---|
| every length $\times 2$, $\times 5$, $\times \tfrac14$ | **exactly 0 drift** | scale by the factor to 1e-10 | identical |
| $p_t \times 10$ | exact, 1e-16 | scale by 10 | identical |
| $T_t \times 4$ | 8e-16 — round-off | — | identical |
| $R \times 2.5$ | 3e-14 — round-off | — | identical |

All four are exact, and the iteration counts match. The last two were not always
so, and the reason is worth keeping, because it is the kind of error a
non-dimensionalisation invites.

**The convergence criterion used to be mixed-dimension.** The conserved
variables scale as $\rho_t$, $\rho_t a_t$ and $\rho_t a_t^2$ — three different
powers of $a_t$ — but the residual was a single RMS norm over all four
components, divided by the single scale $\rho_t a_t / L$. No one scale can
non-dimensionalise a mixed-dimension norm, so changing $a_t$ (via $T_t$ or $R$)
reweighted the components against each other and the march crossed the
tolerance in a different place: ~7e-11 of drift at `tolerance=1e-10`, and 50
iterations' difference.

The solution it converged to never moved — the field at a fixed iteration was
invariant to the last bit — so this only ever changed *where the march stopped*.
It is fixed regardless: each component is divided by its own power of $a_t$
before the sum, so every term is dimensionless against the same reference. See
`component_weights` in `src/backends/base.py`.

Scaling $p_t$ was immune to this all along, because it multiplies all four
components by the *same* factor.

**To put results in physical units**, multiply by your own reference values:
thrust by $p_t L$ (times the depth), mass flow by $\rho_t a_t L$, and so on.
Nothing in the solver needs to change — the non-dimensional solution is the
same for every reservoir condition at a given $\gamma$ and back-pressure ratio,
which is exactly why it is solved this way.
