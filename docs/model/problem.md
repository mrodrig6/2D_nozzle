# Problem and assumptions

The solver computes steady, inviscid, two-dimensional flow of a calorically perfect gas. This page states the domain, assumptions and limits.

## Scope and conventions

The solver computes steady, inviscid, compressible flow through a planar
converging–diverging nozzle. *Planar* means the channel is two-dimensional and
of unit depth, not axisymmetric. One consequence matters throughout: the
one-dimensional *area* is the channel height, so

```math
A(x) = 2\,y_{\mathrm{wall}}(x),
\qquad
\frac{A_{\mathrm{exit}}}{A_{\mathrm{throat}}}
= \frac{y_{\mathrm{wall}}(L)}{y_{\mathrm{throat}}}
\equiv \mathrm{AR}.
```

Isentropic tables may be used directly, because the area–Mach relation depends
on $A/A^{*}$ and not on the cross-section's shape.

Only the upper half of the channel is meshed; the lower half follows by symmetry
about $y = 0$. All reported *integral* quantities (thrust, mass flow) are
doubled to describe the full channel per unit depth.

### Notation

Vectors and tensors are set bold. Latin symbols take upright bold,
$\mathbf{U}$, $\mathbf{F}$, $\mathbf{n}$; Greek symbols take bold italic,
$\boldsymbol{\psi}$, because `\mathbf` has no effect on a Greek glyph. Velocity
is the vector $\mathbf{v}$, and its components are the scalars $v_x$ and $v_y$:

```math
\mathbf{v} = (v_x,\ v_y)^{T},
\qquad
v_n = \mathbf{v}\cdot\mathbf{n} = v_x n_x + v_y n_y,
\qquad
\mathbf{v}_t = \mathbf{v} - v_n \mathbf{n}.
```

The code follows the same convention: `vx`, `vy`, `vn`, `vt`. A subscript $n$ on
a flux means the same projection, $\mathbf{F}_n = \mathbf{F} n_x + \mathbf{G} n_y$.

Everything bold here is Latin or Greek, deliberately. A bold calligraphic
letter for the flux pair would read well in print, but
`\boldsymbol{\mathcal{F}}` renders *unbold* on GitHub: KaTeX accepts it, finds
no bold calligraphic glyph in its fonts, and silently drops the weight — so the
projected flux is written out instead.

## Reference scales

See [units and reference scales](units.md) for the canonical nondimensional definitions.

## Assumptions and constraints

Every result from this code is conditional on the following. None of them is a
bug or a gap to be filled later; they are the modelling choices that define what
the solver *is*, and reading a result without them is the main way to misuse it.

**Physical assumptions**

| | Assumption | Consequence |
|---|---|---|
| 1 | **Inviscid.** The Euler equations — no viscosity, no boundary layer, no heat conduction | No skin friction, no separation, no viscous losses. Despite the course title, Reynolds number does not appear anywhere: the model is the high-Re *limit*, not a high-Re flow |
| 2 | **Calorically perfect gas**, constant $\gamma = 1.4$ | No dissociation or vibrational excitation; not valid for the very hot exhaust of a real rocket |
| 3 | **Adiabatic**, no body forces | Total enthalpy is conserved, and is the check used to verify the solver |
| 4 | **Steady** | Marched in pseudo-time to a fixed point. Unsteady phenomena — buzz, screech, transient start-up — are outside the model, not merely unresolved |
| 5 | **Two-dimensional planar**, unit depth | A channel, not a body of revolution. Area is the channel height, so isentropic tables apply directly, but thrust is per unit depth and an axisymmetric nozzle is a *different* problem |
| 6 | **Symmetric about the axis** | Only the upper half is meshed. Asymmetric modes cannot be represented, so they can neither be found nor ruled out |

**Numerical and operational constraints**

| | Constraint | Consequence |
|---|---|---|
| 7 | **No shock inside the diverging section.** Back-pressure ratios between the second and first critical values are *refused* | The march does not reach a steady state there, so rather than return numbers that look like an answer, `solve_nozzle` raises and names the band. `allow_shock_in_nozzle=True` opts back in and reports `converged=False` |
| 8 | **Shocks outside the exit are not computed**, only evaluated in closed form | `src/external/` gives the lip wave and the shock-cell pattern from the exit state. It is not a plume solver: the barrel shock and Mach disc need the external region in the mesh |
| 9 | **Non-dimensional throughout** | No SI anywhere. $p_t = T_t = L = 1$ and $R = \gamma - 1$, giving $\rho_t = 2.5$ and $a_t = 0.7483$. Compare runs with `thrust_coefficient` and `discharge_coefficient`, not raw `thrust` |
| 10 | **Gradients are of the discrete problem** | Which is what optimisation needs — but use a shock-free point: a limiter switching on and off introduces kinks |

Constraint 7 is not a restriction on nozzle *design* — it is most of the point
of it. Sizing a nozzle means avoiding a shock in the diverging section, and the
interesting waves form outside the exit plane, which is what `src/external/`
draws.

→ `docs/theory.md` and
`docs/usage.md`

---

See [compatibility](../reference/compatibility.md) for implementation restrictions.
