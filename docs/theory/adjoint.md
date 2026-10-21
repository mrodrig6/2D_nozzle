# Discrete adjoint

The discrete adjoint differentiates the converged residual equation. It avoids differentiating the full pseudo-time history.

## Design sensitivities

For a scalar functional $J$ and design variables $\mathbf{a}$, the converged
state satisfies $\mathbf{R}(\mathbf{U}, \mathbf{a}) = \mathbf{0}$.
Differentiating implicitly,

```math
\frac{\mathrm{d}J}{\mathrm{d}\mathbf{a}}
= \frac{\partial J}{\partial \mathbf{a}}
  - \frac{\partial J}{\partial \mathbf{U}}
    \left(\frac{\partial \mathbf{R}}{\partial \mathbf{U}}\right)^{-1}
    \frac{\partial \mathbf{R}}{\partial \mathbf{a}}
= \frac{\partial J}{\partial \mathbf{a}}
  - \boldsymbol{\psi}^{T}\frac{\partial \mathbf{R}}{\partial \mathbf{a}},
```

where the adjoint variable solves

```math
\left(\frac{\partial \mathbf{R}}{\partial \mathbf{U}}\right)^{T}
\boldsymbol{\psi}
= \left(\frac{\partial J}{\partial \mathbf{U}}\right)^{T}.
```

The cost is one adjoint solve *regardless of the number of design variables* —
where finite differences would need two flow solves each.

The adjoint system is solved by pseudo-time march,

```math
\frac{\mathrm{d}\boldsymbol{\psi}}{\mathrm{d}\tau}
= -\mathbf{M}^{-1}\left(
    \left(\partial \mathbf{R}/\partial \mathbf{U}\right)^{T}\boldsymbol{\psi}
    - \left(\partial J/\partial \mathbf{U}\right)^{T}\right),
```

mirroring the forward solver. Because $\mathbf{M}$ is symmetric positive
definite,
$\mathbf{M}^{-1}(\partial\mathbf{R}/\partial\mathbf{U})^{T}$ is similar to
$\mathbf{M}^{-1}\partial\mathbf{R}/\partial\mathbf{U}$ and has the same
spectrum, so the forward solver's time step and CFL limit carry over unchanged
and no preconditioner is needed. Both Jacobian actions come from automatic
differentiation of the residual; no matrix is ever assembled.

**Mesh consistency.** The gradient is taken on a *fixed logical grid*. This
matters: with throat-following node clustering, rebuilding the mesh at a
perturbed $x_{\mathrm{th}}$ also redistributes the nodes, so a finite difference
would measure node motion as well as the design change. Freezing the logical
grid removes that inconsistency.

**Verification.** See the captured [gradient comparison](../validation/gradients.md).

**Validity.** The gradient is of the *discrete* problem, which is what
gradient-based optimisation needs. Two caveats: the geometry must be smooth in
$\mathbf{a}$ (all contour families here are), and the flow should be shock-free
with the limiter inactive — a limiter switching on and off introduces kinks, and
a shock's position is only piecewise differentiable in the design variables.
