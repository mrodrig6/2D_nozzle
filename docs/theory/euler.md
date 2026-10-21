# Euler equations

The compressible Euler equations describe the inviscid gas model. Their conservative flux and characteristic speeds determine the numerical and boundary fluxes.

## Governing equations

The two-dimensional Euler equations in conservative form are

```math
\frac{\partial \mathbf{U}}{\partial t}
+ \frac{\partial \mathbf{F}}{\partial x}
+ \frac{\partial \mathbf{G}}{\partial y} = \mathbf{0},
```

with

```math
\mathbf{U} =
\begin{pmatrix} \rho \\ \rho v_x \\ \rho v_y \\ \rho E \end{pmatrix},
\qquad
\mathbf{F} =
\begin{pmatrix} \rho v_x \\ \rho v_x^{2} + p \\ \rho v_x v_y \\ \rho v_x H \end{pmatrix},
\qquad
\mathbf{G} =
\begin{pmatrix} \rho v_y \\ \rho v_x v_y \\ \rho v_y^{2} + p \\ \rho v_y H \end{pmatrix}.
```

The system is closed by the ideal-gas relation and the definition of total
enthalpy,

```math
p = (\gamma - 1)\left(\rho E - \tfrac{1}{2}\rho\,|\mathbf{v}|^{2}\right),
\qquad
H = E + \frac{p}{\rho},
\qquad
a = \sqrt{\frac{\gamma p}{\rho}} .
```

It is convenient to write the flux projected onto a direction
$\mathbf{n} = (n_x, n_y)$,

```math
\mathbf{F}_n(\mathbf{U})
\;\equiv\; \mathbf{F} n_x + \mathbf{G} n_y
=
\begin{pmatrix}
  \rho v_n \\ \rho v_x v_n + p\,n_x \\ \rho v_y v_n + p\,n_y \\ \rho H v_n
\end{pmatrix},
```

since every boundary and interface term below is of this form. The flux
Jacobian $\partial\mathbf{F}_n/\partial\mathbf{U}$ has eigenvalues

```math
\lambda = \{\,v_n + a,\; v_n - a,\; v_n,\; v_n\,\},
```

whose signs determine how many conditions may be imposed at a boundary
([Boundary conditions](../method/boundary-conditions.md#boundary-conditions)).
