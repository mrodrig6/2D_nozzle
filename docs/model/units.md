# Units and reference scales

The solver uses nondimensional values throughout. These reference scales define the meaning of its inputs and reported coefficients.

## Units: the solver is non-dimensional

**Nothing in this code is in SI.** There are no metres, kelvin or newtons
anywhere in the solve, and no unit conversion is applied on the way out. The
equations are solved in a non-dimensional form fixed by four choices:

| Reference | Symbol | Value | Set by |
|---|---|---|---|
| Reservoir pressure | $p_t$ | `1` | `total_pressure` |
| Reservoir temperature | $T_t$ | `1` | `total_temperature` |
| Gas constant | $R$ | `0.4` | `Rgas`, chosen as $\gamma - 1$ |
| Length | $L$ | `1` | **every length together** — see below |

> **`length` is not the length scale.** It sets the axial extent while
> `throat_half_height` and `inlet_half_height` set the transverse one, so
> changing `length` alone makes the nozzle more *slender* — a different shape,
> entitled to a different answer. Changing all three together is the rescale,
> and that is exactly invariant.

Everything else follows, and these two are worth knowing because they are
**not** 1:

```math
\rho_t = \frac{p_t}{R T_t} = 2.5, \qquad
a_t = \sqrt{\gamma R T_t} = 0.7483
```

Taking $R = \gamma - 1$ is the choice that makes the rest tidy: static
temperature is then numerically equal to the specific internal energy,
$T = p/(\rho R) = e$.

**Read every output as a ratio.** The reported quantities are already
non-dimensional, or are reported alongside a non-dimensional form:

| Output | Non-dimensionalised by | Note |
|---|---|---|
| `thrust_coefficient` | $p_t A^{*}$ | the headline number |
| `discharge_coefficient` | $\rho_t a_t A^{*}$ | choked flow fixes this at 0.5787 for $\gamma=1.4$ |
| `exit_pressure_ratio` | $p_t$ | |
| `exit_temperature_ratio` | $T_t$ | |
| `specific_thrust_ratio` | $a_t$ | effective exhaust velocity as a Mach number |
| `thrust_efficiency`, `entropy_error` | — | already dimensionless |
| `exit_mach_area_averaged` | — | already dimensionless |
| `thrust`, `mass_flow_in/out`, `specific_thrust` | — | **carry the scales above**: $p_t L$, $\rho_t a_t L$ and $a_t$ respectively |

So `thrust = 0.049868` is a force per unit depth in units of $p_t L$; to compare
two nozzles, use `thrust_coefficient` instead. `C_d` is the better mass-flow
number for the same reason — once the throat is sonic it depends on $\gamma$
alone, so it is the same for every choked run regardless of reservoir and
throat size. Measured 0.5756 against the closed form

```math
\frac{\dot m}{\rho_t a_t A^{*}}
  = \left(\frac{2}{\gamma+1}\right)^{\frac{\gamma+1}{2(\gamma-1)}} = 0.5787
```

a 0.5% gap that is discretisation error, and is pinned by a test.
