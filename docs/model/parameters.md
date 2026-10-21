# Parameters

These tables use the defaults in `src/config.py` and `src/geometry/contour.py`.

## NozzleGeometry

| Name in code | Meaning / symbol | Default | Units |
|---|---|---|---|
| `area_ratio` | Exit area / throat area, AR | `2.501888210773931` | dimensionless |
| `throat_x` | Throat position / length | `0.13880048` | dimensionless |
| `throat_half_height` | Throat half-height | `0.13989434` | reference length |
| `inlet_half_height` | Inlet half-height | `0.15` | reference length |
| `length` | Axial extent | `1.0` | reference length |
| `contour` | Wall family | `'bell'` | choice |
| `theta_initial_deg` | Wall angle just after the throat; None selects the family rule | `None` | degrees |
| `theta_exit_deg` | Exit wall angle | `0.0` | degrees |
| `bezier_w1` | First Bézier control weight | `0.55` | dimensionless |
| `bezier_w2` | Second Bézier control weight | `0.9` | dimensionless |

## FlowConditions

| Name in code | Meaning / symbol | Default | Units |
|---|---|---|---|
| `gamma` | Specific-heat ratio, γ | `1.4` | dimensionless |
| `Rgas` | Gas constant, R | `0.4` | reservoir reference scales |
| `total_temperature` | Reservoir temperature, T_t | `1.0` | reservoir reference scales |
| `total_pressure` | Reservoir pressure, p_t | `1.0` | reservoir reference scales |
| `back_pressure_ratio` | Exit back pressure / reservoir pressure | `0.15` | dimensionless |
| `inflow_angle` | Inlet velocity direction | `0.0` | radians |
| `entropy_fix` | Roe entropy-fix coefficient | `0.05` | dimensionless |
| `flux` | Interface Riemann solver | `'roe'` | choice |
| `hllc_low_mach` | HLLC low-Mach cutoff; zero disables correction | `0.0` | dimensionless |
| `backflow` | Enable reverse-flow exit branch | `False` | boolean |
| `allow_shock_in_nozzle` | Allow an otherwise refused internal-shock experiment | `False` | boolean |

## Discretization

| Name in code | Meaning / symbol | Default | Units |
|---|---|---|---|
| `element` | Triangle or quadrilateral | `'tri'` | choice |
| `order` | Solution polynomial degree, p | `1` | count |
| `geometry_order` | Geometry polynomial degree, Q | `1` | count |
| `refine` | Uniform mesh refinement level | `0` | count |
| `nx` | Base axial cell count | `14` | count |
| `nr` | Base transverse cell count | `5` | count |
| `x_spacing` | Axial node distribution | `'throat'` | choice |
| `cluster_strength` | Far/throat spacing ratio | `3.0` | dimensionless |
| `cluster_width` | Cluster width / length | `0.12` | dimensionless |

## SolverOptions

| Name in code | Meaning / symbol | Default | Units |
|---|---|---|---|
| `cfl` | Courant multiplier; None selects the measured recommendation | `None` | dimensionless |
| `tolerance` | Convergence threshold on the scaled residual | `1e-06` | dimensionless |
| `max_iterations` | Maximum pseudo-time steps | `200000` | count |
| `scheme` | Time integrator; nozzle None selects by polynomial degree | `None` | choice |
| `limiter` | Polynomial limiting strategy | `'positivity'` | choice |
| `tvb_constant` | Superbee smoothness threshold | `50.0` | dimensionless |
| `positivity_cfl` | Additional step multiplier while limiting; zero disables | `0.0` | dimensionless |
| `initial_condition` | Uniform or quasi-1D starting field | `'quasi1d'` | choice |
| `p_continuation` | Solve lower degrees before the target degree | `False` | boolean |
| `check_interval` | Steps between convergence checks | `50` | count |
| `print_interval` | Steps between progress lines; zero is silent | `500` | count |
| `divergence_factor` | Residual growth threshold for aborting | `10000.0` | dimensionless |
| `stall_window` | Number of convergence checks used to detect a stall | `40` | count |
| `stall_ratio` | Stall detection ratio | `0.98` | dimensionless |

## API and CLI names

Nozzle `theta_initial_deg` and `theta_exit_deg` are degrees. Adjoint design dictionaries instead use `theta_initial` and `theta_exit` in radians; see `NozzleGeometry.parameters` and `src/sensitivity.py`. The `analytic` contour fixes its shape and does not honor every design parameter.

The CLI exposes a subset of the Python configuration. [Command line](../running/command-line.md) records parser defaults separately; [configuration](../running/configuration.md) explains deck overrides. [Units](units.md) defines reference scales.
