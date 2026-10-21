# Command line

The Bash launcher expands decks and `name=value` arguments. The Python CLI uses ordinary flags; its parser is implemented in `src/cli.py`.

## Commands

| Python subcommand | Purpose |
|---|---|
| `solve` | Solve one point |
| `sweep` | Sweep one parameter |
| `geometry` | Inspect geometry |
| `bench` | Measure backend throughput |

| Launcher-only command | Purpose |
|---|---|
| `install [extra]` | Editable pip installation; defaults to `all` |
| `list` | List examples and case decks |
| `run <name-or-path>` | Run one example or user script |
| `test [pytest arguments]` | Run existing tests |
| `verify [--fast]` | Tests, lint, format and spelling |
| `docs` | Print documentation entry points |
| `check` | Report interpreter and optional tools; this is launcher-only |

## Flags and positional arguments

Defaults below are parser defaults, which can differ from dataclass defaults or deck values. See [configuration precedence](configuration.md) and [parameters](../model/parameters.md).

| Flag / argument | Commands accepting it | Meaning / choices | Parser default |
|---|---|---|---|
| `--contour` | `solve`, `sweep`, `geometry`, `bench` | contour family (default: bell) | `'bell'` |
| `--area-ratio` | `solve`, `sweep`, `geometry`, `bench` | exit/throat area ratio | `2.5019` |
| `--throat-x` | `solve`, `sweep`, `geometry`, `bench` | throat location, 0-1 | `0.1388` |
| `--theta-initial-deg` | `solve`, `sweep`, `geometry`, `bench` | wall angle at throat | `None` |
| `--theta-exit-deg` | `solve`, `sweep`, `geometry`, `bench` | wall angle at exit | `0.0` |
| `--bezier-w1` | `solve`, `sweep`, `geometry`, `bench` | First Bézier control weight | `0.55` |
| `--bezier-w2` | `solve`, `sweep`, `geometry`, `bench` | Second Bézier control weight | `0.9` |
| `--back-pressure-ratio` | `solve`, `sweep`, `geometry`, `bench` | p_back / p_total | `0.15` |
| `--gamma` | `solve`, `sweep`, `geometry`, `bench` | Specific-heat ratio, γ | `1.4` |
| `--element` | `solve`, `sweep`, `bench` |  Choices: `tri`, `quad`. | `'tri'` |
| `--flux` | `solve`, `sweep`, `bench` | interface flux: roe (default, most accurate), hllc (positivity-preserving), ausmup (AUSM+-up, no wave decomposition) Choices: `roe`, `hllc`, `ausmup`. | `'roe'` |
| `-p`, `--order` | `solve`, `sweep`, `bench` | polynomial order p | `1` |
| `-Q`, `--geometry-order` | `solve`, `sweep`, `bench` | geometry order Q | `1` |
| `-r`, `--refine` | `solve`, `sweep`, `bench` | refinement level | `0` |
| `--x-spacing` | `solve`, `sweep`, `bench` |  Choices: `throat`, `inlet`, `uniform`. | `'throat'` |
| `--backend` | `solve`, `sweep`, `bench` |  Choices: `numba`, `numpy`, `jax`. | `'numba'` |
| `--cfl` | `solve`, `sweep`, `bench` | Courant number: cfl/(2p+1) times the geometric step. The default is 70%% of the largest value measured to converge at this order | `None` |
| `--tolerance` | `solve`, `sweep`, `bench` | Convergence threshold on the scaled residual | `1e-06` |
| `--max-iterations` | `solve`, `sweep`, `bench` | Maximum pseudo-time steps | `200000` |
| `--scheme` | `solve`, `sweep`, `bench` | time scheme; omit for the recommended one (euler at p=0, ssprk3 above) Choices: `euler`, `ssprk3`. | `None` |
| `--limiter` | `solve`, `sweep`, `bench` |  Choices: `none`, `positivity`, `superbee`. | `'positivity'` |
| `--positivity-cfl` | `solve`, `sweep`, `bench` | extra time-step cap while a limiter is active (0 disables) | `0.0` |
| `--allow-shock-in-nozzle` | `solve`, `sweep`, `bench` | run an operating point whose shock sits inside the diverging section (it will not converge) | `False` |
| `--backflow` | `solve`, `sweep`, `bench` | treat reverse flow at the exit with its own branch | `False` |
| `--tvb-constant` | `solve`, `sweep`, `bench` | Cockburn-Shu TVB threshold M for limiter=superbee | `50.0` |
| `--p-continuation` | `solve`, `sweep`, `bench` | solve at p=0 and re-project upward; a fallback for a high-order solve that will not start, not a speed-up | `False` |
| `-q`, `--quiet` | `solve`, `sweep`, `bench` | quiet | `False` |
| `--figure` | `solve`, `sweep`, `geometry` | write an overview figure to this path | `None` |
| `--save` | `solve` | write the solution to this .npz path | `None` |
| `parameter` (positional) | `sweep` | parameter to sweep, e.g. area_ratio | `None` |
| `start` (positional) | `sweep` | start | `None` |
| `stop` (positional) | `sweep` | stop | `None` |
| `count` (positional) | `sweep` | count | `None` |
| `--metrics` | `sweep` | metrics | `['thrust_coefficient', 'exit_mach', 'entropy_error']` |
| `--csv` | `sweep` | write the full table here | `None` |
| `--steps` | `bench` | steps | `200` |

`-h` / `--help` works on every parser. Do not pass a solve deck to a geometry-only command merely because the deck names the same geometry.

## Launcher spelling

Underscores in `name=value` become hyphens in Python flags. Aliases are `p=order`, `Q=geometry_order`, `ref=refine`, `ar=area_ratio`, `pb=back_pressure_ratio`, and `tvb=tvb_constant`. `M` is not an alias.

Boolean flags without value-taking parser actions should be passed as literal flags. For example: `./dg2d.sh solve @shocked --allow-shock-in-nozzle`; `allow_shock_in_nozzle=True` is a Python keyword, not a supported launcher boolean translation.

`DG2D_PYTHON` selects the launcher's interpreter. Otherwise it looks for `python3`, then `python`. The installed `dgnozzle` command and `python -m src` use the Python subcommands above; launcher-only commands are not interchangeable with them.
