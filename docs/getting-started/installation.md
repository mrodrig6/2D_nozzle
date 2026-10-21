# Installation

Install the solver from its checkout in an isolated Python environment. Python 3.10 is the minimum; dependencies select versions compatible with the interpreter.

## Create an environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[all]"
./dg2d.sh check
```

Run these commands from this repository's root. On Windows, activate with `.venv\Scripts\Activate.ps1` and run launcher commands through Bash, or use `dgnozzle` for its Python subcommands. Windows execution has not been validated in this change.

The companion project also installs a package named `src`. Give each checkout its own environment; do not install both distributions together or add both roots to `PYTHONPATH`.

## Dependency groups

| Extra | Purpose |
|---|---|
| No extra | NumPy and SciPy for the reference solver |
| `fast` | Numba acceleration |
| `jax` | JAX differentiation |
| `plot` | Matplotlib figures |
| `all` | Solver, gradients and plotting |
| `dev` | pytest, Ruff and codespell plus coverage |
| `docs` | MkDocs, Material and Markdown extensions plus PyMuPDF for TikZ rasterization |

For contributions and site work:

```bash
python -m pip install -e ".[all,dev,docs]"
./dg2d.sh verify --fast
mkdocs build --strict
```

`./dg2d.sh install` installs `[all]`, not the developer tools. `verify` reports a skipped tool if it is absent; the CI jobs install their required extras explicitly.

## Runtime and optional tools

Run CLI and Python recipes from this root, with the environment active. Use `MPLBACKEND=Agg` for headless plotting. Numba compiles on first use, then caches kernels; benchmark times exclude compilation unless stated.

No server, database, GPU, account or secret is required. Scientific tests run locally. Rebuilding TikZ images additionally needs TeX Live; see the figure inventory.

See [quick start](quick-start.md), [compatibility](../reference/compatibility.md), and [figure rebuilding](../reference/figures.md).
