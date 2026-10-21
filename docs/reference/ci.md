# Continuous integration

Five workflows check changes on pull requests and pushes to `main`. The documentation workflow checks links, builds the site and saves a downloadable preview.

| Workflow | Checks | Same check locally |
|---|---|---|
| [Tests](../../.github/workflows/tests.yml) | Full suite on Python 3.10 and newest stable (`3.x`), with NumPy, Numba and JAX installed | `python -m pytest -o addopts='' -q` |
| [Lint](../../.github/workflows/lint.yml) | Ruff diagnostics | `python -m ruff check .` |
| [Formatting](../../.github/workflows/format.yml) | Ruff formatting | `python -m ruff format --check .` |
| [Spelling](../../.github/workflows/spelling.yml) | codespell | `python -m codespell_lib src tests docs README.md examples` |
| [Documentation](../../.github/workflows/docs.yml) | Recursive GitHub links, strict MkDocs build and preview artifact | `python docs/check_links.py` then `mkdocs build --strict` |

Install `[all,dev,docs]` before running every check. `./dg2d.sh verify` remains a local convenience; it skips missing tools, while individual CI jobs install their tools explicitly.

The tests job requires its optional backends to import before pytest. It fails visibly if a compatible numerical dependency is unavailable for the newest Python; it does not silently convert missing backend coverage into a pass. Version-matrix runs on GitHub have not been claimed by a local Python 3.12 run.

The documentation workflow has read-only repository access and runs on pull requests, pushes to `main` and manual requests. It uploads the built `site/` directory as a regular Actions artifact named `documentation-preview`, retained for 14 days. It requires no hosting configuration or deployment permissions.

## Preview documentation

Open a successful **Actions → Documentation** run and download **documentation-preview** from its artifacts. Extract the archive, open a terminal in the directory containing `index.html`, and run:

```bash
python -m http.server 8000
```

Open `http://localhost:8000` to browse the preview. To preview directly from a checkout with the documentation dependencies installed, run `mkdocs serve`. The Markdown documentation can also be read directly on GitHub.
