# Continuous integration

Five workflows check changes on pull requests and pushes to `main`. The documentation workflow publishes only after a successful default-branch build.

| Workflow | Checks | Same check locally |
|---|---|---|
| [Tests](../../.github/workflows/tests.yml) | Full suite on Python 3.10 and newest stable (`3.x`), with NumPy, Numba and JAX installed | `python -m pytest -o addopts='' -q` |
| [Lint](../../.github/workflows/lint.yml) | Ruff diagnostics | `python -m ruff check .` |
| [Formatting](../../.github/workflows/format.yml) | Ruff formatting | `python -m ruff format --check .` |
| [Spelling](../../.github/workflows/spelling.yml) | codespell | `python -m codespell_lib src tests docs README.md examples` |
| [Documentation](../../.github/workflows/docs.yml) | Recursive GitHub links and strict MkDocs build; deploy from main only | `python docs/check_links.py` then `mkdocs build --strict` |

Install `[all,dev,docs]` before running every check. `./dg2d.sh verify` remains a local convenience; it skips missing tools, while individual CI jobs install their tools explicitly.

The tests job requires its optional backends to import before pytest. It fails visibly if a compatible numerical dependency is unavailable for the newest Python; it does not silently convert missing backend coverage into a pass. Version-matrix runs on GitHub have not been claimed by a local Python 3.12 run.

The build job has read-only repository access. Only the default-branch deployment job receives Pages and identity-token write permissions. Pull requests build an artifact but never deploy it. A manual run also deploys when the selected branch is `main`.

## Enable Pages and recover a failed deployment

A repository administrator must open [Settings → Pages](https://github.com/mrodrig6/2D_nozzle/settings/pages), choose **GitHub Actions** under **Build and deployment → Source**, and save the setting if prompted. Do this once for each repository before publishing.

If the deployment reports `404 Not Found` or “Ensure GitHub Pages has been enabled,” check that setting first. The documentation build and scientific tests can pass while publication fails because Pages is disabled. The deployment now checks the Pages configuration before uploading the site to the Pages service. The workflow's default token cannot enable Pages; no additional token is needed when an administrator enables it in Settings.

After enabling Pages, open **Actions → Documentation** and rerun the failed deployment. After the manual trigger is merged, you can also select **Run workflow**, choose `main`, and run it to rebuild and publish without a new commit. Deployment errors remain failures and must be resolved before publication succeeds.

The configured site is `https://mrodrig6.github.io/2D_nozzle/`. Its badge is marked pending until the first successful deployment. A local build, an open PR and a live Pages site are separate results.
