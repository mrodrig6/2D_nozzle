# Repository map

This table describes the top-level files and folders. Generated study output should not replace committed scientific figures.

| Path | Purpose |
|---|---|
| [`src`](../../src) | Solver modules and public Python interface |
| [`tests`](../../tests) | Existing scientific and interface assertions |
| [`docs`](../../docs) | Topic pages, site hook, figures and figure helpers |
| [`examples`](../../examples) | Worked study scripts |
| [`cases`](../../cases) | Launcher argument decks |
| [`nozzle.sh`](../../nozzle.sh) | Bash launcher |
| [`pyproject.toml`](../../pyproject.toml) | Package metadata and dependency extras |
| [`README.md`](../../README.md) | Short front door and source of the website home |
| [`LICENSE`](../../LICENSE) | MIT license |
| [`.gitignore`](../../.gitignore) | Generated-file exclusions |
| [`mkdocs.yml`](../../mkdocs.yml) | Site navigation and strict validation |
| [`.github`](../../.github) | Tests, quality checks and documentation validation workflows |

The [code map](code-map.md) describes individual modules. The [figure inventory](figures.md) gives rebuild commands. `site/`, Python caches and local virtual environments are generated and ignored.

## Documentation migration

| Previous document | Canonical destinations |
|---|---|
| `README.md` | [problem](../model/problem.md) · [quick-start](../getting-started/quick-start.md) · [bibliography](bibliography.md) |
| `docs/README.md` | [figures](figures.md) |
| `examples/README.md` | [examples](../getting-started/examples.md) |

Scientific derivations, lab exercises, numerical tables and research retractions were retained. Repeated introductions, manual tables of contents, duplicated parameter/gradient tables and obsolete present-tense status text were replaced by links or rewritten. Old documentation paths remain as pointer pages. No numerical implementation or existing tests were changed.

## Remaining limitations

- The worked sweep records stalled AR=4.25 and AR=4.5 points at refine=1; nine of its eleven points converge. The study reports those failures and still exits zero.
- Geometry reports still append `m` to nondimensional lengths (`NozzleGeometry.describe`). This is an application-output defect; values remain nondimensional.
- `@shocked` does not add its own shock opt-in. Use the explicit CLI flag for an experiment, as documented in configuration.
- Historic benchmark timings are hardware-specific; not every old image has a committed, byte-for-byte reproduction script.
