#!/usr/bin/env bash
#
# dg2d.sh -- one entry point for the 2D DG nozzle code.
#
# The point of this script is that nothing has to be set up first.  It finds a
# Python, puts the repository on the import path if the package is not installed,
# and runs what you asked for from wherever you happen to be:
#
#     ./dg2d.sh run sweep
#     ./dg2d.sh solve area_ratio=3.0 back_pressure_ratio=0.12 order=1
#
# `./dg2d.sh help` lists everything.  `./dg2d.sh install` is optional.

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SELF="$(basename -- "${BASH_SOURCE[0]}")"

# --------------------------------------------------------------------------
# Python
# --------------------------------------------------------------------------
# Whatever environment you have loaded is the one used.  Set DG2D_PYTHON to
# override, e.g. DG2D_PYTHON=python3.12 ./dg2d.sh solve
find_python() {
    if [[ -n ${DG2D_PYTHON:-} ]]; then
        command -v "$DG2D_PYTHON" >/dev/null 2>&1 || die \
            "DG2D_PYTHON=$DG2D_PYTHON is not an executable on PATH"
        printf '%s\n' "$DG2D_PYTHON"
        return
    fi
    local candidate
    for candidate in python3 python; do
        if command -v "$candidate" >/dev/null 2>&1; then
            printf '%s\n' "$candidate"
            return
        fi
    done
    die "no python3 on PATH.  Load your environment first, or set DG2D_PYTHON."
}

die() { printf '%s: %s\n' "$SELF" "$*" >&2; exit 2; }

# An installed package wins; otherwise the repository root goes on the path, so
# the code runs straight out of a clone with no install step and from any working
# directory.  This is deliberately not an error: `install` is a convenience, not
# a prerequisite.  The package sits at the root, so `python -m src` already
# works when you are standing in the clone -- this is what makes it work when
# you are not.
setup_import_path() {
    if ! "$PY" -c 'import src' >/dev/null 2>&1; then
        export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
    fi
    if ! "$PY" -c 'import src' >/dev/null 2>&1; then
        printf '%s: cannot import the solver package.  ' "$SELF" >&2
        "$PY" -c 'import src' 2>&1 | tail -3 >&2
        printf '\nTry:  %s/%s install\n' "." "$SELF" >&2
        exit 2
    fi
}

# --------------------------------------------------------------------------
# Argument decks
# --------------------------------------------------------------------------
# `@name` is replaced by the `name=value` lines in that file, so a case you run
# often lives in a file instead of in your shell history:
#
#     ./dg2d.sh solve @design              # cases/design.dg
#     ./dg2d.sh solve @design pb=0.12      # the same deck, one value overridden
#     ./dg2d.sh sweep back_pressure_ratio 0.05 0.3 12 @design
#
# A deck holds exactly what you would have typed, so there is no second
# vocabulary to learn and no schema to keep in step with the code: `#` starts a
# comment, blank lines are ignored, and a deck may name another deck to include
# it.  Later arguments win, which is what makes the override above work.
#
# Because this is argument expansion and nothing more, a deck works with any
# subcommand that accepts the keys in it: `solve`, `sweep` and `bench` take the
# full set, while `geometry` takes only the geometry keys and will say
# `unrecognized arguments` if handed a deck carrying solver options.  That is
# the intended behaviour -- it names the keys it rejected rather than ignoring
# them, which is what a typo needs too.
DECK_DIR="$ROOT/cases"
DECK_EXT=".dg"
DECK_MAX_DEPTH=8

list_decks() {
    local path stem found=0
    for path in "$DECK_DIR"/*"$DECK_EXT"; do
        [[ -e $path ]] || continue
        found=1
        stem=$(basename "$path" "$DECK_EXT")
        printf '  @%-13s %s\n' "$stem" "$(deck_summary "$path")"
    done
    (( found )) || printf '  (none in %s)\n' "${DECK_DIR#"$ROOT"/}"
}

deck_summary() {
    # the summary is the first comment line, which is why decks start with one
    sed -n '1{/^[[:space:]]*#/{s/^[[:space:]]*#[[:space:]]*//;p;};q;}' "$1"
}

resolve_deck() {
    local want=$1 candidate
    for candidate in \
        "$want" "$want$DECK_EXT" \
        "$DECK_DIR/$want" "$DECK_DIR/$want$DECK_EXT"
    do
        if [[ -f $candidate ]]; then printf '%s\n' "$candidate"; return 0; fi
    done
    printf '%s: no deck named %s.  Available:\n' "$SELF" "$want" >&2
    list_decks >&2
    return 1
}

# Replace every `@deck` token with that deck's lines, recursively.
# EXPANDED is set as a side effect; bash functions cannot return arrays.
expand_decks() {
    local depth=$1; shift
    (( depth <= DECK_MAX_DEPTH )) || die \
        "decks nested more than $DECK_MAX_DEPTH deep -- does a deck include itself?"
    local out=() lines=() token path line
    for token in ${@+"$@"}; do
        if [[ $token != @* || $token == *::* ]]; then
            out+=("$token")
            continue
        fi
        path=$(resolve_deck "${token#@}") || exit 2
        lines=()
        while IFS= read -r line || [[ -n $line ]]; do
            line=${line%%#*}                        # drop comments
            line=${line#"${line%%[![:space:]]*}"}   # trim left
            line=${line%"${line##*[![:space:]]}"}   # trim right
            [[ -n $line ]] && lines+=("$line")
        done < "$path"
        expand_decks $((depth + 1)) ${lines[@]+"${lines[@]}"}
        out+=(${EXPANDED[@]+"${EXPANDED[@]}"})
    done
    EXPANDED=(${out[@]+"${out[@]}"})
}

# --------------------------------------------------------------------------
# Argument translation
# --------------------------------------------------------------------------
# `key=value` becomes `--key-with-dashes value`, so the names you type are the
# names the Python API uses -- one vocabulary to learn instead of two.  Real
# flags and bare positionals pass through untouched, so anything the underlying
# CLI accepts still works.
#
# TRANSLATED is set as a side effect; bash functions cannot return arrays.
translate() {
    expand_decks 1 ${@+"$@"}
    set -- ${EXPANDED[@]+"${EXPANDED[@]}"}
    local out=() token key value
    for token in ${@+"$@"}; do
        if [[ $token == -* || $token != *=* ]]; then
            out+=("$token")
            continue
        fi
        key=${token%%=*}
        value=${token#*=}
        case $key in
            p|order)                              key=order ;;
            Q|geometry_order|geometry-order)      key=geometry-order ;;
            ref|refine)                           key=refine ;;
            # NOT `M`: in a compressible-flow code M is the Mach number,
            # and the TVB constant is not a Mach number
            tvb|tvb_constant|tvb-constant)        key=tvb-constant ;;
            ar|area_ratio|area-ratio)             key=area-ratio ;;
            pb|back_pressure|back_pressure_ratio) key=back-pressure-ratio ;;
            *)                                    key=${key//_/-} ;;
        esac
        # only the two store_true options take a yes/no value; everything else
        # keeps its value verbatim, so `refine=0` is not mistaken for a flag
        case $key in
            quiet|p-continuation)
                case $value in
                    yes|true|on|1)  out+=("--$key"); continue ;;
                    no|false|off|0) continue ;;
                esac ;;
        esac
        out+=("--$key" "$value")
    done
    TRANSLATED=(${out[@]+"${out[@]}"})
}

# Run a dgnozzle subcommand with translated arguments.
dgnozzle() {
    local subcommand=$1; shift
    translate "$@"
    exec "$PY" -m src "$subcommand" ${TRANSLATED[@]+"${TRANSLATED[@]}"}
}

# --------------------------------------------------------------------------
# Cases
# --------------------------------------------------------------------------
# The runnable scripts in examples/.  Numbered for reading order; `run` matches
# on the descriptive part, so `run sweep` and `run 02` both work.
list_cases() {
    local path stem
    for path in "$ROOT"/examples/[0-9][0-9]_*.py; do
        [[ -e $path ]] || continue
        stem=$(basename "$path" .py)
        printf '  %-14s %s\n' "${stem#*_}" "$(case_summary "$path")"
    done
}

case_summary() {
    # the one-line summary is the first docstring line, after the em dash
    sed -n '2{s/^"""//;s/ *$//;s/\.$//;s/.*--- *//;p;q;}' "$1"
}

resolve_case() {
    local want=$1 matches=()
    # exact path, exact stem, then the descriptive suffix or the number
    if [[ -f $want ]]; then printf '%s\n' "$want"; return; fi
    if [[ -f "$ROOT/examples/$want.py" ]]; then
        printf '%s\n' "$ROOT/examples/$want.py"; return
    fi
    local path stem
    for path in "$ROOT"/examples/[0-9][0-9]_*.py; do
        [[ -e $path ]] || continue
        stem=$(basename "$path" .py)
        if [[ ${stem#*_} == "$want" || ${stem%%_*} == "$want" ]]; then
            matches+=("$path")
        fi
    done
    if (( ${#matches[@]} == 1 )); then
        printf '%s\n' "${matches[0]}"
        return
    fi
    if (( ${#matches[@]} > 1 )); then
        die "'$want' matches more than one case"
    fi
    printf '%s: no case named %s.  Available:\n' "$SELF" "$want" >&2
    list_cases >&2
    exit 2
}

# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------
cmd_help() {
    cat <<EOF
$SELF -- 2D discontinuous Galerkin nozzle code

Usage: ./$SELF <command> [arguments]

Getting started
  install            install the package and its optional extras with pip
  check              report the Python, the version, and which backends work
  list               list the cases 'run' can execute and the decks '@' expands

Running
  run <case> [...]   run one of the case scripts in examples/
  solve [...]        solve one operating point
  sweep <var> <lo> <hi> <n> [...]
                     sweep one design variable
  geometry [...]     inspect a contour; no flow solve
  bench [...]        time the backends against each other
  test [...]         run the test suite
  verify [--fast]    everything the project checks about itself: tests, lint,
                     format and spelling.  '--fast' skips the full solves.
  docs               print where the documentation is

Arguments are 'name=value', using the same names as the Python API:

  ./$SELF solve area_ratio=3.0 back_pressure_ratio=0.12 order=1
  ./$SELF solve ar=3.0 pb=0.12 p=1 figure=result.png     # short aliases
  ./$SELF sweep area_ratio 2.0 4.0 9 order=1 csv=sweep.csv
  ./$SELF geometry contour=bezier bezier_w1=0.7
  ./$SELF run sensitivity

A '@name' argument is replaced by the lines of cases/name.dg, so a case you run
often lives in a file rather than in your shell history.  Later arguments win,
so a deck is a starting point you can override:

  ./$SELF solve @design                 # exactly the lines in cases/design.dg
  ./$SELF solve @design pb=0.12         # the deck, with one value changed
  ./$SELF sweep back_pressure_ratio 0.05 0.3 12 @design
  ./$SELF bench @converged
  ./$SELF geometry @contour              # a geometry-only deck

Decks are the same 'name=value' lines you would have typed, '#' starts a
comment, and a deck may name another deck to build on it.  A deck works with
any subcommand that accepts the keys it holds -- 'solve', 'sweep' and 'bench'
take the full set, 'geometry' only the geometry ones.

Aliases: p=order, Q=geometry_order, ref=refine, ar=area_ratio,
pb=back_pressure_ratio, tvb=tvb_constant.  Ordinary --flags work too, so
anything 'python -m src --help' documents is still available.
'M' is deliberately NOT an alias: in this code M always means Mach number.

Cases ('run <name>'):
$(list_cases)

Decks ('@name'):
$(list_decks)

Environment
  DG2D_PYTHON        the interpreter to use (default: python3, then python)

No install is required: if the solver package is not importable, the repository root is
put on PYTHONPATH automatically.  Documentation is in docs/ -- start with
docs/theory.md for the formulation and docs/lab_guide.md for the exercises.
EOF
}

cmd_install() {
    local extras=${1:-all}
    printf '%s: installing with pip into %s\n' "$SELF" "$("$PY" -c 'import sys; print(sys.prefix)')"
    "$PY" -m pip install -e "$ROOT[$extras]"
    printf '\n%s: done.  Checking it worked:\n\n' "$SELF"
    cmd_check
}

cmd_check() {
    "$PY" - <<'PYEOF'
import importlib.util
import sys
from pathlib import Path

print(f"python   {sys.version.split()[0]}  ({sys.executable})")

spec = importlib.util.find_spec("src")
if spec is None:
    print("solver package NOT IMPORTABLE")
    raise SystemExit(2)

import src
from src.backends import available_backends

where = Path(src.__file__).parent
print(f"dgnozzle {src.__version__}  ({where})")
print(f"backends {', '.join(available_backends())}")

for name, what in (("numba", "the fast solver"),
                   ("jax", "gradients and sensitivity"),
                   ("matplotlib", "figures"),
                   ("scipy", "optimisation")):
    ok = importlib.util.find_spec(name) is not None
    print(f"  {'yes' if ok else 'NO '}  {name:12s} {what}")
PYEOF
}

cmd_run() {
    (( $# >= 1 )) || die "run needs a case name.  Try: ./$SELF list"
    local case_path
    case_path=$(resolve_case "$1") || exit 2
    shift
    printf '%s: running %s\n\n' "$SELF" "${case_path#"$ROOT"/}"
    exec "$PY" "$case_path" "$@"
}

cmd_test() {
    translate "$@"
    # Only default to the whole tree when no path was given -- appending $ROOT
    # unconditionally made `dg2d.sh test tests/test_docs.py` run everything,
    # which looks like the filter being ignored because it is.
    local target=("$ROOT")
    local arg
    for arg in ${TRANSLATED[@]+"${TRANSLATED[@]}"}; do
        if [[ -e $arg || $arg == *::* ]]; then
            target=()
            break
        fi
    done
    exec "$PY" -m pytest ${target[@]+"${target[@]}"} ${TRANSLATED[@]+"${TRANSLATED[@]}"}
}

# Everything the project checks about itself, in one command.
#
# This used to be a GitHub Actions workflow across six OS/Python combinations.
# It is a teaching code with one author, so a matrix proving it also works on
# Windows py3.10 was buying nothing, and a push waiting on a queue is a worse
# feedback loop than a terminal.  The checks themselves were worth keeping, so
# they moved here, where they run on the machine the code is actually used on
# and the author decides when.
#
# Each step is skipped with a note rather than failing if its tool is missing,
# because a missing linter should not look like a broken solver.  The exit code
# is non-zero if any step that *did* run failed.
cmd_verify() {
    local fast=0
    [[ ${1:-} == --fast ]] && fast=1

    local failed=()
    local skipped=()

    step() {  # step <name> <tool-to-probe> <command...>
        local name=$1 probe=$2; shift 2
        if [[ -n $probe ]] && ! "$PY" -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('$probe') else 1)"; then
            printf '\n==> %s -- SKIPPED (%s is not installed)\n' "$name" "$probe"
            skipped+=("$name")
            return 0
        fi
        printf '\n==> %s\n' "$name"
        if "$@"; then
            return 0
        fi
        failed+=("$name")
    }

    step "tests (fast)" pytest "$PY" -m pytest "$ROOT" -q --fast
    if (( ! fast )); then
        # 40-odd end-to-end solves, including finite-difference gradient checks.
        # Minutes, not seconds -- which is why --fast exists.
        step "tests (full solves and adjoint checks)" pytest \
            "$PY" -m pytest "$ROOT" -q --slow --durations=10
    fi
    step "lint" ruff "$PY" -m ruff check "$ROOT"
    step "format" ruff "$PY" -m ruff format --check "$ROOT"
    step "spelling" codespell_lib "$PY" -m codespell_lib \
        "$ROOT/src" "$ROOT/tests" "$ROOT/examples" "$ROOT/docs" "$ROOT/README.md"

    printf '\n%s\n' "----------------------------------------------------------"
    if (( ${#skipped[@]} )); then
        printf 'skipped: %s\n' "${skipped[*]}"
    fi
    if (( ${#failed[@]} )); then
        printf 'FAILED:  %s\n' "${failed[*]}"
        return 1
    fi
    printf 'all checks passed\n'
}

cmd_docs() {
    cat <<EOF
docs/theory.md      the formulation, the geometry definition and the
                    verification evidence -- equations in LaTeX, renders on
                    GitHub with nothing to build
docs/lab_guide.md   the exercises, and what to report
docs/tikz/          TikZ sources for every figure
README.md           the quick start and the API tour
EOF
}

# --------------------------------------------------------------------------
main() {
    local command=${1:-help}
    if (( $# > 0 )); then shift; fi

    case $command in
        help|-h|--help|'') cmd_help; return 0 ;;
    esac

    PY=$(find_python) || exit 2   # die() runs in a subshell; check here too

    case $command in
        install)   cmd_install ${1+"$1"} ;;
        check)     setup_import_path; cmd_check ;;
        list)      printf "Cases ('run <name>'):\n"; list_cases
                   printf "\nDecks ('@name'):\n"; list_decks ;;
        docs)      cmd_docs ;;
        run)       setup_import_path; cmd_run "$@" ;;
        test)      setup_import_path; cmd_test "$@" ;;
        verify)    setup_import_path; cmd_verify ${1+"$1"} ;;
        solve|sweep|geometry|bench)
                   setup_import_path; dgnozzle "$command" "$@" ;;
        *)         printf '%s: unknown command %s\n\n' "$SELF" "$command" >&2
                   cmd_help >&2
                   exit 2 ;;
    esac
}

main "$@"
