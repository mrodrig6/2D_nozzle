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
# a prerequisite.  The package sits at the root, so `python -m dgnozzle` already
# works when you are standing in the clone -- this is what makes it work when
# you are not.
setup_import_path() {
    if ! "$PY" -c 'import dgnozzle' >/dev/null 2>&1; then
        export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
    fi
    if ! "$PY" -c 'import dgnozzle' >/dev/null 2>&1; then
        printf '%s: cannot import dgnozzle.  ' "$SELF" >&2
        "$PY" -c 'import dgnozzle' 2>&1 | tail -3 >&2
        printf '\nTry:  %s/%s install\n' "." "$SELF" >&2
        exit 2
    fi
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
    local out=() token key value
    for token in "$@"; do
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
            mg|multigrid)                         key=multigrid ;;
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
    exec "$PY" -m dgnozzle "$subcommand" ${TRANSLATED[@]+"${TRANSLATED[@]}"}
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
  list               list the cases 'run' can execute

Running
  run <case> [...]   run one of the case scripts in examples/
  solve [...]        solve one operating point
  sweep <var> <lo> <hi> <n> [...]
                     sweep one design variable
  geometry [...]     inspect a contour; no flow solve
  bench [...]        time the backends against each other
  test [...]         run the test suite
  docs               print where the documentation is

Arguments are 'name=value', using the same names as the Python API:

  ./$SELF solve area_ratio=3.0 back_pressure_ratio=0.12 order=1
  ./$SELF solve ar=3.0 pb=0.12 p=1 figure=result.png     # short aliases
  ./$SELF sweep area_ratio 2.0 4.0 9 order=1 csv=sweep.csv
  ./$SELF geometry contour=bezier bezier_w1=0.7
  ./$SELF run sensitivity

Aliases: p=order, Q=geometry_order, ref=refine, ar=area_ratio,
pb=back_pressure_ratio, mg=multigrid.  Ordinary --flags work too, so anything
'python -m dgnozzle --help' documents is still available.

Cases:
$(list_cases)

Environment
  DG2D_PYTHON        the interpreter to use (default: python3, then python)

No install is required: if dgnozzle is not importable, the repository root is
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

spec = importlib.util.find_spec("dgnozzle")
if spec is None:
    print("dgnozzle NOT IMPORTABLE")
    raise SystemExit(2)

import dgnozzle
from dgnozzle.backends import available_backends

where = Path(dgnozzle.__file__).parent
print(f"dgnozzle {dgnozzle.__version__}  ({where})")
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
        list)      list_cases ;;
        docs)      cmd_docs ;;
        run)       setup_import_path; cmd_run "$@" ;;
        test)      setup_import_path; cmd_test "$@" ;;
        solve|sweep|geometry|bench)
                   setup_import_path; dgnozzle "$command" "$@" ;;
        *)         printf '%s: unknown command %s\n\n' "$SELF" "$command" >&2
                   cmd_help >&2
                   exit 2 ;;
    esac
}

main "$@"
