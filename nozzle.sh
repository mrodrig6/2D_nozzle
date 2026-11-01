#!/usr/bin/env bash
#
# nozzle.sh -- one entry point for the 2D DG nozzle code.
#
# The point of this script is that nothing has to be set up first.  It finds a
# Python, puts the repository on the import path if the package is not installed,
# and runs what you asked for from wherever you happen to be:
#
#     ./nozzle.sh run cases/design.json
#     ./nozzle.sh run area_ratio=3.0 back_pressure_ratio=0.12 order=1
#
# `./nozzle.sh help` lists everything.  `./nozzle.sh install` is optional.

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SELF="$(basename -- "${BASH_SOURCE[0]}")"

# --------------------------------------------------------------------------
# Python
# --------------------------------------------------------------------------
# Whatever environment you have loaded is the one used.  Set NOZZLE_PYTHON to
# override, e.g. NOZZLE_PYTHON=python3.12 ./nozzle.sh run
find_python() {
    if [[ -n ${NOZZLE_PYTHON:-} ]]; then
        command -v "$NOZZLE_PYTHON" >/dev/null 2>&1 || die \
            "NOZZLE_PYTHON=$NOZZLE_PYTHON is not an executable on PATH"
        printf '%s\n' "$NOZZLE_PYTHON"
        return
    fi
    local candidate
    for candidate in python3 python; do
        if command -v "$candidate" >/dev/null 2>&1; then
            printf '%s\n' "$candidate"
            return
        fi
    done
    die "no python3 on PATH.  Load your environment first, or set NOZZLE_PYTHON."
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
# Case files (cases/*.json)
# --------------------------------------------------------------------------
# A case is a JSON object of name: value pairs, using the names the Python API
# uses.  Pass its path where the arguments go, and any name=value after it
# overrides that value.  The key "description" is for people and is not passed
# on.
#
#     ./nozzle.sh run cases/design.json pb=0.12
#
# A case file does not include another, so expansion is one level deep.
CASE_DIR="$ROOT/cases"

# Output a case as 'name=value' lines, without "description".
load_case() {
    "$PY" - "$1" <<'PYEOF'
import json
import sys

with open(sys.argv[1]) as f:
    case = json.load(f)
if not isinstance(case, dict):
    sys.exit("a case file holds one JSON object of name: value pairs")
for key, value in case.items():
    if key == "description":
        continue
    if isinstance(value, bool):
        value = "yes" if value else "no"
    elif isinstance(value, list):
        value = ",".join(str(item) for item in value)
    print(f"{key}={value}")
PYEOF
}

list_case_files() {
    local path found=0
    for path in "$CASE_DIR"/*.json; do
        [[ -e $path ]] || continue
        found=1
        printf '  %-28s %s\n' "${path#"$ROOT"/}" "$(case_description "$path")"
    done
    (( found )) || printf '  (none in %s)\n' "${CASE_DIR#"$ROOT"/}"
}

case_description() {
    local py=${PY:-$(find_python)}
    "$py" -c 'import json, sys; print(json.load(open(sys.argv[1])).get("description", ""))' "$1" \
        2>/dev/null || echo ""
}

# Replace every case-file argument with that file's name=value lines.  EXPANDED
# is set as a side effect; bash functions cannot return arrays.
expand_cases() {
    EXPANDED=()
    local token text line
    for token in ${@+"$@"}; do
        if [[ $token == *=* || $token != *.json ]]; then
            EXPANDED+=("$token")
            continue
        fi
        [[ -f $token ]] || die "no case file $token.  Available cases:
$(list_case_files)"
        text=$(load_case "$token") || die "$token is not a readable case file"
        while IFS= read -r line; do EXPANDED+=("$line"); done <<< "$text"
    done
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
    expand_cases ${@+"$@"}
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

# Run a Python subcommand with translated arguments.
python_command() {
    local subcommand=$1; shift
    translate "$@"
    exec "$PY" -m src "$subcommand" ${TRANSLATED[@]+"${TRANSLATED[@]}"}
}

# --------------------------------------------------------------------------
# Examples
# --------------------------------------------------------------------------
# The runnable scripts in examples/.  Numbered for reading order; `example`
# matches on the descriptive part, so `example sweep` and `example 02` both work.
list_examples() {
    local path stem
    for path in "$ROOT"/examples/[0-9][0-9]_*.py; do
        [[ -e $path ]] || continue
        stem=$(basename "$path" .py)
        printf '  %-14s %s\n' "${stem#*_}" "$(example_summary "$path")"
    done
}

example_summary() {
    # the one-line summary is the first docstring line, after the em dash
    sed -n '2{s/^"""//;s/ *$//;s/\.$//;s/.*--- *//;p;q;}' "$1"
}

resolve_example() {
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
        die "'$want' matches more than one example"
    fi
    printf '%s: no example named %s.  Available:\n' "$SELF" "$want" >&2
    list_examples >&2
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
  list               list the case files in cases/, and the examples

Running
  run CASE.json      solve the operating point in a case file and report it
  sweep <var> <lo> <hi> <n> [...]
                     sweep one design variable
  geometry [...]     inspect a contour; no flow solve
  bench [...]        time the backends against each other
  example <name>     run one of the scripts in examples/
  test [...]         run the test suite
  verify [--fast]    everything the project checks about itself: tests, lint,
                     format and spelling.  '--fast' skips the full solves.
  docs               print where the documentation is

Arguments are 'name=value', using the same names as the Python API:

  ./$SELF run area_ratio=3.0 back_pressure_ratio=0.12 order=1
  ./$SELF run ar=3.0 pb=0.12 p=1 figure=result.png     # short aliases
  ./$SELF run cases/design.json figure=design.png
  ./$SELF run cases/design.json pb=0.12                 # a case, one value changed
  ./$SELF sweep area_ratio 2.0 4.0 9 order=1 csv=sweep.csv
  ./$SELF geometry contour=bezier bezier_w1=0.7
  ./$SELF example sweep

A case file is a JSON object of name: value pairs, and a name=value argument
after it overrides that value.  The file goes where the arguments go.  A command
reports any key it does not use, so geometry takes the geometry keys of a case
and run takes the operating point as well.

Aliases: p=order, Q=geometry_order, ref=refine, ar=area_ratio,
pb=back_pressure_ratio, tvb=tvb_constant.  Ordinary --flags work too, so
anything 'python -m src --help' documents is still available.
'M' is deliberately NOT an alias: in this code M always means Mach number.

Cases (cases/*.json):
$(list_case_files)

Examples ('example <name>'):
$(list_examples)

Environment
  NOZZLE_PYTHON      the interpreter to use (default: python3, then python)

No install is required: if the solver package is not importable, the repository root
is put on PYTHONPATH automatically.  Documentation is in docs/ -- start with
docs/README.md for the index.
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
print(f"nozzle {src.__version__}  ({where})")
print(f"backends {', '.join(available_backends())}")

for name, what in (("numba", "the fast solver"),
                   ("jax", "gradients and sensitivity"),
                   ("matplotlib", "figures"),
                   ("scipy", "optimisation")):
    ok = importlib.util.find_spec(name) is not None
    print(f"  {'yes' if ok else 'NO '}  {name:12s} {what}")
PYEOF
}

cmd_example() {
    if (( $# == 0 )); then
        printf "Examples ('example <name>'):\n"
        list_examples
        return 0
    fi
    local case_path
    case_path=$(resolve_example "$1") || exit 2
    shift
    printf '%s: running %s\n\n' "$SELF" "${case_path#"$ROOT"/}"
    exec "$PY" "$case_path" "$@"
}

cmd_test() {
    translate "$@"
    # Only default to the whole tree when no path was given -- appending $ROOT
    # unconditionally made `nozzle.sh test tests/test_docs.py` run everything,
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
docs/README.md                 the index of every documentation page
docs/theory/euler.md           the formulation, the geometry definition and the verification evidence
docs/running/lab-guide.md      the exercises, and what to report
docs/figures/tikz/             TikZ sources for every figure
README.md                      the quick start and the API tour
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
        list)      printf "Cases (cases/*.json):\n"; list_case_files
                   printf "\nExamples ('example <name>'):\n"; list_examples ;;
        docs)      cmd_docs ;;
        run)       setup_import_path; python_command solve "$@" ;;
        example)   setup_import_path; cmd_example "$@" ;;
        test)      setup_import_path; cmd_test "$@" ;;
        verify)    setup_import_path; cmd_verify ${1+"$1"} ;;
        sweep|geometry|bench)
                   setup_import_path; python_command "$command" "$@" ;;
        *)         printf '%s: unknown command %s\n\n' "$SELF" "$command" >&2
                   cmd_help >&2
                   exit 2 ;;
    esac
}

main "$@"
