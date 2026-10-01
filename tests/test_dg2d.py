"""The ``dg2d.sh`` launcher.

This is the entry point students are told to use, and the thing it exists to
fix is that ``python -m dgnozzle`` does not work from a clone: the package is
under ``src/``, so without an install there is nothing to import.  So the tests
here run the script the way a student would -- from an arbitrary directory,
with no arguments beyond the ones documented -- rather than inspecting it.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "dg2d.sh"

pytestmark = pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("bash") is None,
    reason="dg2d.sh needs bash",
)

CASES = ("solve", "sweep", "sensitivity", "optimise", "convergence", "shock")


def run(*args, cwd=None, timeout=300):
    """Run the launcher, returning the completed process."""
    env = dict(os.environ)
    # a student's shell has no PYTHONPATH pointing at this repo; the script is
    # supposed to cope with that by itself
    env.pop("PYTHONPATH", None)
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        cwd=str(cwd or ROOT), env=env, capture_output=True, text=True, timeout=timeout,
    )


def test_the_script_is_executable():
    assert SCRIPT.exists()
    assert os.access(SCRIPT, os.X_OK), "dg2d.sh must be committed with its +x bit"


def test_bash_parses_it():
    assert subprocess.run(["bash", "-n", str(SCRIPT)]).returncode == 0


@pytest.mark.parametrize("flag", ["help", "-h", "--help"])
def test_help_lists_every_command(flag):
    out = run(flag)
    assert out.returncode == 0
    for command in ("install", "check", "list", "run", "solve", "sweep",
                    "geometry", "bench", "test", "docs"):
        assert command in out.stdout


def test_no_arguments_is_help():
    out = run()
    assert out.returncode == 0
    assert "Usage:" in out.stdout


def test_unknown_command_fails_and_explains():
    out = run("nonsense")
    assert out.returncode != 0
    assert "unknown command" in out.stderr
    assert "Usage:" in out.stderr


def test_list_names_every_case():
    out = run("list")
    assert out.returncode == 0
    listed = [line.split()[0] for line in out.stdout.splitlines() if line.strip()]
    assert listed == list(CASES)


@pytest.mark.parametrize("name", CASES)
def test_every_listed_case_resolves_to_exactly_one_file(name):
    """``run`` matches on the descriptive half of the filename.

    If a case is renamed, ``list`` and ``run`` must move together -- this is the
    invariant that keeps the names in the README from going stale.
    """
    matches = sorted((ROOT / "examples").glob(f"[0-9][0-9]_{name}.py"))
    assert len(matches) == 1, matches


def test_unknown_case_fails_and_lists_the_real_ones():
    out = run("run", "nonsense")
    assert out.returncode != 0
    assert "no case named nonsense" in out.stderr
    for name in CASES:
        assert name in out.stderr


def test_run_needs_a_case_name():
    out = run("run")
    assert out.returncode != 0


def test_a_bad_interpreter_is_reported_not_ignored():
    """``die`` runs inside a command substitution, so its exit must propagate."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["DG2D_PYTHON"] = "definitely-not-an-interpreter"
    out = subprocess.run(
        ["bash", str(SCRIPT), "check"],
        cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=60,
    )
    assert out.returncode != 0
    assert "DG2D_PYTHON" in out.stderr


def test_docs_points_at_the_formulation():
    out = run("docs")
    assert out.returncode == 0
    assert "docs/theory.md" in out.stdout
    assert "docs/lab_guide.md" in out.stdout


def test_check_works_from_an_unrelated_directory(tmp_path):
    """The whole point: no install, no cd into src/, any working directory."""
    out = run("check", cwd=tmp_path)
    assert out.returncode == 0, out.stderr
    assert "dgnozzle" in out.stdout
    assert "backends" in out.stdout


def test_key_value_arguments_reach_the_solver(tmp_path):
    """``geometry`` runs no flow solve, so this is a cheap test of translation."""
    out = run("geometry", "contour=conical", "area_ratio=3.0", cwd=tmp_path)
    assert out.returncode == 0, out.stderr
    assert "conical contour" in out.stdout
    assert "AR=3" in out.stdout


def test_short_aliases_mean_the_same_thing(tmp_path):
    long = run("geometry", "contour=conical", "area_ratio=3.0", cwd=tmp_path)
    short = run("geometry", "contour=conical", "ar=3.0", cwd=tmp_path)
    assert short.returncode == 0, short.stderr
    assert short.stdout == long.stdout


def test_ordinary_flags_still_pass_through(tmp_path):
    """Nothing the underlying CLI accepts is taken away."""
    out = run("geometry", "--contour", "conical", "--area-ratio", "3.0", cwd=tmp_path)
    assert out.returncode == 0, out.stderr
    assert "conical contour" in out.stdout


def test_a_zero_valued_option_is_not_mistaken_for_a_flag(tmp_path):
    """``refine=0`` must pass ``--refine 0``, not swallow the 0 as a boolean."""
    out = run("geometry", "contour=conical", "theta_exit_deg=0", cwd=tmp_path)
    assert out.returncode == 0, out.stderr


@pytest.mark.slow
def test_solve_runs_end_to_end(tmp_path):
    out = run("solve", "p=0", "contour=smooth", "quiet=yes", cwd=tmp_path, timeout=900)
    assert out.returncode == 0, out.stderr
    assert "thrust" in out.stdout
    assert "mass flow" in out.stdout


@pytest.mark.slow
def test_run_executes_a_case(tmp_path):
    """The shortest case, to prove ``run`` actually runs one."""
    pytest.importorskip("matplotlib")
    out = run("run", "solve", cwd=tmp_path, timeout=900)
    assert out.returncode == 0, out.stderr
    assert "running examples/01_solve.py" in out.stdout
