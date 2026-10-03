"""The ``dg2d.sh`` launcher.

This is the entry point students are told to use.  The package sits at the
repository root, so ``python -m src`` works when you are standing in a
clone; what the launcher adds is that it works from *anywhere*, with no install
and without having to know where the clone is.  So the tests here run the script
the way a student would -- from an arbitrary directory, with no arguments beyond
the ones documented -- rather than inspecting it.
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
    """``list`` prints two sections; the cases are the entries without an ``@``."""
    out = run("list")
    assert out.returncode == 0
    listed = [
        line.split()[0]
        for line in out.stdout.splitlines()
        if line.startswith("  ") and line.strip() and not line.strip().startswith("@")
    ]
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


def test_test_passes_a_path_filter_through(tmp_path):
    """``test <path>`` must narrow the run, not quietly run everything.

    The command defaults to the whole tree, and appending that default
    unconditionally made a path argument look ignored.
    """
    out = run("test", "-q", "--collect-only", "tests/test_docs.py", timeout=180)
    assert out.returncode == 0, out.stderr
    assert "test_docs.py" in out.stdout
    assert "test_physics.py" not in out.stdout


def test_docs_points_at_the_formulation():
    out = run("docs")
    assert out.returncode == 0
    assert "docs/theory.md" in out.stdout
    assert "docs/lab_guide.md" in out.stdout


def test_check_works_from_an_unrelated_directory(tmp_path):
    """The whole point: no install, no clone to stand in, any working directory."""
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


# --------------------------------------------------------------------------
# Argument decks
# --------------------------------------------------------------------------
# These check expansion, not the solver, so they lean on `geometry` -- which
# does no flow solve and echoes the contour it was given -- and on the error
# message argparse prints for a key a subcommand does not take.  That message
# names the keys it rejected, which makes it a readout of exactly what the
# launcher passed through.

DECKS = ("design", "overexpanded", "shocked", "converged", "contour")


def test_the_shipped_decks_are_listed(tmp_path):
    out = run("list", cwd=tmp_path)
    assert out.returncode == 0, out.stderr
    for name in DECKS:
        assert f"@{name}" in out.stdout, f"{name} missing from `list`"


def test_every_shipped_deck_has_a_summary_line(tmp_path):
    """The summary is the deck's first comment, so a deck without one is silent."""
    out = run("list", cwd=tmp_path)
    for line in out.stdout.splitlines():
        if line.strip().startswith("@"):
            name, _, summary = line.strip().partition(" ")
            assert summary.strip(), f"{name} has no summary comment"


def test_a_deck_expands_to_the_values_it_holds(tmp_path):
    """``geometry @contour`` must behave as if the lines had been typed."""
    deck = run("geometry", "@contour", cwd=tmp_path)
    typed = run("geometry", "contour=bell", "ar=2.5", "theta_initial_deg=30",
                cwd=tmp_path)
    assert deck.returncode == 0, deck.stderr
    assert deck.stdout == typed.stdout


def test_a_later_argument_overrides_the_deck(tmp_path):
    """The whole point of a deck is that it is a starting point, not a cage."""
    deck = run("geometry", "@contour", "ar=4.0", cwd=tmp_path)
    typed = run("geometry", "contour=bell", "ar=4.0", "theta_initial_deg=30",
                cwd=tmp_path)
    assert deck.returncode == 0, deck.stderr
    assert deck.stdout == typed.stdout
    assert "AR=4.0" in deck.stdout


def test_a_deck_can_include_another_deck(tmp_path):
    """``overexpanded`` is ``design`` with one value changed."""
    out = run("geometry", "@overexpanded", cwd=tmp_path)
    # geometry rejects the solver keys, and in doing so lists what it was given
    assert "--area-ratio" not in out.stderr  # geometry *does* take this one
    assert "--order" in out.stderr, out.stderr
    assert out.returncode != 0


def test_a_deck_resolves_by_name_or_by_path(tmp_path):
    by_name = run("geometry", "@contour", cwd=tmp_path)
    by_path = run("geometry", f"@{ROOT / 'cases' / 'contour.dg'}", cwd=tmp_path)
    assert by_name.stdout == by_path.stdout
    assert by_name.returncode == 0, by_name.stderr


def test_an_unknown_deck_names_the_ones_that_exist(tmp_path):
    out = run("geometry", "@nosuchdeck", cwd=tmp_path)
    assert out.returncode == 2
    assert "no deck named nosuchdeck" in out.stderr
    assert "@design" in out.stderr, "the error should list what is available"


def test_comments_and_blank_lines_are_ignored(tmp_path, monkeypatch):
    deck = tmp_path / "noisy.dg"
    deck.write_text(
        "# a summary\n"
        "\n"
        "contour=bell   # trailing comment\n"
        "   \n"
        "# a whole-line comment\n"
        "  ar=2.5  \n",
        encoding="utf-8",
    )
    noisy = run("geometry", f"@{deck}", cwd=tmp_path)
    plain = run("geometry", "contour=bell", "ar=2.5", cwd=tmp_path)
    assert noisy.returncode == 0, noisy.stderr
    assert noisy.stdout == plain.stdout


def test_a_self_including_deck_fails_instead_of_hanging(tmp_path):
    """A deck that includes itself must hit the depth limit, not spin forever."""
    deck = tmp_path / "loop.dg"
    deck.write_text(f"# recursive\n@{deck}\n", encoding="utf-8")
    out = run("geometry", f"@{deck}", cwd=tmp_path, timeout=60)
    assert out.returncode == 2
    assert "nested more than" in out.stderr, out.stderr


def test_a_pytest_node_id_is_not_mistaken_for_a_deck(tmp_path):
    """``test tests/x.py::name`` must not try to expand anything."""
    out = run("test", "tests/test_dg2d.py::test_the_shipped_decks_are_listed",
              "--collect-only", "-q", cwd=tmp_path)
    assert "no deck named" not in out.stderr
