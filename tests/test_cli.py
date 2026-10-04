"""Command-line interface.

Every subcommand is invoked here, because the absence of that coverage is how a
real defect survived: ``_collect`` read solver attributes by direct attribute
access, but the ``geometry`` subparser declares no solver options, so
``python -m src geometry`` raised ``AttributeError`` before doing any work.
Nothing that only inspects ``--help`` would have caught it.

Only ``geometry`` runs here without a flow solve; the subcommands that do are
marked ``slow``.
"""

from __future__ import annotations

import pytest

from src.cli import _collect, build_parser, main

SUBCOMMANDS = ("solve", "sweep", "geometry", "bench")


@pytest.mark.parametrize("command", SUBCOMMANDS)
def test_help_is_available_for_every_subcommand(command, capsys):
    with pytest.raises(SystemExit) as exc:
        main([command, "--help"])
    assert exc.value.code == 0
    assert command in capsys.readouterr().out


@pytest.mark.parametrize("command", SUBCOMMANDS)
def test_collect_works_with_whatever_the_parser_defined(command):
    """The defect: a subcommand that omits an argument group must still collect."""
    parser = build_parser()
    argv = {
        "solve": [command],
        "bench": [command],
        "geometry": [command],
        "sweep": [command, "area_ratio", "2.0", "3.0", "3"],
    }[command]
    args = parser.parse_args(argv)
    collected = _collect(args)
    # geometry declares no solver or discretisation options, so those keys must
    # simply be absent rather than causing a failure
    assert "contour" in collected
    if command == "geometry":
        assert "element" not in collected
        assert "cfl" not in collected
    else:
        assert "element" in collected
        assert "cfl" in collected


def test_geometry_subcommand_runs_and_reports(capsys):
    """Exercises the real path: no flow solve, but the full report."""
    assert main(["geometry", "--contour", "bell", "--area-ratio", "3.0"]) == 0
    out = capsys.readouterr().out
    assert "bell contour" in out
    assert "area_ratio" in out
    assert "throat_wall_angle_deg" in out
    assert "choking" in out  # the critical-ratio line


@pytest.mark.parametrize("contour", ["bell", "smooth", "conical", "bezier", "analytic"])
def test_geometry_subcommand_accepts_every_contour(contour, capsys):
    assert main(["geometry", "--contour", contour]) == 0
    assert contour in capsys.readouterr().out


def test_geometry_subcommand_writes_a_figure(tmp_path):
    pytest.importorskip("matplotlib")
    path = tmp_path / "wall.png"
    assert main(["geometry", "--contour", "smooth", "--figure", str(path)]) == 0
    assert path.exists() and path.stat().st_size > 0


def test_unknown_subcommand_is_rejected():
    with pytest.raises(SystemExit) as exc:
        main(["nonsense"])
    assert exc.value.code != 0


def _choices_for(option: str, subcommand: str = "solve") -> tuple[str, ...]:
    """The choices a subcommand's parser declares for one option."""
    parser = build_parser()
    subparsers = next(
        a for a in parser._actions if isinstance(a.choices, dict) and subcommand in a.choices
    )
    action = next(a for a in subparsers.choices[subcommand]._actions if a.dest == option)
    return tuple(action.choices or ())


def test_every_x_spacing_the_cli_offers_actually_builds_a_mesh():
    """The CLI must not offer a spacing the mesh generator rejects.

    The choices are read off the parser rather than written out here, so this
    keeps checking the real invariant if they are ever renamed.
    """
    from src import NozzleGeometry, build_case

    choices = _choices_for("x_spacing")
    assert choices, "the CLI declares no x_spacing choices"

    geom = NozzleGeometry(contour="smooth")
    for choice in choices:
        case = build_case(geom, x_spacing=choice)
        assert case.topology.n_elem > 0, choice


def test_every_contour_the_cli_offers_is_a_real_contour():
    from src import CONTOURS

    # --contour is free text rather than a choice list, so check the default
    parser = build_parser()
    assert parser.parse_args(["solve"]).contour in CONTOURS


@pytest.mark.slow
def test_solve_subcommand_end_to_end(capsys):
    assert main(["solve", "--order", "0", "--contour", "smooth", "--quiet"]) == 0
    out = capsys.readouterr().out
    assert "thrust" in out
    assert "mass flow" in out


@pytest.mark.slow
def test_solve_subcommand_reports_failure_in_its_exit_code():
    """A run that does not converge must not exit 0."""
    assert (
        main(
            [
                "solve",
                "--order",
                "1",
                "--contour",
                "smooth",
                "--cfl",
                "80",
                "--max-iterations",
                "600",
                "--quiet",
            ]
        )
        == 1
    )


@pytest.mark.slow
def test_sweep_subcommand_writes_a_csv(tmp_path):
    csv = tmp_path / "sweep.csv"
    code = main(
        [
            "sweep",
            "area_ratio",
            "2.0",
            "2.5",
            "2",
            "--order",
            "0",
            "--contour",
            "smooth",
            "--quiet",
            "--csv",
            str(csv),
        ]
    )
    assert code == 0
    lines = csv.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    assert "area_ratio" in lines[0]
