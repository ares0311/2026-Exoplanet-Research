"""Behavioral tests for the persistent EXO-Hunter slash terminal."""
from __future__ import annotations

import io
import tomllib
from collections.abc import Sequence
from pathlib import Path

import pytest

from exo_toolkit.hunter_shell import (
    HunterShell,
    _animation_allowed,
    _script_commands,
)


def _recorder(
    name: str,
    calls: list[tuple[str, tuple[str, ...]]],
    *,
    exit_code: int = 0,
):
    def _record(argv: Sequence[str] | None = None) -> int:
        calls.append((name, tuple(argv or ())))
        return exit_code

    return _record


def _shell(
    tmp_path: Path,
    calls: list[tuple[str, tuple[str, ...]]],
    *,
    create_exit: int = 0,
) -> HunterShell:
    return HunterShell(
        db_path=tmp_path / "hunter.sqlite3",
        history_path=tmp_path / "history",
        no_color=True,
        no_animation=True,
        create_fn=_recorder("create", calls, exit_code=create_exit),
        run_fn=_recorder("run", calls),
        show_fn=_recorder("show", calls),
        import_fn=_recorder("import", calls),
        recheck_fn=_recorder("recheck", calls),
        input_stream=io.StringIO(),
        output_stream=io.StringIO(),
        error_stream=io.StringIO(),
    )


def test_pyproject_registers_required_persistent_entry_points() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    scripts = project["project"]["scripts"]
    assert scripts["EXO-Hunter"] == "exo_toolkit.hunter_shell:exohunter_entry"
    assert scripts["ExoHunter"] == "exo_toolkit.hunter_shell:exohunter_entry"


def test_slash_and_help_expose_required_workflow(tmp_path: Path) -> None:
    calls: list[tuple[str, tuple[str, ...]]] = []
    output = io.StringIO()
    shell = _shell(tmp_path, calls)
    shell._console = shell._console.__class__(  # noqa: SLF001
        file=output,
        no_color=True,
        force_terminal=False,
        highlight=False,
    )

    assert shell.dispatch("/").exit_code == 0
    help_output = output.getvalue()
    for command in (
        "/Create-New-Search",
        "/New-Search",
        "/Follow-Up-Search",
        "/Run-New-Search",
        "/Run-Search",
        "/Show-Follow-Ups",
        "/Help",
        "/Exit",
    ):
        assert command in help_output
    assert calls == []


def test_canonical_create_command_delegates_exact_options(tmp_path: Path) -> None:
    calls: list[tuple[str, tuple[str, ...]]] = []
    shell = _shell(tmp_path, calls)

    assert (
        shell.dispatch(
            "/Create-New-Search --targets 3 --mode follow-up --workers 4"
        ).exit_code
        == 0
    )

    assert calls == [
        (
            "create",
            (
                "--targets",
                "3",
                "--mode",
                "follow-up",
                "--workers",
                "4",
                "--db",
                str(tmp_path / "hunter.sqlite3"),
                "--no-color",
            ),
        )
    ]


def test_new_and_follow_up_delegate_to_one_canonical_creator(tmp_path: Path) -> None:
    calls: list[tuple[str, tuple[str, ...]]] = []
    shell = _shell(tmp_path, calls)

    assert shell.dispatch("/New-Search 3 --workers 4 --tmag-min 11").exit_code == 0
    assert shell.dispatch("/Follow-Up-Search 2 --workers 5").exit_code == 0

    new_args = calls[0][1]
    follow_up_args = calls[1][1]
    assert new_args[:4] == ("--targets", "3", "--mode", "new")
    assert follow_up_args[:4] == ("--targets", "2", "--mode", "follow-up")
    assert ("--db", str(tmp_path / "hunter.sqlite3")) == (
        new_args[new_args.index("--db")],
        new_args[new_args.index("--db") + 1],
    )
    assert "--no-color" in new_args
    assert calls[0][0] == calls[1][0] == "create"


@pytest.mark.parametrize(
    ("command", "handler"),
    (
        ("/Run-New-Search --workers 6 --scorer bayesian", "run"),
        ("/Run-Search --workers 6 --scorer bayesian", "run"),
        ("/Show-Follow-Ups --status all", "show"),
        ("/Import-Follow-Up --evidence-file evidence.json", "import"),
        ("/Recheck-Follow-Ups --workers 4", "recheck"),
    ),
)
def test_workflow_commands_delegate_without_reimplementing_logic(
    tmp_path: Path, command: str, handler: str
) -> None:
    calls: list[tuple[str, tuple[str, ...]]] = []
    shell = _shell(tmp_path, calls)

    assert shell.dispatch(command).exit_code == 0
    assert calls[0][0] == handler
    assert "--db" in calls[0][1]


@pytest.mark.parametrize(
    "command",
    (
        "/New-Search",
        "/New-Search zero",
        "/New-Search 0",
        "/New-Search 1 --targets 2",
        "/Follow-Up-Search 1 --mode new",
        "/Exit unexpected",
        "/Not-A-Command",
        "'unterminated",
    ),
)
def test_useful_errors_are_nonzero_without_closing_shell(
    tmp_path: Path, command: str
) -> None:
    calls: list[tuple[str, tuple[str, ...]]] = []
    shell = _shell(tmp_path, calls)

    result = shell.dispatch(command)

    assert result.exit_code == 2
    assert result.should_exit is False


def test_exit_is_explicit_and_case_insensitive(tmp_path: Path) -> None:
    shell = _shell(tmp_path, [])
    result = shell.dispatch("/eXiT")
    assert result.exit_code == 0
    assert result.should_exit is True


def test_argparse_system_exit_does_not_kill_persistent_shell(tmp_path: Path) -> None:
    def parser_failure(_argv: Sequence[str] | None = None) -> int:
        raise SystemExit(2)

    shell = HunterShell(
        db_path=tmp_path / "hunter.sqlite3",
        no_color=True,
        no_animation=True,
        run_fn=parser_failure,
        input_stream=io.StringIO(),
        output_stream=io.StringIO(),
        error_stream=io.StringIO(),
    )

    result = shell.dispatch("/Run-Search --not-a-real-option")

    assert result.exit_code == 2
    assert result.should_exit is False


def test_script_mode_stops_after_first_failed_canonical_command(tmp_path: Path) -> None:
    calls: list[tuple[str, tuple[str, ...]]] = []
    shell = _shell(tmp_path, calls, create_exit=2)

    exit_code = shell.run_commands(
        ("/New-Search 1", "/Run-Search", "/Show-Follow-Ups")
    )

    assert exit_code == 2
    assert [name for name, _ in calls] == ["create"]


def test_script_mode_does_not_prefix_machine_readable_stdout(tmp_path: Path) -> None:
    calls: list[tuple[str, tuple[str, ...]]] = []
    output = io.StringIO()
    shell = HunterShell(
        db_path=tmp_path / "hunter.sqlite3",
        no_color=True,
        no_animation=True,
        show_fn=_recorder("show", calls),
        input_stream=io.StringIO(),
        output_stream=output,
        error_stream=io.StringIO(),
    )

    assert shell.run_commands(("/Show-Follow-Ups --json",)) == 0

    assert output.getvalue() == ""
    assert "--json" in calls[0][1]


def test_interactive_session_survives_failure_until_exit(tmp_path: Path) -> None:
    calls: list[tuple[str, tuple[str, ...]]] = []
    shell = _shell(tmp_path, calls, create_exit=2)
    commands = iter(("/New-Search 1", "/Help", "/Exit"))

    assert shell.run_interactive(input_fn=lambda _prompt: next(commands)) == 0
    assert [name for name, _ in calls] == ["create"]
    assert (tmp_path / "history").is_file()


def test_script_reader_supports_file_and_redirected_stdin(tmp_path: Path) -> None:
    script = tmp_path / "commands.txt"
    script.write_text("/Help\n/Exit\n", encoding="utf-8")

    assert _script_commands(script, input_stream=io.StringIO()) == ["/Help", "/Exit"]
    assert _script_commands(Path("-"), input_stream=io.StringIO("/Help\n")) == ["/Help"]


def test_animation_disables_for_redirected_output_and_accessibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    non_tty = io.StringIO()
    assert not _animation_allowed(
        disabled=False,
        input_stream=non_tty,
        stream=non_tty,
    )

    class TTY(io.StringIO):
        def isatty(self) -> bool:
            return True

    monkeypatch.setenv("REDUCE_MOTION", "1")
    assert not _animation_allowed(
        disabled=False,
        input_stream=TTY(),
        stream=TTY(),
    )


def test_explicit_db_option_is_not_overridden(tmp_path: Path) -> None:
    calls: list[tuple[str, tuple[str, ...]]] = []
    shell = _shell(tmp_path, calls)

    assert shell.dispatch("/Show-Follow-Ups --db=custom.sqlite3").exit_code == 0

    assert "--db=custom.sqlite3" in calls[0][1]
    assert calls[0][1].count("--db") == 0
