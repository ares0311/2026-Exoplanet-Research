"""Persistent slash-command terminal for the canonical EXO-Hunter workflow."""
from __future__ import annotations

import argparse
import os
import shlex
import sys
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import TextIO

from rich.console import Console
from rich.table import Table

from exo_toolkit import __version__
from exo_toolkit.hunter_cli import (
    DEFAULT_HUNTER_DB,
    create_new_search,
    import_follow_up,
    recheck_follow_ups,
    run_new_search,
    show_follow_ups,
)

CommandFn = Callable[[Sequence[str] | None], int]

_SLASH_COMMANDS = (
    "/Create-New-Search",
    "/New-Search",
    "/Follow-Up-Search",
    "/Run-New-Search",
    "/Run-Search",
    "/Show-Follow-Ups",
    "/Import-Follow-Up",
    "/Recheck-Follow-Ups",
    "/Help",
    "/Exit",
)
_ANIMATION_FRAMES = (
    "      ·  ◉  ·      ",
    "    ·   ◉   ·      ",
    "  ·    ◉    ·      ",
    "·     ◉     ·      ",
    "  ·    ◉    ·      ",
    "    ·   ◉   ·      ",
)


@dataclass(frozen=True)
class CommandResult:
    """One slash-command outcome."""

    exit_code: int
    should_exit: bool = False


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="EXO-Hunter",
        description=(
            "Persistent EXO-Hunter terminal. Enter / for commands; "
            "the session remains active until /Exit."
        ),
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_HUNTER_DB)
    parser.add_argument(
        "--command",
        action="append",
        default=[],
        help="Run one slash command non-interactively; may be repeated.",
    )
    parser.add_argument(
        "--script",
        type=Path,
        help="Run newline-delimited slash commands from a file; use '-' for stdin.",
    )
    parser.add_argument(
        "--history-file",
        type=Path,
        default=Path("data/.exohunter_history"),
    )
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument("--no-animation", action="store_true")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def _animation_allowed(*, disabled: bool, input_stream: TextIO, stream: TextIO) -> bool:
    """Return whether motion is appropriate for this terminal."""
    reduce_motion = any(
        os.environ.get(name, "").strip().lower() not in {"", "0", "false", "no"}
        for name in ("CI", "REDUCE_MOTION", "EXOHUNTER_REDUCE_MOTION")
    )
    return bool(
        not disabled
        and not reduce_motion
        and os.environ.get("TERM", "").lower() != "dumb"
        and input_stream.isatty()
        and stream.isatty()
    )


class _OrbitAnimator:
    """Small exoplanet-orbit animation whose lifetime matches real work."""

    def __init__(
        self,
        label: str,
        *,
        enabled: bool,
        stream: TextIO,
        interval_seconds: float = 0.12,
    ) -> None:
        self._label = label
        self._enabled = enabled
        self._stream = stream
        self._interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = 0.0

    def __enter__(self) -> _OrbitAnimator:
        self._started = time.monotonic()
        if self._enabled:
            self._thread = threading.Thread(
                target=self._animate,
                name="exohunter-orbit-animation",
                daemon=True,
            )
            self._thread.start()
        else:
            print(f"[orbit] {self._label}", file=self._stream, flush=True)
        return self

    def _animate(self) -> None:
        index = 0
        while not self._stop.is_set():
            frame = _ANIMATION_FRAMES[index % len(_ANIMATION_FRAMES)]
            elapsed = time.monotonic() - self._started
            self._stream.write(f"\r{frame}  {self._label}  {elapsed:5.1f}s")
            self._stream.flush()
            index += 1
            self._stop.wait(self._interval_seconds)

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=max(1.0, self._interval_seconds * 3))
            self._stream.write("\r" + (" " * 79) + "\r")
            self._stream.flush()


class HunterShell:
    """Thin terminal adapter over the canonical Hunter CLI functions."""

    def __init__(
        self,
        *,
        db_path: Path = DEFAULT_HUNTER_DB,
        history_path: Path = Path("data/.exohunter_history"),
        no_color: bool = False,
        no_animation: bool = False,
        create_fn: CommandFn = create_new_search,
        run_fn: CommandFn = run_new_search,
        show_fn: CommandFn = show_follow_ups,
        import_fn: CommandFn = import_follow_up,
        recheck_fn: CommandFn = recheck_follow_ups,
        input_stream: TextIO = sys.stdin,
        output_stream: TextIO = sys.stdout,
        error_stream: TextIO = sys.stderr,
    ) -> None:
        self.db_path = db_path
        self.history_path = history_path
        self.no_color = no_color
        self.no_animation = no_animation
        self._create_fn = create_fn
        self._run_fn = run_fn
        self._show_fn = show_fn
        self._import_fn = import_fn
        self._recheck_fn = recheck_fn
        self._input = input_stream
        self._output = output_stream
        self._error = error_stream
        self._console = Console(
            file=output_stream,
            no_color=no_color,
            force_terminal=False if no_color else None,
            highlight=False,
        )
        self._readline: ModuleType | None = None

    @property
    def animate(self) -> bool:
        return _animation_allowed(
            disabled=self.no_animation,
            input_stream=self._input,
            stream=self._error,
        )

    def print_banner(self) -> None:
        """Render a startup identity and a short orbit/transit motif."""
        if self.animate:
            with _OrbitAnimator(
                "aligning transit ephemerides",
                enabled=True,
                stream=self._error,
                interval_seconds=0.08,
            ):
                time.sleep(0.32)
        self._console.print(
            f"[bold cyan]EXO-Hunter {__version__}[/bold cyan]  "
            "[dim]adaptive search → durable evidence[/dim]"
        )
        self._console.print("Enter [bold]/[/bold] for commands; exit only with [bold]/Exit[/bold].")

    def show_help(self) -> None:
        """Show the complete slash-command surface."""
        table = Table(title="EXO-Hunter commands", show_lines=False)
        table.add_column("Command", style="cyan", no_wrap=True)
        table.add_column("Canonical action")
        table.add_row(
            "/Create-New-Search --targets <N> --mode <new|follow-up> [options]",
            "Canonical search-creation command",
        )
        table.add_row("/New-Search <N> [options]", "Create exact best-available new-target search")
        table.add_row(
            "/Follow-Up-Search <N> [options]",
            "Create exact best-available follow-up search",
        )
        table.add_row("/Run-New-Search [options]", "Canonical exact-manifest execution command")
        table.add_row("/Run-Search [options]", "Run or resume the exact pending manifest")
        table.add_row("/Show-Follow-Ups [options]", "Show evidence, priority, and next action")
        table.add_row(
            "/Import-Follow-Up --evidence-file <path>",
            "Import checksum-verified reviewed evidence",
        )
        table.add_row(
            "/Recheck-Follow-Ups [options]",
            "Recheck deferred rows for new MAST coverage",
        )
        table.add_row("/Help", "Show this table")
        table.add_row("/Exit", "Close the persistent terminal")
        self._console.print(table)
        self._console.print(
            "[dim]Options after a slash command are passed to the canonical one-shot CLI. "
            "Use --json and --no-color for automation.[/dim]"
        )

    def _with_shared_options(self, args: list[str]) -> list[str]:
        if not any(value == "--db" or value.startswith("--db=") for value in args):
            args.extend(("--db", str(self.db_path)))
        if self.no_color and "--no-color" not in args:
            args.append("--no-color")
        return args

    def _invoke(self, label: str, fn: CommandFn, args: list[str]) -> int:
        try:
            with _OrbitAnimator(label, enabled=self.animate, stream=self._error):
                return int(fn(args))
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 2
            return code

    def _create(self, tokens: list[str], *, mode: str) -> CommandResult:
        if len(tokens) < 2:
            print(f"{tokens[0]} requires a positive target count N.", file=self._error)
            return CommandResult(2)
        try:
            count = int(tokens[1])
        except ValueError:
            print(f"{tokens[0]} target count must be an integer: {tokens[1]!r}", file=self._error)
            return CommandResult(2)
        if count < 1:
            print(f"{tokens[0]} target count must be at least 1.", file=self._error)
            return CommandResult(2)
        extra = tokens[2:]
        forbidden = ("--targets", "--mode")
        if any(
            option == name or option.startswith(f"{name}=")
            for option in extra
            for name in forbidden
        ):
            print(
                f"{tokens[0]} owns N and mode; do not pass --targets or --mode.",
                file=self._error,
            )
            return CommandResult(2)
        args = ["--targets", str(count), "--mode", mode, *extra]
        args = self._with_shared_options(args)
        label = (
            "sweeping the stellar catalog for new transits"
            if mode == "new"
            else "rephasing prior evidence for follow-up value"
        )
        return CommandResult(self._invoke(label, self._create_fn, args))

    def dispatch(self, line: str) -> CommandResult:
        """Parse and execute one slash command without duplicating business logic."""
        stripped = line.strip()
        if not stripped:
            return CommandResult(0)
        if stripped == "/":
            self.show_help()
            return CommandResult(0)
        try:
            tokens = shlex.split(stripped)
        except ValueError as exc:
            print(f"Command parse error: {exc}", file=self._error)
            return CommandResult(2)
        command = tokens[0].casefold()

        if command == "/new-search":
            return self._create(tokens, mode="new")
        if command == "/follow-up-search":
            return self._create(tokens, mode="follow-up")
        if command == "/create-new-search":
            args = self._with_shared_options(tokens[1:])
            return CommandResult(
                self._invoke(
                    "building the deterministic exact-target search",
                    self._create_fn,
                    args,
                )
            )
        if command in {"/run-new-search", "/run-search"}:
            args = self._with_shared_options(tokens[1:])
            return CommandResult(
                self._invoke("tracking exact targets through transit analysis", self._run_fn, args)
            )
        if command == "/show-follow-ups":
            args = self._with_shared_options(tokens[1:])
            return CommandResult(
                self._invoke("resolving the follow-up orbit", self._show_fn, args)
            )
        if command == "/import-follow-up":
            args = self._with_shared_options(tokens[1:])
            return CommandResult(
                self._invoke("validating prior transit evidence", self._import_fn, args)
            )
        if command == "/recheck-follow-ups":
            args = self._with_shared_options(tokens[1:])
            return CommandResult(
                self._invoke("sweeping MAST for new observing sectors", self._recheck_fn, args)
            )
        if command == "/help":
            self.show_help()
            return CommandResult(0)
        if command == "/exit":
            if len(tokens) != 1:
                print("/Exit does not accept arguments.", file=self._error)
                return CommandResult(2)
            return CommandResult(0, should_exit=True)

        print(
            f"Unknown command {tokens[0]!r}. Enter / or /Help for the command list.",
            file=self._error,
        )
        return CommandResult(2)

    def _configure_history(self) -> None:
        try:
            import readline
        except ImportError:
            print(
                "Warning: terminal history navigation is unavailable because "
                "the Python readline module is missing.",
                file=self._error,
            )
            return
        self._readline = readline
        readline.set_history_length(1000)
        readline.set_completer_delims(" \t\n")

        def _complete(text: str, state: int) -> str | None:
            matches = [
                command
                for command in _SLASH_COMMANDS
                if command.casefold().startswith(text.casefold())
            ]
            return matches[state] if state < len(matches) else None

        readline.set_completer(_complete)
        readline.parse_and_bind("tab: complete")
        if self.history_path.is_file():
            try:
                readline.read_history_file(str(self.history_path))
            except OSError as exc:
                print(f"Warning: could not read command history: {exc}", file=self._error)

    def _save_history(self) -> None:
        if self._readline is None:
            return
        try:
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            self._readline.write_history_file(str(self.history_path))
        except OSError as exc:
            print(f"Warning: could not persist command history: {exc}", file=self._error)

    def run_interactive(self, *, input_fn: Callable[[str], str] = input) -> int:
        """Stay active until the operator enters ``/Exit`` or closes stdin."""
        self._configure_history()
        self.print_banner()
        try:
            while True:
                try:
                    line = input_fn("EXO-Hunter> ")
                    result = self.dispatch(line)
                except KeyboardInterrupt:
                    print(
                        "\nCommand interrupted; EXO-Hunter remains active. Use /Exit to close.",
                        file=self._error,
                    )
                    continue
                except EOFError:
                    print("\nInput closed; exiting EXO-Hunter.", file=self._error)
                    return 0
                if result.should_exit:
                    self._console.print("[dim]EXO-Hunter session closed.[/dim]")
                    return result.exit_code
        finally:
            self._save_history()

    def run_commands(self, commands: Iterable[str]) -> int:
        """Run slash commands non-interactively, stopping on the first failure."""
        for line in commands:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                result = self.dispatch(stripped)
            except KeyboardInterrupt:
                print("EXO-Hunter command interrupted.", file=self._error)
                return 130
            if result.should_exit:
                return result.exit_code
            if result.exit_code != 0:
                return result.exit_code
        return 0


def _script_commands(path: Path, *, input_stream: TextIO) -> list[str]:
    if str(path) == "-":
        return input_stream.read().splitlines()
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise RuntimeError(f"Could not read EXO-Hunter command script {path}: {exc}") from exc


def exohunter(argv: Sequence[str] | None = None) -> int:
    """Run the persistent or scriptable EXO-Hunter terminal."""
    args = _parser().parse_args(argv)
    shell = HunterShell(
        db_path=args.db,
        history_path=args.history_file,
        no_color=args.no_color,
        no_animation=args.no_animation,
    )
    commands = list(args.command)
    if args.script is not None:
        try:
            commands.extend(_script_commands(args.script, input_stream=sys.stdin))
        except RuntimeError as exc:
            print(f"EXO-Hunter failed: {exc}", file=sys.stderr)
            return 2
    if commands:
        return shell.run_commands(commands)
    if not sys.stdin.isatty():
        print(
            "EXO-Hunter requires an interactive terminal, --command, or --script "
            "when stdin is redirected.",
            file=sys.stderr,
        )
        return 2
    return shell.run_interactive()


def exohunter_entry() -> None:
    """Installed console-script entry point."""
    raise SystemExit(exohunter())
