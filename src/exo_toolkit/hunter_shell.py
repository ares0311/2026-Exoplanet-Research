"""Persistent slash-command terminal for the canonical EXO-Hunter workflow."""
from __future__ import annotations

import argparse
import contextlib
import os
import select
import shlex
import shutil
import sys
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
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
    inspect_target,
    recheck_follow_ups,
    run_new_search,
    show_follow_ups,
)
from exo_toolkit.hunter_ux import (
    COMMAND_SPECS,
    CommandSpec,
    GuidedEntry,
    Key,
    KeyDecoder,
    KeyEvent,
    PaletteState,
    ValidationError,
    command_index,
    filter_commands,
    render_action_preview,
    render_palette,
    truncate_cell,
    validate_target_count,
)

try:  # POSIX only; Windows falls back to the line-buffered loop.
    import termios
    import tty

    _RAW_MODE_SUPPORTED = True
except ImportError:  # pragma: no cover - exercised only on non-POSIX
    _RAW_MODE_SUPPORTED = False

CommandFn = Callable[[Sequence[str] | None], int]

_SLASH_COMMANDS = tuple(
    sorted({spec.name for spec in COMMAND_SPECS} | {"/Run-New-Search", "/Create-New-Search"})
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
        # Clipped to the real terminal width so the motif cannot wrap onto a
        # second line in a narrow terminal (UX-TABLE-01 / CLI-UX section 11).
        width = max(20, shutil.get_terminal_size(fallback=(80, 24)).columns)
        while not self._stop.is_set():
            frame = _ANIMATION_FRAMES[index % len(_ANIMATION_FRAMES)]
            elapsed = time.monotonic() - self._started
            line = f"{frame}  {self._label}  {elapsed:5.1f}s"
            self._stream.write("\r" + truncate_cell(line, width))
            self._stream.flush()
            index += 1
            self._stop.wait(self._interval_seconds)

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=max(1.0, self._interval_seconds * 3))
            self._stream.write("\r" + (" " * 79) + "\r")
            self._stream.flush()


# Commands whose first guided field is supplied positionally by ``dispatch``
# rather than as a flag. Keeping this mapping here means the guided editor
# produces exactly the line the existing dispatcher already accepts, instead of
# introducing a second parsing path (CLI-03).
_POSITIONAL_FIRST_FIELD = {
    "/New-Search": "--targets",
    "/Follow-Up-Search": "--targets",
    "/Inspect-Target": "--rank-or-id",
}


def _command_line(spec: CommandSpec, argv: Sequence[str]) -> str:
    """Render a guided entry as the slash-command line ``dispatch`` expects."""
    pairs = list(zip(argv[::2], argv[1::2], strict=False))
    positional = _POSITIONAL_FIRST_FIELD.get(spec.name)
    tokens: list[str] = [spec.name]
    if positional is not None:
        for flag, value in pairs:
            if flag == positional:
                tokens.append(value)
        pairs = [(flag, value) for flag, value in pairs if flag != positional]
    for flag, value in pairs:
        tokens.extend((flag, value))
    return " ".join(tokens)


@contextlib.contextmanager
def _raw_terminal(fd: int, stream: TextIO) -> Iterator[None]:
    """Put ``fd`` into cbreak mode for the duration of the block.

    Canonical (line-buffered) mode withholds every byte until Enter, so ``/``
    could never open the palette on the keystroke alone (UX-CMD-01). cbreak is
    preferred over full raw mode because it leaves signal handling intact, so
    Ctrl-C still interrupts.

    The original attributes are always restored and the cursor is always shown
    again, including on exception, which is what keeps the terminal usable
    after a crash or cancellation (CLI/UX spec section 11).
    """
    saved = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)
        with contextlib.suppress(OSError, ValueError):
            stream.write("\x1b[?25h")  # show cursor
            stream.flush()


class _KeyStream:
    """Decode keypresses from a file descriptor already in cbreak mode."""

    # A lone ESC and the start of an arrow-key sequence are the same first
    # byte, so a bare Escape is only recognised after this quiet period.
    ESCAPE_TIMEOUT = 0.06

    def __init__(self, fd: int) -> None:
        self._fd = fd
        self._decoder = KeyDecoder()
        self._queue: list[KeyEvent] = []

    def next_event(self) -> KeyEvent:
        """Block until one key event is available. Raises EOFError at EOF."""
        while True:
            if self._queue:
                return self._queue.pop(0)
            timeout = self.ESCAPE_TIMEOUT if self._decoder.pending else None
            ready, _, _ = select.select([self._fd], [], [], timeout)
            if not ready:
                self._queue.extend(self._decoder.flush())
                continue
            data = os.read(self._fd, 1024)
            if not data:
                raise EOFError
            self._queue.extend(self._decoder.feed(data.decode("utf-8", "replace")))


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
        inspect_fn: CommandFn = inspect_target,
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
        self._inspect_fn = inspect_fn
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

    def show_palette(self, query: str = "") -> None:
        """Open the searchable command palette (UX-CMD-01/UX-CMD-02/UX-CMD-03).

        Rendered from the single ``COMMAND_SPECS`` registry so the palette and
        ``/Help`` can never drift apart.
        """
        width = self._console.width if self._console.width else 80
        # Printed through the console (not a bare print) so palette output honors
        # --no-color and is captured by the same stream as every other command.
        self._console.print(render_palette(query, terminal_width=width), highlight=False)

    def show_help(self) -> None:
        """Show the complete slash-command surface with parameter shapes."""
        table = Table(title="EXO-Hunter commands", show_lines=False)
        table.add_column("Command", style="cyan", no_wrap=True)
        table.add_column("Canonical action")
        table.add_column("Required", no_wrap=True)
        table.add_column("Optional", no_wrap=True)
        for spec in COMMAND_SPECS:
            table.add_row(
                spec.name,
                spec.summary,
                spec.required_names(),
                spec.optional_names(),
            )
        self._console.print(table)
        self._console.print(
            "[dim]Enter / to open the searchable palette, or /<text> to filter it. "
            "Options after a slash command are passed to the canonical one-shot CLI; "
            "use --json and --no-color for automation.[/dim]"
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
            print(
                f"{tokens[0]} requires a positive target count N.\n"
                f"  Usage: {tokens[0]} <N> [options]",
                file=self._error,
            )
            return CommandResult(2)
        # UX-IN-04: the interactive path uses the same canonical validator as the
        # scriptable path, so a value rejected here is rejected identically there.
        try:
            count = validate_target_count(tokens[1])
        except ValidationError as exc:
            print(f"{tokens[0]} targets: {tokens[1]}\n  {exc}", file=self._error)
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
            self.show_palette()
            return CommandResult(0)
        try:
            tokens = shlex.split(stripped)
        except ValueError as exc:
            print(f"Command parse error: {exc}", file=self._error)
            return CommandResult(2)
        command = tokens[0].casefold()

        # A bare "/text" that is not itself a command, but does prefix-match one
        # or more real commands, filters the palette so discovery never requires
        # /Help (UX-CMD-01). A token matching nothing stays a hard error: a typo
        # must not be silently absorbed into a success exit code.
        known = {spec.name.casefold() for spec in COMMAND_SPECS} | set(command_index())
        if (
            len(tokens) == 1
            and command.startswith("/")
            and command not in known
            and filter_commands(tokens[0])
        ):
            self.show_palette(tokens[0])
            return CommandResult(0)

        if command == "/inspect-target":
            if len(tokens) < 2:
                print(
                    "/Inspect-Target requires a rank number or target identifier.\n"
                    "  Usage: /Inspect-Target <rank-or-id>",
                    file=self._error,
                )
                return CommandResult(2)
            args = ["--rank-or-id", tokens[1], *tokens[2:]]
            args = self._with_shared_options(args)
            return CommandResult(
                self._invoke("resolving target identity and provenance", self._inspect_fn, args)
            )

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

    # ---------------------------------------------------------------- raw UI

    def terminal_width(self) -> int:
        """Best available terminal width, honouring COLUMNS and the real tty."""
        try:
            return max(20, os.get_terminal_size(self._output.fileno()).columns)
        except (OSError, ValueError, AttributeError):
            return max(20, shutil.get_terminal_size(fallback=(80, 24)).columns)

    def _emit(self, text: str = "") -> None:
        """Write one width-clipped line, using CRLF for cbreak mode.

        In cbreak mode a bare newline does not return the cursor to column
        zero, so every line ends CRLF. Clipping keeps output inside the
        terminal width (UX-TABLE-01) even at 40 columns.
        """
        width = self.terminal_width()
        for line in text.split("\n"):
            self._output.write(truncate_cell(line, width) + "\r\n")
        self._output.flush()

    def _raw_capable(self) -> bool:
        """Whether character-at-a-time input is possible on this terminal.

        Redirected or non-TTY streams fall back to the line-buffered loop, which
        keeps scripted and CI use working unchanged (UX-START-04).
        """
        if not _RAW_MODE_SUPPORTED:
            return False
        try:
            return self._input.isatty() and self._output.isatty()
        except (AttributeError, ValueError):
            return False

    def _render_palette(self, state: PaletteState) -> None:
        self._emit()
        self._emit(
            render_palette(
                state.query,
                terminal_width=self.terminal_width(),
                selected_index=state.index,
            )
        )

    def _palette_loop(self, keys: _KeyStream) -> CommandSpec | None:
        """Run the ``/`` palette until a command is chosen or it is dismissed.

        Returns the chosen command, or ``None`` when the operator closed the
        palette with Escape or backspaced past the leading slash.
        """
        state = PaletteState()
        self._render_palette(state)
        while True:
            event = keys.next_event()
            if event.key is Key.CHAR:
                state.type_char(event.char)
            elif event.key is Key.BACKSPACE:
                if not state.backspace():
                    self._emit("(palette closed)")
                    return None
            elif event.key is Key.UP:
                state.move(-1)
            elif event.key is Key.DOWN:
                state.move(1)
            elif event.key in (Key.ESCAPE, Key.INTERRUPT):
                self._emit("(palette closed)")
                return None
            elif event.key is Key.EOF:
                raise EOFError
            elif event.key is Key.ENTER:
                chosen = state.selected()
                if chosen is not None:
                    return chosen
                self._emit(f"No command matches {state.query!r}.")
            else:
                continue
            self._render_palette(state)

    def _guided_entry(self, keys: _KeyStream, spec: CommandSpec) -> list[str] | None:
        """Collect a command's fields with live validation (UX-IN-01..03).

        Returns the canonical argument vector, or ``None`` if cancelled. The
        validators are the shared ones from :mod:`exo_toolkit.hunter_ux`, so a
        value rejected here is rejected identically on the scriptable path
        (UX-IN-04).
        """
        entry = GuidedEntry(spec)
        if not spec.fields:
            return []
        self._emit()
        self._emit(entry.render())
        for field_spec in spec.fields:
            buffer = ""
            self._emit()
            self._emit(f"{field_spec.name} {field_spec.label()}  -- {field_spec.description}")
            while True:
                event = keys.next_event()
                if event.key in (Key.ESCAPE, Key.INTERRUPT):
                    self._emit("(cancelled)")
                    return None
                if event.key is Key.EOF:
                    raise EOFError
                if event.key is Key.BACKSPACE:
                    buffer = buffer[:-1]
                elif event.key is Key.CHAR:
                    buffer += event.char
                elif event.key in (Key.ENTER, Key.TAB):
                    try:
                        entry.set_value(field_spec.name, buffer)
                    except ValidationError as exc:
                        self._emit(f"{field_spec.name}: {buffer}")
                        self._emit(f"  {exc}")
                        continue
                    # UX-IN-02: Enter executes as soon as every required field
                    # is valid; Tab is what moves on to an optional field.
                    if event.key is Key.ENTER and not entry.missing_required():
                        return entry.to_argv()
                    break
                else:
                    continue
                # UX-IN-03: validate during entry so an invalid value is
                # flagged before Enter is ever pressed.
                self._emit(f"{field_spec.name}: {buffer}")
                if buffer.strip():
                    try:
                        entry.set_value(field_spec.name, buffer)
                    except ValidationError as exc:
                        self._emit(f"  {exc}")
        missing = entry.missing_required()
        if missing:
            self._emit(f"Invalid - {', '.join(missing)} is required.")
            return None
        return entry.to_argv()

    def _confirm_preview(self, keys: _KeyStream, spec: CommandSpec, argv: Sequence[str]) -> bool:
        """Show the resolved-action preview and wait for confirm or cancel.

        Only values actually resolved are shown; nothing here invents a source,
        freshness, or estimate (UX-START-03, UX-RUN-02).
        """
        values = dict(zip(argv[::2], argv[1::2], strict=False))
        preview: dict[str, object] = {
            "mode": "New" if spec.name == "/New-Search" else spec.name.lstrip("/"),
            "requested_targets": values.get("--targets", "not resolved"),
            "primary_sources": "MAST TIC catalogue; NASA Exoplanet Archive",
            "output_behavior": "freeze an exact target manifest; execute with /Run-Search",
        }
        if "--max-download-gb" in values:
            preview["estimated_storage"] = f"{values['--max-download-gb']} GB limit"
        self._emit()
        self._emit(render_action_preview(preview))
        self._emit("Press Enter to confirm, or Escape to cancel.")
        while True:
            event = keys.next_event()
            if event.key is Key.ENTER:
                return True
            if event.key in (Key.ESCAPE, Key.INTERRUPT):
                self._emit("(cancelled)")
                return False
            if event.key is Key.EOF:
                raise EOFError

    def _run_raw_interactive(self) -> int:
        """Character-at-a-time operator loop (UX-CMD-01)."""
        fd = self._input.fileno()
        with _raw_terminal(fd, self._output):
            keys = _KeyStream(fd)
            buffer = ""
            self._output.write("EXO-Hunter> ")
            self._output.flush()
            while True:
                try:
                    event = keys.next_event()
                except EOFError:
                    self._emit()
                    self._emit("Input closed; exiting EXO-Hunter.")
                    return 0
                if event.key is Key.CHAR:
                    # The defining behaviour: a leading "/" opens the palette
                    # immediately, with no Enter required.
                    if not buffer and event.char == "/":
                        result = self._palette_flow(keys)
                        if result is not None:
                            return result
                        buffer = ""
                        self._output.write("EXO-Hunter> ")
                        self._output.flush()
                        continue
                    buffer += event.char
                    self._output.write(event.char)
                    self._output.flush()
                elif event.key is Key.BACKSPACE:
                    if buffer:
                        buffer = buffer[:-1]
                        self._output.write("\b \b")
                        self._output.flush()
                elif event.key is Key.ENTER:
                    self._emit()
                    line, buffer = buffer, ""
                    if line.strip():
                        outcome = self.dispatch(line)
                        if outcome.should_exit:
                            self._emit("EXO-Hunter session closed.")
                            return outcome.exit_code
                    self._output.write("EXO-Hunter> ")
                    self._output.flush()
                elif event.key is Key.INTERRUPT:
                    self._emit()
                    self._emit(
                        "Command interrupted; EXO-Hunter remains active. Use /Exit to close."
                    )
                    buffer = ""
                    self._output.write("EXO-Hunter> ")
                    self._output.flush()
                elif event.key is Key.EOF:
                    self._emit()
                    self._emit("Input closed; exiting EXO-Hunter.")
                    return 0

    def _palette_flow(self, keys: _KeyStream) -> int | None:
        """Palette -> guided entry -> preview -> canonical dispatch.

        Returns an exit code when the chosen command ends the session,
        otherwise ``None`` to return to the prompt.
        """
        spec = self._palette_loop(keys)
        if spec is None:
            return None
        if spec.name == "/Exit":
            self._emit("EXO-Hunter session closed.")
            return 0
        if spec.name == "/Help":
            self.show_help()
            return None
        argv = self._guided_entry(keys, spec)
        if argv is None:
            return None
        if spec.name in ("/New-Search", "/Follow-Up-Search") and not self._confirm_preview(
            keys, spec, argv
        ):
            return None
        # Presentation ends here: execution always re-enters the canonical
        # dispatcher, so interactive and scriptable paths share one pipeline.
        outcome = self.dispatch(_command_line(spec, argv))
        return outcome.exit_code if outcome.should_exit else None

    def run_interactive(self, *, input_fn: Callable[[str], str] = input) -> int:
        """Stay active until the operator enters ``/Exit`` or closes stdin."""
        if self._raw_capable():
            self.print_banner()
            try:
                return self._run_raw_interactive()
            finally:
                self._save_history()
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
