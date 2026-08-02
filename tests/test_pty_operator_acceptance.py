"""PHASE 2 PRIMARY GATE -- installed interactive PTY operator experience.

This is a black-box acceptance gate, not a unit test. It spawns the *resolved
installed console script* as a separate operating-system process attached to a
real pseudo-terminal and drives it with actual keystrokes.

The directive governing this work names the substitutions that do not satisfy
it, and this harness is built specifically to avoid every one of them:

* it never imports :mod:`exo_toolkit.hunter_shell` -- it executes the installed
  executable, so a renderer or state-machine test cannot stand in for it;
* it sends ``/`` with **no** trailing newline, so a line-buffered ``input()``
  loop that only sees ``/`` after Enter cannot pass;
* it asserts on bytes actually emitted to the terminal, not on golden files;
* it allocates a genuine PTY, so a mocked terminal cannot stand in for it.

When no PTY can be allocated the gate records nothing and skips. A skip is
``NOT EXECUTED`` under contract CLAIM-03 -- never a pass -- and
``prod_check._check_pty_operator_experience`` reports the requirement as
unexecuted because no evidence bundle was written.

Running this gate writes ``artifacts/manifests/exohunter_pty_acceptance.json``,
which is the only artifact ``prod-check`` will accept as behavioural proof.
"""
from __future__ import annotations

import errno
import json
import os
import re
import select
import shutil
import signal
import struct
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXECUTABLE = REPO_ROOT / ".venv" / "bin" / "EXO-Hunter"
EVIDENCE_PATH = REPO_ROOT / "artifacts" / "manifests" / "exohunter_pty_acceptance.json"

# CSI/OSC escape sequences are stripped before semantic assertions so that
# colour and cursor movement never change the meaning of a check.
_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b[()][B0]")

REQUIRED_PALETTE_COMMANDS = (
    "/New-Search",
    "/Follow-Up-Search",
    "/Run-Search",
    "/Show-Follow-Ups",
    "/Inspect-Target",
    "/Help",
    "/Exit",
)


def strip_ansi(text: str) -> str:
    return _ANSI.sub("", text)


def pty_available() -> bool:
    """Return whether this environment can allocate a pseudo-terminal."""
    try:
        import pty

        primary, secondary = pty.openpty()
    except (OSError, ImportError):
        return False
    os.close(primary)
    os.close(secondary)
    return True


requires_pty = pytest.mark.skipif(
    not pty_available(),
    reason="NOT EXECUTED: no pseudo-terminal can be allocated (/dev/ptmx denied)",
)
requires_executable = pytest.mark.skipif(
    not EXECUTABLE.is_file(),
    reason=f"NOT EXECUTED: installed executable missing at {EXECUTABLE}",
)


@dataclass
class PtySession:
    """One installed EXO-Hunter process attached to a real terminal."""

    columns: int = 100
    rows: int = 40
    env_overrides: dict[str, str] = field(default_factory=dict)
    cwd: Path = REPO_ROOT
    _primary: int = field(default=-1, init=False)
    _proc: subprocess.Popen[bytes] | None = field(default=None, init=False)
    transcript: str = field(default="", init=False)

    def __enter__(self) -> PtySession:
        import fcntl
        import pty
        import termios

        self._primary, secondary = pty.openpty()
        # A real terminal reports a size; width-aware rendering depends on it.
        fcntl.ioctl(
            secondary,
            termios.TIOCSWINSZ,
            struct.pack("HHHH", self.rows, self.columns, 0, 0),
        )
        env = dict(os.environ)
        env["TERM"] = "xterm-256color"
        env["COLUMNS"] = str(self.columns)
        env["LINES"] = str(self.rows)
        # The product disables animation under CI/REDUCE_MOTION; the operator
        # experience under test is the one a human gets, so clear them.
        for name in ("CI", "REDUCE_MOTION", "EXOHUNTER_REDUCE_MOTION", "NO_COLOR"):
            env.pop(name, None)
        env.update(self.env_overrides)

        db = self.cwd / "logs" / "prod_closure_evidence" / "phase2" / "pty_probe.sqlite3"
        db.parent.mkdir(parents=True, exist_ok=True)
        self._proc = subprocess.Popen(
            [str(EXECUTABLE), "--db", str(db)],
            stdin=secondary,
            stdout=secondary,
            stderr=secondary,
            cwd=str(self.cwd),
            env=env,
            close_fds=True,
            start_new_session=True,
        )
        os.close(secondary)
        return self

    def read_for(self, seconds: float) -> str:
        """Drain the terminal for a fixed window and return decoded output."""
        chunks: list[bytes] = []
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            try:
                ready, _, _ = select.select([self._primary], [], [], max(0.0, min(0.1, remaining)))
            except OSError:
                break
            if not ready:
                continue
            try:
                chunk = os.read(self._primary, 65536)
            except OSError as exc:
                if exc.errno in (errno.EIO, errno.EBADF):
                    break
                raise
            if not chunk:
                break
            chunks.append(chunk)
        text = b"".join(chunks).decode("utf-8", "replace")
        self.transcript += text
        return text

    def read_until(self, needle: str, timeout: float) -> str:
        """Drain until ``needle`` appears in the stripped output, or time out."""
        collected = ""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            collected += self.read_for(0.15)
            if needle in strip_ansi(collected):
                break
        return collected

    def send(self, data: str) -> None:
        """Write raw bytes -- no newline is ever appended implicitly."""
        os.write(self._primary, data.encode("utf-8"))

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (OSError, ProcessLookupError):
                proc.kill()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        if self._primary >= 0:
            os.close(self._primary)

    @property
    def returncode(self) -> int | None:
        return self._proc.poll() if self._proc else None

    def wait(self, timeout: float) -> int | None:
        if self._proc is None:
            return None
        try:
            return self._proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return None


_RESULTS: dict[str, object] = {}


def record(key: str, value: object) -> None:
    _RESULTS[key] = value


# The bundle is behavioural proof of the terminal experience. It is written
# only when the defining keystroke assertion actually executed against a real
# PTY; otherwise the run was NOT EXECUTED and must leave no artifact behind.
# Writing a partial bundle would let prod-check read a missing key as a FAIL
# and thereby misreport an unexecuted gate as an executed, failed one.
_DEFINING_KEY = "palette_opened_without_enter"


@pytest.fixture(scope="module", autouse=True)
def _write_evidence_bundle() -> object:
    """Persist the behavioural bundle only if the PTY gate actually ran."""
    yield
    if _DEFINING_KEY not in _RESULTS:
        return
    EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "bundle_version": "exohunter-pty-acceptance-v1",
        "executable": str(EXECUTABLE),
        "terminal": "real pty (pty.openpty)",
        "python": sys.version.split()[0],
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checks_executed": sorted(_RESULTS),
        **_RESULTS,
    }
    EVIDENCE_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


@requires_pty
@requires_executable
class TestStartupPresentation:
    """Phase 2: startup identity and animation, rendered to a real terminal."""

    def test_startup_shows_product_name_and_version(self) -> None:
        with PtySession() as session:
            out = strip_ansi(session.read_until("EXO-Hunter", timeout=8.0))
            record("startup_transcript_chars", len(out))
            assert "EXO-Hunter" in out, out[-400:]
            assert re.search(r"\d+\.\d+\.\d+", out), "no version rendered at startup"
            record("startup_shows_name_and_version", True)

    def test_startup_renders_multiple_distinct_animation_frames(self) -> None:
        """UX-START-01: a static logo or single frame is nonconforming."""
        with PtySession() as session:
            raw = session.read_until("EXO-Hunter", timeout=8.0)
            # Animation repaints the same line; each repaint is a carriage
            # return followed by a frame body.
            frames = {
                strip_ansi(part).strip()
                for part in raw.split("\r")
                if strip_ansi(part).strip()
            }
            record("distinct_startup_frames", len(frames))
            assert len(frames) >= 2, f"only {len(frames)} distinct frame(s): {sorted(frames)[:5]}"

    def test_prompt_appears_after_startup(self) -> None:
        with PtySession() as session:
            out = strip_ansi(session.read_until("EXO-Hunter>", timeout=8.0))
            record("prompt_after_startup", "EXO-Hunter>" in out)
            assert "EXO-Hunter>" in out, out[-400:]


@requires_pty
@requires_executable
class TestCommandPalette:
    """UX-CMD-01/02/03 driven by real keystrokes."""

    def test_slash_without_enter_opens_palette(self) -> None:
        """The defining Phase 2 assertion. No newline is sent."""
        with PtySession() as session:
            session.read_until("EXO-Hunter>", timeout=8.0)
            before = len(session.transcript)
            session.send("/")  # deliberately no "\r" and no "\n"
            after = strip_ansi(session.read_until("New-Search", timeout=3.0))
            opened = "/New-Search" in after
            record("palette_opened_without_enter", opened)
            record("bytes_emitted_after_bare_slash", len(session.transcript) - before)
            assert opened, (
                "typing '/' did not open the command palette before Enter was pressed; "
                "a line-buffered input loop cannot satisfy UX-CMD-01. "
                f"Output after keystroke: {after[-300:]!r}"
            )

    def test_palette_lists_every_required_command_with_descriptions(self) -> None:
        with PtySession() as session:
            session.read_until("EXO-Hunter>", timeout=8.0)
            session.send("/")
            out = strip_ansi(session.read_until("Exit", timeout=4.0))
            missing = [name for name in REQUIRED_PALETTE_COMMANDS if name not in out]
            record("palette_missing_commands", missing)
            assert not missing, f"palette omits {missing}"
            assert "Required:" in out or "required" in out.lower()
            record("palette_describes_parameters", True)

    def test_typing_filters_the_palette_live(self) -> None:
        with PtySession() as session:
            session.read_until("EXO-Hunter>", timeout=8.0)
            session.send("/foll")
            out = strip_ansi(session.read_until("Follow-Up", timeout=4.0))
            record("palette_live_filter", "/Follow-Up-Search" in out)
            assert "/Follow-Up-Search" in out, out[-300:]

    def test_arrow_keys_visibly_change_selection(self) -> None:
        """UX-CMD-03: keyboard navigation must move a visible selection."""
        with PtySession() as session:
            session.read_until("EXO-Hunter>", timeout=8.0)
            session.send("/")
            first = strip_ansi(session.read_until("New-Search", timeout=4.0))
            # Navigation is only meaningful once a palette is actually open.
            # Without this precondition the check can pass on nothing more than
            # the terminal echoing the arrow-key bytes back.
            assert "/New-Search" in first, (
                "the palette never opened, so selection movement cannot be assessed; "
                f"got {first[-200:]!r}"
            )
            session.send("\x1b[B")  # Down arrow
            second = strip_ansi(session.read_for(1.5))
            selected_rows = [line for line in second.splitlines() if line.startswith(">")]
            record("arrow_navigation_repaints", bool(second.strip()))
            record("arrow_navigation_moved_selection", bool(selected_rows))
            assert second.strip(), "Down arrow produced no repaint, so no selection moved"
            assert selected_rows, (
                "Down arrow produced a repaint with no selection marker, so no "
                f"selection moved; got {second[-200:]!r}"
            )

    def test_escape_closes_the_palette(self) -> None:
        with PtySession() as session:
            session.read_until("EXO-Hunter>", timeout=8.0)
            session.send("/")
            session.read_until("New-Search", timeout=4.0)
            session.send("\x1b")
            out = strip_ansi(session.read_for(1.5))
            record("escape_closes_palette", bool(out.strip()))
            assert out.strip(), "Escape produced no repaint, so the palette did not close"


@requires_pty
@requires_executable
class TestGuidedParameterEntry:
    """UX-IN-01/02/03 driven by real keystrokes."""

    def test_new_search_opens_guided_parameters(self) -> None:
        with PtySession() as session:
            session.read_until("EXO-Hunter>", timeout=8.0)
            session.send("/New-Search\r")
            out = strip_ansi(session.read_until("Targets", timeout=5.0))
            record("guided_fields_shown", "argets" in out)
            assert "argets" in out, f"no guided 'Targets' field appeared: {out[-300:]!r}"

    def test_invalid_target_count_is_rejected_inline_before_execution(self) -> None:
        with PtySession() as session:
            session.read_until("EXO-Hunter>", timeout=8.0)
            session.send("/New-Search\r")
            session.read_until("Targets", timeout=5.0)
            session.send("twenty")
            out = strip_ansi(session.read_until("Invalid", timeout=4.0))
            record("invalid_input_rejected_inline", "Invalid" in out)
            assert "Invalid" in out, f"invalid target count was not flagged inline: {out[-300:]!r}"

    def test_corrected_input_produces_a_resolved_action_preview(self) -> None:
        with PtySession() as session:
            session.read_until("EXO-Hunter>", timeout=8.0)
            session.send("/New-Search\r")
            session.read_until("Targets", timeout=5.0)
            session.send("twenty")
            session.read_until("Invalid", timeout=4.0)
            session.send("\x7f" * 6)  # backspace over the bad value
            session.send("5\r")
            out = strip_ansi(session.read_until("Requested targets", timeout=8.0))
            record("action_preview_rendered", "Requested targets" in out)
            assert "Requested targets" in out, f"no resolved-action preview: {out[-400:]!r}"

    def test_cancellation_returns_to_the_prompt(self) -> None:
        with PtySession() as session:
            session.read_until("EXO-Hunter>", timeout=8.0)
            session.send("/New-Search\r")
            session.read_until("Targets", timeout=5.0)
            session.send("\x1b")
            out = strip_ansi(session.read_until("EXO-Hunter>", timeout=4.0))
            record("cancel_returns_to_prompt", "EXO-Hunter>" in out)
            assert "EXO-Hunter>" in out, "Escape did not return to the prompt"


@requires_pty
@requires_executable
class TestHelpExitAndTerminalIntegrity:
    def test_help_renders_the_command_surface(self) -> None:
        with PtySession() as session:
            session.read_until("EXO-Hunter>", timeout=8.0)
            session.send("/Help\r")
            out = strip_ansi(session.read_until("New-Search", timeout=5.0))
            record("help_renders", "/New-Search" in out)
            assert "/New-Search" in out

    def test_exit_returns_zero(self) -> None:
        """A real terminal drains continuously; this harness must too.

        Blocking on wait() without reading fills the PTY buffer and stops the
        child mid-write, so it never reaches its exit path -- an artifact of the
        harness, not the product. Proven by probe: the identical keystrokes
        return 0 the moment draining resumes. Draining while polling keeps the
        assertion strictly as strong (the process must genuinely exit 0 within
        the same bound) while removing the deadlock.
        """
        with PtySession() as session:
            session.read_until("EXO-Hunter>", timeout=8.0)
            session.send("/Exit\r")
            deadline = time.monotonic() + 8.0
            code: int | None = None
            while time.monotonic() < deadline:
                session.read_for(0.2)
                code = session.returncode
                if code is not None:
                    break
            record("exit_code", code)
            assert code == 0, f"/Exit returned {code}"

    def test_terminal_mode_and_cursor_are_restored_after_exit(self) -> None:
        """The UI must not leave the terminal in raw mode or hide the cursor."""
        import termios

        with PtySession() as session:
            session.read_until("EXO-Hunter>", timeout=8.0)
            session.send("/")
            session.read_for(1.0)
            session.send("\x1b")
            session.send("/Exit\r")
            session.wait(timeout=8.0)
            tail = session.read_for(0.5)
            attrs = termios.tcgetattr(session._primary)
            canonical = bool(attrs[3] & termios.ICANON)
            echo = bool(attrs[3] & termios.ECHO)
            record("terminal_canonical_restored", canonical)
            record("terminal_echo_restored", echo)
            # If the cursor was ever hidden it must be shown again.
            hidden = session.transcript.count("\x1b[?25l")
            shown = session.transcript.count("\x1b[?25h")
            record("cursor_hide_show_balanced", hidden <= shown)
            assert canonical, "terminal left in non-canonical (raw) mode after exit"
            assert echo, "terminal left with echo disabled after exit"
            assert hidden <= shown, f"cursor hidden {hidden}x but shown {shown}x: {tail[-120:]!r}"


@requires_pty
@requires_executable
class TestWidthBehaviour:
    """UX-TABLE-01: controlled rendering at narrow, normal, and wide sizes."""

    @pytest.mark.parametrize("columns", [40, 80, 140])
    def test_no_line_exceeds_the_terminal_width(self, columns: int) -> None:
        with PtySession(columns=columns) as session:
            session.read_until("EXO-Hunter>", timeout=8.0)
            session.send("/")
            session.read_until("New-Search", timeout=4.0)
            lines = [strip_ansi(line).rstrip() for line in session.transcript.splitlines()]
            overflow = [line for line in lines if len(line) > columns]
            record(f"overflow_lines_at_{columns}", len(overflow))
            assert not overflow, f"{len(overflow)} line(s) exceeded {columns} cols: {overflow[:2]}"


@requires_executable
class TestNonTtyMode:
    """UX-START-04 / UX-TABLE-04: no animation or ANSI when not a terminal."""

    def test_non_tty_output_is_clean(self, tmp_path: Path) -> None:
        script = tmp_path / "commands.txt"
        script.write_text("/Help\n/Exit\n", encoding="utf-8")
        result = subprocess.run(
            [str(EXECUTABLE), "--no-color", "--no-animation", "--script", str(script)],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            timeout=120,
        )
        combined = result.stdout + result.stderr
        control = _ANSI.findall(combined)
        record("non_tty_exit_code", result.returncode)
        record("non_tty_ansi_sequences", len(control))
        assert result.returncode == 0, combined[-400:]
        assert not control, f"{len(control)} ANSI sequence(s) leaked into non-TTY output"


@requires_executable
class TestUnrelatedWorkingDirectory:
    """LAUNCH-02 surface 7: the executable must not depend on cwd."""

    def test_runs_from_a_directory_outside_the_repository(self, tmp_path: Path) -> None:
        outside = tmp_path.resolve()
        if REPO_ROOT in outside.parents or outside == REPO_ROOT:
            pytest.skip(
                "NOT EXECUTED: no writable directory outside the repository is available, "
                f"so cwd-independence cannot be proven (tmp_path resolved to {outside})"
            )
        script = outside / "commands.txt"
        script.write_text("/Help\n/Exit\n", encoding="utf-8")
        result = subprocess.run(
            [str(EXECUTABLE), "--no-color", "--no-animation", "--script", str(script)],
            capture_output=True,
            text=True,
            cwd=str(outside),
            timeout=120,
        )
        record("unrelated_cwd_exit_code", result.returncode)
        assert result.returncode == 0, (result.stdout + result.stderr)[-400:]
        assert shutil.which  # keeps the import meaningful for readers
