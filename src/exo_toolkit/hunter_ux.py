"""Shared EXO-Hunter CLI interaction contract.

This module is a presentation and validation layer only. Per
``docs/CLI_UX_SPEC.md`` §12 it must not duplicate candidate selection,
scientific scoring, execution, persistence, provenance, or business rules --
those stay in :mod:`exo_toolkit.hunter_cli` and
:mod:`exo_toolkit.search_lifecycle`.

It supplies three things the specification requires:

* ``COMMAND_SPECS`` -- the described command surface behind the ``/`` palette
  (UX-CMD-02) and ``/Help``, so both render from one registry rather than two
  hand-maintained tables that can drift apart.
* canonical field validators (UX-IN-03/UX-IN-04) used by both the interactive
  guided editor and the scriptable argument path, so a value rejected in one
  is rejected identically in the other.
* width-aware table rendering (UX-TABLE-01).
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

SPEC_VERSION = "HUNTER-CLI-UX-2026-07-30.3"


class Key(Enum):
    """One decoded keypress.

    Character-at-a-time input is what lets ``/`` open the palette before the
    operator presses Enter (UX-CMD-01); a line-buffered read cannot observe it.
    Decoding is kept here, separate from terminal I/O, so the whole interaction
    state machine is testable without allocating a pseudo-terminal.
    """

    CHAR = "char"
    ENTER = "enter"
    ESCAPE = "escape"
    BACKSPACE = "backspace"
    TAB = "tab"
    SHIFT_TAB = "shift_tab"
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"
    INTERRUPT = "interrupt"
    EOF = "eof"


@dataclass(frozen=True)
class KeyEvent:
    """A decoded key, carrying its literal character when ``key`` is CHAR."""

    key: Key
    char: str = ""


# Escape sequences this UI recognises. Anything else beginning with ESC is
# consumed and discarded rather than leaking raw bytes into the input buffer.
_ESCAPE_SEQUENCES = {
    "[A": Key.UP,
    "[B": Key.DOWN,
    "[C": Key.RIGHT,
    "[D": Key.LEFT,
    "[Z": Key.SHIFT_TAB,
    "OA": Key.UP,
    "OB": Key.DOWN,
    "OC": Key.RIGHT,
    "OD": Key.LEFT,
}


class KeyDecoder:
    """Turn a byte stream into key events.

    A terminal may deliver an escape sequence in pieces, so partial input is
    buffered until it either completes or is proven to be a bare Escape.
    """

    def __init__(self) -> None:
        self._pending = ""

    @property
    def pending(self) -> bool:
        """Whether an incomplete sequence is buffered awaiting more bytes."""
        return bool(self._pending)

    def feed(self, data: str) -> list[KeyEvent]:
        """Decode ``data``, returning every complete key event it contains."""
        self._pending += data
        events: list[KeyEvent] = []
        while self._pending:
            char = self._pending[0]
            if char == "\x1b":
                consumed, event = self._decode_escape(self._pending)
                if consumed == 0:
                    break  # incomplete sequence; wait for more bytes
                self._pending = self._pending[consumed:]
                if event is not None:
                    events.append(event)
                continue
            self._pending = self._pending[1:]
            events.append(self._decode_plain(char))
        return events

    def flush(self) -> list[KeyEvent]:
        """Resolve a trailing lone ESC as the Escape key.

        Called when no further bytes arrived within the escape timeout, which
        is how a terminal distinguishes Escape from the start of an arrow key.
        """
        if self._pending == "\x1b":
            self._pending = ""
            return [KeyEvent(Key.ESCAPE)]
        pending, self._pending = self._pending, ""
        return [self._decode_plain(char) for char in pending if char != "\x1b"]

    @staticmethod
    def _decode_plain(char: str) -> KeyEvent:
        if char in "\r\n":
            return KeyEvent(Key.ENTER)
        if char in ("\x7f", "\x08"):
            return KeyEvent(Key.BACKSPACE)
        if char == "\t":
            return KeyEvent(Key.TAB)
        if char == "\x03":
            return KeyEvent(Key.INTERRUPT)
        if char == "\x04":
            return KeyEvent(Key.EOF)
        return KeyEvent(Key.CHAR, char)

    @staticmethod
    def _decode_escape(buffer: str) -> tuple[int, KeyEvent | None]:
        """Return (bytes consumed, event). Zero consumed means "need more"."""
        if len(buffer) == 1:
            return 0, None
        for prefix, key in _ESCAPE_SEQUENCES.items():
            if buffer[1:].startswith(prefix):
                return 1 + len(prefix), KeyEvent(key)
        # A CSI sequence still being delivered: "\x1b[" with nothing after it.
        if buffer[1] in "[O" and len(buffer) == 2:
            return 0, None
        # ESC immediately followed by an ordinary character is Escape, then
        # that character (how most terminals send Alt-<key>).
        return 1, KeyEvent(Key.ESCAPE)


@dataclass
class PaletteState:
    """Live filtering and selection for the ``/`` command palette (UX-CMD-03).

    Pure state: it never touches a terminal, so every navigation rule is
    verifiable without a pseudo-terminal.
    """

    query: str = "/"
    index: int = 0

    def matches(self) -> tuple[CommandSpec, ...]:
        return filter_commands(self.query)

    def clamp(self) -> None:
        count = len(self.matches())
        if count == 0:
            self.index = 0
        else:
            self.index = max(0, min(self.index, count - 1))

    def move(self, delta: int) -> None:
        """Move the selection, wrapping at both ends."""
        count = len(self.matches())
        if count:
            self.index = (self.index + delta) % count

    def type_char(self, char: str) -> None:
        self.query += char
        self.index = 0

    def backspace(self) -> bool:
        """Delete one character; return False once the leading ``/`` is gone."""
        self.query = self.query[:-1]
        self.index = 0
        return bool(self.query)

    def selected(self) -> CommandSpec | None:
        matches = self.matches()
        if not matches:
            return None
        self.clamp()
        return matches[self.index]


class ValidationError(ValueError):
    """One operator-facing validation failure.

    Carries the concise sentinel text UX-IN-03 requires. Raw ``argparse`` usage
    dumps are explicitly not the normal interactive error response (UX-IN-04).
    """


@dataclass(frozen=True)
class FieldSpec:
    """One guided parameter field (UX-IN-01/UX-IN-02)."""

    name: str
    description: str
    required: bool = False
    default: str | None = None
    choices: tuple[str, ...] = ()
    validator: Callable[[str], Any] | None = None

    def label(self) -> str:
        """Return the bracketed value hint shown in the guided editor."""
        if self.choices:
            return f"[{'|'.join(self.choices)}]"
        if self.default is not None:
            return f"[{self.default}]"
        return "[required]" if self.required else "[optional]"


@dataclass(frozen=True)
class CommandSpec:
    """One described palette entry (UX-CMD-02)."""

    name: str
    summary: str
    fields: tuple[FieldSpec, ...] = ()
    aliases: tuple[str, ...] = ()
    requires_pending_search: bool = False

    @property
    def required_fields(self) -> tuple[FieldSpec, ...]:
        return tuple(item for item in self.fields if item.required)

    @property
    def optional_fields(self) -> tuple[FieldSpec, ...]:
        return tuple(item for item in self.fields if not item.required)

    def required_names(self) -> str:
        names = [item.name for item in self.required_fields]
        return ", ".join(names) if names else "none"

    def optional_names(self) -> str:
        names = [item.name for item in self.optional_fields]
        return ", ".join(names) if names else "none"


def validate_target_count(raw: str) -> int:
    """Validate the ``targets`` field.

    The exact sentinel strings below are the ones UX-IN-03 specifies.
    """
    text = raw.strip()
    if not text:
        raise ValidationError("Invalid - enter a positive whole number.")
    try:
        value = int(text)
    except ValueError:
        raise ValidationError("Invalid - enter a positive whole number.") from None
    if value <= 0:
        raise ValidationError("Invalid - targets must be greater than zero.")
    return value


def validate_positive_float(raw: str) -> float:
    """Validate an optional positive real-valued limit."""
    text = raw.strip()
    try:
        value = float(text)
    except ValueError:
        raise ValidationError("Invalid - enter a positive number.") from None
    if value <= 0:
        raise ValidationError("Invalid - enter a value greater than zero.")
    return value


def validate_target_reference(raw: str) -> str:
    """Validate an ``/Inspect-Target`` argument: a rank or a target identifier."""
    text = raw.strip()
    if not text:
        raise ValidationError("Invalid - enter a rank number or a target identifier.")
    if text.isdigit():
        if int(text) <= 0:
            raise ValidationError("Invalid - rank must be greater than zero.")
        return text
    if len(text) < 2:
        raise ValidationError("Invalid - enter a rank number or a target identifier.")
    return text


def validate_choice(raw: str, choices: Sequence[str]) -> str:
    """Validate an enumerated field (UX-IN-03 enumeration rule)."""
    text = raw.strip()
    if text not in choices:
        raise ValidationError(f"Invalid - choose one of: {', '.join(choices)}.")
    return text


_TARGETS_FIELD = FieldSpec(
    name="targets",
    description="How many targets to select and freeze.",
    required=True,
    default="20",
    validator=validate_target_count,
)
_MAX_DOWNLOAD_FIELD = FieldSpec(
    name="max-download-gb",
    description="Upper bound on total acquisition size, in gigabytes.",
    validator=validate_positive_float,
)

COMMAND_SPECS: tuple[CommandSpec, ...] = (
    CommandSpec(
        name="/New-Search",
        summary="Select and freeze the best available never-before-searched targets.",
        fields=(_TARGETS_FIELD, _MAX_DOWNLOAD_FIELD),
    ),
    CommandSpec(
        name="/Follow-Up-Search",
        summary="Select and freeze the highest-value previously-searched targets.",
        fields=(_TARGETS_FIELD, _MAX_DOWNLOAD_FIELD),
    ),
    CommandSpec(
        name="/Run-Search",
        summary="Execute or resume the exact frozen manifest without regenerating it.",
        aliases=("/Run-New-Search",),
        requires_pending_search=True,
    ),
    CommandSpec(
        name="/Show-Follow-Ups",
        summary="Show follow-up evidence, priority, and the recommended next action.",
        fields=(
            FieldSpec(
                name="status",
                description="Which registry dispositions to list.",
                default="open",
                choices=("open", "scheduled", "completed", "deferred", "all"),
            ),
        ),
    ),
    CommandSpec(
        name="/Inspect-Target",
        summary="Show full identity, score components, provenance, and prior-search evidence.",
        fields=(
            FieldSpec(
                name="rank-or-id",
                description="Manifest rank number, or a target/canonical identifier.",
                required=True,
                validator=validate_target_reference,
            ),
        ),
    ),
    CommandSpec(
        name="/Create-New-Search",
        summary="Lower-level creation command taking explicit --targets and --mode.",
    ),
    CommandSpec(
        name="/Import-Follow-Up",
        summary="Import one checksum-verified reviewed prior result.",
        fields=(
            FieldSpec(
                name="evidence-file",
                description="Path to the reviewed evidence JSON document.",
                required=True,
            ),
        ),
    ),
    CommandSpec(
        name="/Recheck-Follow-Ups",
        summary="Recheck deferred follow-ups for newly available MAST sectors.",
    ),
    CommandSpec(
        name="/Help",
        summary="Show the full command surface with parameters.",
    ),
    CommandSpec(
        name="/Exit",
        summary="Close the persistent terminal.",
    ),
)


def command_index() -> dict[str, CommandSpec]:
    """Return a case-folded lookup covering canonical names and aliases."""
    index: dict[str, CommandSpec] = {}
    for spec in COMMAND_SPECS:
        index[spec.name.casefold()] = spec
        for alias in spec.aliases:
            index[alias.casefold()] = spec
    return index


def filter_commands(query: str) -> tuple[CommandSpec, ...]:
    """Return palette entries matching a live filter (UX-CMD-03).

    ``query`` may include the leading slash the operator has already typed.
    Matching is case-insensitive over both the command name and its summary so
    that typing a concept ("follow", "freeze") finds the command.
    """
    text = query.strip().lstrip("/").casefold()
    if not text:
        return COMMAND_SPECS
    prefix = tuple(
        spec for spec in COMMAND_SPECS if spec.name.lstrip("/").casefold().startswith(text)
    )
    if prefix:
        return prefix
    return tuple(
        spec
        for spec in COMMAND_SPECS
        if text in spec.name.casefold() or text in spec.summary.casefold()
    )


def truncate_cell(value: str, width: int) -> str:
    """Truncate to ``width`` with a visible marker (UX-TABLE-01)."""
    if width <= 0:
        return ""
    if len(value) <= width:
        return value
    if width == 1:
        return "…"
    return value[: width - 1] + "…"


@dataclass(frozen=True)
class Column:
    """One results-table column with a stable width budget."""

    header: str
    key: str
    width: int
    priority: int = 0
    """Lower numbers survive width pressure first; rank/identity use 0."""


DEFAULT_RESULT_COLUMNS: tuple[Column, ...] = (
    Column("Rank", "rank", 5, priority=0),
    Column("Target", "target_id", 16, priority=0),
    Column("Score", "ranking_score", 8, priority=1),
    Column("Status", "search_status", 14, priority=2),
    Column("Class", "object_classification", 10, priority=3),
    Column("Reason", "selection_reason", 34, priority=4),
)


def select_columns(columns: Sequence[Column], terminal_width: int) -> tuple[Column, ...]:
    """Drop lowest-priority columns until the table fits ``terminal_width``.

    Rank and identity columns carry priority 0 and are never dropped, satisfying
    UX-TABLE-01's "preserve rank and identity visibility" requirement even at
    very small widths.
    """
    chosen = list(columns)
    separator = 1

    def total(items: Sequence[Column]) -> int:
        if not items:
            return 0
        return sum(item.width for item in items) + separator * (len(items) - 1)

    while len(chosen) > 1 and total(chosen) > terminal_width:
        droppable = [item for item in chosen if item.priority > 0]
        if not droppable:
            break
        victim = max(droppable, key=lambda item: item.priority)
        chosen.remove(victim)
    return tuple(chosen)


def render_results_table(
    rows: Iterable[Mapping[str, Any]],
    *,
    terminal_width: int = 80,
    columns: Sequence[Column] = DEFAULT_RESULT_COLUMNS,
) -> str:
    """Render a width-aware fixed-column table (UX-TABLE-01).

    Every cell is truncated to its column budget, so a long selection reason
    can never cause uncontrolled multi-line wrapping.
    """
    chosen = select_columns(columns, terminal_width)
    lines = [" ".join(truncate_cell(item.header, item.width).ljust(item.width) for item in chosen)]
    lines.append(" ".join("-" * item.width for item in chosen))
    for row in rows:
        cells = []
        for item in chosen:
            raw = row.get(item.key)
            text = "" if raw is None else str(raw)
            cells.append(truncate_cell(text, item.width).ljust(item.width))
        lines.append(" ".join(cells))
    return "\n".join(line.rstrip() for line in lines)


def render_palette(
    query: str = "", *, terminal_width: int = 80, selected_index: int | None = None
) -> str:
    """Render the searchable command palette (UX-CMD-01/UX-CMD-02).

    ``selected_index`` marks the highlighted entry so Up/Down navigation is
    visible in the rendered text itself (UX-CMD-03). Every line is truncated to
    ``terminal_width`` so the palette cannot wrap uncontrollably in a narrow
    terminal.
    """
    matches = filter_commands(query)
    header = "EXO-Hunter commands" if not query.strip("/") else f"Commands matching {query!r}"
    lines = [header, "-" * min(len(header), max(terminal_width, 10))]
    if not matches:
        lines.append(f"No command matches {query!r}. Press Escape to close, or enter /Help.")
        return "\n".join(lines)
    for position, spec in enumerate(matches):
        marker = ">" if selected_index is not None and position == selected_index else " "
        lines.append(f"{marker} {spec.name}")
        lines.append(f"    {spec.summary}")
        lines.append(f"    Required: {spec.required_names()}")
        lines.append(f"    Optional: {spec.optional_names()}")
        if spec.aliases:
            lines.append(f"    Also: {', '.join(spec.aliases)}")
    lines.append("")
    lines.append("Type to filter - Up/Down to navigate - Enter to select - Escape to close.")
    width = max(terminal_width, 20)
    return "\n".join(truncate_cell(line, width) for line in lines)


def render_action_preview(preview: Mapping[str, Any]) -> str:
    """Render the resolved-action preview (CLI/UX spec §8).

    Only fields the caller actually resolved are shown; this function never
    invents a source, freshness, or estimate it was not given, per UX-START-03
    and UX-RUN-02's prohibition on fabricated data states.
    """
    order = (
        ("mode", "Mode"),
        ("requested_targets", "Requested targets"),
        ("scientific_constraints", "Scientific constraints"),
        ("primary_sources", "Primary sources"),
        ("source_freshness", "Source freshness"),
        ("history_freshness", "Cross-project history freshness"),
        ("estimated_universe", "Estimated discovery universe"),
        ("estimated_storage", "Estimated storage"),
        ("estimated_compute", "Estimated compute"),
        ("output_behavior", "Output behavior"),
    )
    width = max(len(label) for _, label in order) + 1
    lines = ["Resolved action preview", "-" * 23]
    for key, label in order:
        if key in preview:
            value = preview[key]
            shown = "not resolved" if value is None else str(value)
            lines.append(f"{(label + ':').ljust(width)} {shown}")
    lines.append("")
    lines.append("Confirm, edit, or cancel.")
    return "\n".join(lines)


@dataclass
class GuidedEntry:
    """Collect and validate one command's fields (UX-IN-01..UX-IN-03).

    Values are validated as they are supplied; an invalid value never advances
    and never reaches the canonical business layer.
    """

    spec: CommandSpec
    values: dict[str, str] = field(default_factory=dict)

    def set_value(self, name: str, raw: str) -> None:
        """Validate and store one field, raising ``ValidationError`` if invalid."""
        for item in self.spec.fields:
            if item.name != name:
                continue
            text = raw.strip()
            if not text:
                if item.required:
                    raise ValidationError(f"Invalid - {item.name} is required.")
                self.values.pop(name, None)
                return
            if item.choices:
                validate_choice(text, item.choices)
            elif item.validator is not None:
                item.validator(text)
            self.values[name] = text
            return
        raise ValidationError(f"Invalid - {self.spec.name} has no field named {name!r}.")

    def missing_required(self) -> tuple[str, ...]:
        """Return required field names still unset, after applying defaults."""
        missing = []
        for item in self.spec.required_fields:
            if item.name in self.values:
                continue
            if item.default is not None:
                continue
            missing.append(item.name)
        return tuple(missing)

    def is_executable(self) -> bool:
        """Enter executes only when all required fields are valid (UX-IN-02)."""
        return not self.missing_required()

    def to_argv(self) -> list[str]:
        """Render the collected fields as canonical CLI arguments (CLI-03).

        The interactive path produces exactly the argument vector the scriptable
        path accepts, so both reach the same validators and the same canonical
        pipeline.
        """
        argv: list[str] = []
        for item in self.spec.fields:
            value = self.values.get(item.name, item.default if item.required else None)
            if value is None:
                continue
            argv.extend((f"--{item.name}", value))
        return argv

    def render(self) -> str:
        """Render the inline guided editor (UX-IN-01)."""
        width = max(len(item.name) for item in self.spec.fields) + 2 if self.spec.fields else 2
        lines = [self.spec.name, ""]
        for item in self.spec.fields:
            shown = self.values.get(item.name, item.label())
            marker = " " if item.name in self.values or not item.required else "*"
            lines.append(f"{marker}{item.name.ljust(width)}{shown}")
            lines.append(f"  {item.description}")
        lines.append("")
        lines.append("Tab/Shift-Tab move - Enter executes when valid - Escape cancels.")
        return "\n".join(lines)
