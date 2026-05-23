"""Track per-candidate field-level changes with timestamp and author.

JSON-backed changelog that records every field mutation with the old value,
new value, timestamp, and author.  Distinct from ``candidate_notes``
(free-text notes) and ``candidate_timeline`` (pipeline event sequence).

Public API
----------
ChangeEntry(field, old_value, new_value, timestamp, author)
ChangelogResult(tic_id, n_changes, entries, flag)
record_change(tic_id, field, old_val, new_val, *, author,
              store_path) -> ChangelogResult
get_changelog(tic_id, store_path) -> ChangelogResult
format_changelog_result(result) -> str
"""
from __future__ import annotations

import contextlib
import datetime
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ChangeEntry:
    field: str
    old_value: str | None
    new_value: str | None
    timestamp: str   # ISO-8601
    author: str


@dataclass(frozen=True)
class ChangelogResult:
    tic_id: int | str
    n_changes: int
    entries: tuple[ChangeEntry, ...]
    flag: str  # "OK" | "EMPTY" | "INVALID"


def _load_store(store_path: Path) -> dict:
    if store_path.exists():
        try:
            return json.loads(store_path.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_store(store_path: Path, data: dict) -> None:
    store_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=store_path.parent, suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w") as fh:
            json.dump(data, fh, indent=2)
        os.replace(tmp_path, store_path)
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise


def record_change(
    tic_id: int | str,
    field: str,
    old_val,
    new_val,
    *,
    author: str = "unknown",
    store_path: Path | str = Path("data/changelogs.json"),
) -> ChangelogResult:
    """Append a field-change entry to the changelog for *tic_id*.

    Args:
        tic_id: TIC identifier.
        field: Name of the field that changed.
        old_val: Previous value (will be JSON-stringified).
        new_val: New value (will be JSON-stringified).
        author: Name/ID of the person or process making the change.
        store_path: Path to the JSON store file.

    Returns:
        :class:`ChangelogResult` reflecting the post-append state.
    """
    if not field:
        return ChangelogResult(tic_id, 0, (), "INVALID")

    store_path = Path(store_path)
    key = str(tic_id)
    data = _load_store(store_path)
    if key not in data:
        data[key] = []

    entry = {
        "field": field,
        "old_value": None if old_val is None else str(old_val),
        "new_value": None if new_val is None else str(new_val),
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "author": author,
    }
    data[key].append(entry)
    _save_store(store_path, data)

    return _build_result(tic_id, data[key])


def get_changelog(
    tic_id: int | str,
    store_path: Path | str = Path("data/changelogs.json"),
) -> ChangelogResult:
    """Retrieve the full changelog for *tic_id*.

    Returns:
        :class:`ChangelogResult` with flag ``"EMPTY"`` if no entries found.
    """
    store_path = Path(store_path)
    key = str(tic_id)
    data = _load_store(store_path)
    entries = data.get(key, [])
    return _build_result(tic_id, entries)


def _build_result(tic_id, raw_entries: list) -> ChangelogResult:
    if not raw_entries:
        return ChangelogResult(tic_id, 0, (), "EMPTY")
    entries = tuple(
        ChangeEntry(
            field=e["field"],
            old_value=e.get("old_value"),
            new_value=e.get("new_value"),
            timestamp=e.get("timestamp", ""),
            author=e.get("author", "unknown"),
        )
        for e in raw_entries
    )
    return ChangelogResult(tic_id=tic_id, n_changes=len(entries), entries=entries, flag="OK")


def format_changelog_result(result: ChangelogResult) -> str:
    """Format changelog as Markdown."""
    lines = [
        f"## Candidate Changelog — TIC {result.tic_id}",
        "",
        f"- Total changes recorded: {result.n_changes}",
        f"- **Flag: {result.flag}**",
    ]
    if result.entries:
        lines += ["", "| Field | Old | New | Timestamp | Author |",
                  "|---|---|---|---|---|"]
        for e in result.entries[-10:]:
            ov = e.old_value or "—"
            nv = e.new_value or "—"
            ts = e.timestamp[:19] if e.timestamp else "—"
            lines.append(f"| {e.field} | {ov} | {nv} | {ts} | {e.author} |")
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="candidate_changelog_tracker",
        description="Track per-candidate field-level changes.",
    )
    parser.add_argument("--tic-id", type=int, required=True)
    parser.add_argument("--field", type=str, default=None)
    parser.add_argument("--old-val", type=str, default=None)
    parser.add_argument("--new-val", type=str, default=None)
    parser.add_argument("--author", type=str, default="cli")
    parser.add_argument("--store", type=str, default="data/changelogs.json")
    args = parser.parse_args(argv)

    if args.field:
        result = record_change(
            args.tic_id, args.field, args.old_val, args.new_val,
            author=args.author, store_path=Path(args.store)
        )
    else:
        result = get_changelog(args.tic_id, store_path=Path(args.store))
    print(format_changelog_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
