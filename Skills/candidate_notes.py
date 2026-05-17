"""Persistent free-text annotations for candidate TIC IDs.

Stores notes in a JSON file with atomic writes (tempfile rename).

Public API
----------
CandidateNotes(path)
    .add(tic_id, note, *, tag) -> None
    .get(tic_id) -> list[dict]
    .remove(tic_id) -> int
    .list_tic_ids() -> list[int]
    .search(query) -> list[dict]
    .summary() -> dict
"""
from __future__ import annotations

import contextlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path


class CandidateNotes:
    """Persistent per-TIC annotation store backed by JSON."""

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._data: dict[str, list[dict]] = {}
        if self._path.exists():
            try:
                raw = json.loads(self._path.read_text())
                self._data = {str(k): v for k, v in raw.items()}
            except (json.JSONDecodeError, AttributeError):
                self._data = {}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self._path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as fh:
                json.dump(self._data, fh, indent=2)
            os.replace(tmp, self._path)
        except Exception:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise

    def add(self, tic_id: int, note: str, *, tag: str = "") -> None:
        """Add a note for a TIC ID.

        Args:
            tic_id: TIC identifier.
            note: Free-text annotation.
            tag: Optional short tag (e.g. "followup", "suspicious").
        """
        key = str(int(tic_id))
        entry = {
            "note": str(note),
            "tag": str(tag),
            "added_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        if key not in self._data:
            self._data[key] = []
        self._data[key].append(entry)
        self._save()

    def get(self, tic_id: int) -> list[dict]:
        """Return all notes for a TIC ID (chronological order)."""
        return list(self._data.get(str(int(tic_id)), []))

    def remove(self, tic_id: int) -> int:
        """Delete all notes for a TIC ID.  Returns count removed."""
        key = str(int(tic_id))
        n = len(self._data.get(key, []))
        if n:
            del self._data[key]
            self._save()
        return n

    def list_tic_ids(self) -> list[int]:
        """Return all TIC IDs that have notes."""
        return sorted(int(k) for k in self._data)

    def search(self, query: str) -> list[dict]:
        """Return notes whose text contains ``query`` (case-insensitive).

        Each result dict includes ``tic_id`` alongside the note fields.
        """
        q = query.lower()
        results: list[dict] = []
        for key, entries in self._data.items():
            for entry in entries:
                if q in entry.get("note", "").lower() or q in entry.get("tag", "").lower():
                    results.append({"tic_id": int(key), **entry})
        return results

    def summary(self) -> dict:
        """Return aggregate statistics."""
        n_targets = len(self._data)
        n_notes = sum(len(v) for v in self._data.values())
        tags: dict[str, int] = {}
        for entries in self._data.values():
            for e in entries:
                t = e.get("tag") or ""
                if t:
                    tags[t] = tags.get(t, 0) + 1
        return {
            "n_targets": n_targets,
            "n_notes": n_notes,
            "tags": tags,
        }


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="candidate_notes",
        description="Manage free-text annotations for candidate TIC IDs.",
    )
    parser.add_argument("--db", default="data/candidate_notes.json", metavar="JSON")
    sub = parser.add_subparsers(dest="cmd")

    add_p = sub.add_parser("add")
    add_p.add_argument("tic_id", type=int)
    add_p.add_argument("note")
    add_p.add_argument("--tag", default="")

    get_p = sub.add_parser("get")
    get_p.add_argument("tic_id", type=int)

    rm_p = sub.add_parser("remove")
    rm_p.add_argument("tic_id", type=int)

    sub.add_parser("list")

    search_p = sub.add_parser("search")
    search_p.add_argument("query")

    sub.add_parser("summary")

    args = parser.parse_args(argv)
    notes = CandidateNotes(args.db)

    if args.cmd == "add":
        notes.add(args.tic_id, args.note, tag=args.tag)
        print(f"Added note for TIC {args.tic_id}.")
    elif args.cmd == "get":
        for e in notes.get(args.tic_id):
            print(f"[{e['added_at']}] [{e['tag']}] {e['note']}")
    elif args.cmd == "remove":
        n = notes.remove(args.tic_id)
        print(f"Removed {n} note(s) for TIC {args.tic_id}.")
    elif args.cmd == "list":
        print(notes.list_tic_ids())
    elif args.cmd == "search":
        print(notes.search(args.query))
    elif args.cmd == "summary":
        print(notes.summary())
    else:
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
