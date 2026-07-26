"""Manage a persistent watchlist of TIC IDs for follow-up transit scanning.

Stores a JSON watchlist in a user-specified file.  Entries carry an optional
note and a timestamp.  The watchlist integrates with ``batch_scan.py`` —
pass the watchlist file as the input TIC-ID list.

Public API
----------
Watchlist(path)
    .add(tic_id, note="")
    .remove(tic_id)
    .contains(tic_id) -> bool
    .list_ids() -> list[int]
    .entries() -> list[dict]   # [{tic_id, note, added_at}, ...]
    .clear()
    .summary() -> dict         # {n_entries, path}

Every mutating call (``add``/``remove``/``clear``) reloads the file fresh
under an exclusive ``fcntl`` lock immediately before writing, rather than
caching state from construction time — an earlier version cached the
loaded dict for the life of the ``Watchlist`` object, so a second
``Watchlist`` instance's write to the same file (a second CLI invocation,
a concurrent shard) could be silently lost to a later write from the
first, stale instance. Read-only queries also always read the file fresh
so a ``Watchlist`` object never returns state a concurrent writer has
since replaced.
"""
from __future__ import annotations

import datetime
import fcntl
import json
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any


class Watchlist:
    """Persistent JSON watchlist of TIC IDs."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _load(self) -> dict[str, Any]:
        if self.path.exists():
            return json.loads(self.path.read_text())
        return {"entries": {}}

    def _save(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(data, indent=2)
        tmp = Path(tempfile.mktemp(dir=self.path.parent, suffix=".tmp"))
        tmp.write_text(payload)
        tmp.replace(self.path)

    def _mutate(self, fn: Callable[[dict[str, Any]], Any]) -> Any:
        """Reload fresh, run ``fn(data)`` in place, save -- all under an
        exclusive cross-process lock so a concurrent writer's update to the
        same file is never overwritten by a read that predates it."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        with lock_path.open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            data = self._load()
            result = fn(data)
            self._save(data)
            return result

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, tic_id: int, note: str = "") -> None:
        """Add a TIC ID to the watchlist (idempotent — updates note if present)."""

        def mutate(data: dict[str, Any]) -> None:
            data["entries"][str(tic_id)] = {
                "tic_id": tic_id,
                "note": note,
                "added_at": datetime.datetime.now(datetime.UTC).isoformat(),
            }

        self._mutate(mutate)

    def remove(self, tic_id: int) -> bool:
        """Remove a TIC ID. Returns True if it was present, False otherwise."""

        def mutate(data: dict[str, Any]) -> bool:
            key = str(tic_id)
            if key in data["entries"]:
                del data["entries"][key]
                return True
            return False

        return bool(self._mutate(mutate))

    def contains(self, tic_id: int) -> bool:
        """Return True if the TIC ID is in the watchlist."""
        return str(tic_id) in self._load()["entries"]

    def list_ids(self) -> list[int]:
        """Return sorted list of TIC IDs in the watchlist."""
        return sorted(int(k) for k in self._load()["entries"])

    def entries(self) -> list[dict[str, Any]]:
        """Return all entries sorted by TIC ID."""
        entries = self._load()["entries"]
        return [entries[k] for k in sorted(entries, key=int)]

    def clear(self) -> None:
        """Remove all entries from the watchlist."""

        def mutate(data: dict[str, Any]) -> None:
            data["entries"] = {}

        self._mutate(mutate)

    def summary(self) -> dict[str, Any]:
        """Return a summary dict with entry count and file path."""
        return {
            "n_entries": len(self._load()["entries"]),
            "path": str(self.path),
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli(argv: list[str] | None = None) -> int:
    import argparse  # noqa: PLC0415
    import sys  # noqa: PLC0415

    parser = argparse.ArgumentParser(
        prog="watchlist",
        description="Manage a watchlist of TIC IDs for transit follow-up.",
    )
    parser.add_argument(
        "--watchlist",
        type=Path,
        default=Path("data/watchlist.json"),
        metavar="FILE",
        help="Watchlist JSON file (default: data/watchlist.json).",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # add
    p_add = sub.add_parser("add", help="Add a TIC ID to the watchlist.")
    p_add.add_argument("tic_id", type=int)
    p_add.add_argument("--note", default="", help="Optional note.")

    # remove
    p_rem = sub.add_parser("remove", help="Remove a TIC ID from the watchlist.")
    p_rem.add_argument("tic_id", type=int)

    # list
    sub.add_parser("list", help="Print all watchlist entries.")

    # clear
    sub.add_parser("clear", help="Remove all entries from the watchlist.")

    # summary
    sub.add_parser("summary", help="Print watchlist summary.")

    args = parser.parse_args(argv)
    wl = Watchlist(args.watchlist)

    if args.command == "add":
        wl.add(args.tic_id, note=args.note)
        print(f"Added TIC {args.tic_id}.")

    elif args.command == "remove":
        removed = wl.remove(args.tic_id)
        if removed:
            print(f"Removed TIC {args.tic_id}.")
        else:
            print(f"TIC {args.tic_id} not found in watchlist.", file=sys.stderr)
            return 1

    elif args.command == "list":
        for entry in wl.entries():
            note = f"  # {entry['note']}" if entry["note"] else ""
            print(f"TIC {entry['tic_id']}  [{entry['added_at']}]{note}")
        if not wl.list_ids():
            print("(watchlist is empty)")

    elif args.command == "clear":
        wl.clear()
        print("Watchlist cleared.")

    elif args.command == "summary":
        s = wl.summary()
        print(f"Entries: {s['n_entries']}  |  File: {s['path']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
