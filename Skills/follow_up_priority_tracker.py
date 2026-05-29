"""Persistent JSON follow-up priority tracker for TIC IDs.

Priority levels: 1 (highest) to 5 (lowest).
Records are stored as a list in a JSON file, newest first per TIC ID.

Public API
----------
PriorityRecord(tic_id, priority, reason, updated_at)
FollowUpPriorityTracker(path)
    .update(tic_id, priority, reason)
    .get_history(tic_id) -> list[PriorityRecord]
    .top_priority(n) -> list[PriorityRecord]
format_priority_tracker(tracker, n) -> str
"""
from __future__ import annotations

import contextlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class PriorityRecord:
    tic_id: int
    priority: int   # 1 = highest, 5 = lowest
    reason: str
    updated_at: str  # ISO timestamp


class FollowUpPriorityTracker:
    """Persistent JSON tracker for follow-up priorities."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._data: list[dict] = self._load()

    def _load(self) -> list[dict]:
        if self._path.exists():
            try:
                return json.loads(self._path.read_text())
            except (json.JSONDecodeError, OSError):
                return []
        return []

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=self._path.parent, suffix=".tmp"
        )
        try:
            with os.fdopen(tmp_fd, "w") as fh:
                json.dump(self._data, fh, indent=2)
            os.replace(tmp_path, self._path)
        except Exception:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
            raise

    def update(self, tic_id: int, priority: int, reason: str) -> PriorityRecord:
        """Add or update priority for a TIC ID.

        Args:
            tic_id: TIC target identifier.
            priority: Priority level 1–5 (1 = highest).
            reason: Human-readable reason for the priority.

        Returns:
            The created :class:`PriorityRecord`.
        """
        priority = max(1, min(5, int(priority)))
        record = PriorityRecord(
            tic_id=int(tic_id),
            priority=priority,
            reason=str(reason),
            updated_at=datetime.now(UTC).isoformat(),
        )
        self._data.append(asdict(record))
        self._save()
        return record

    def get_history(self, tic_id: int) -> list[PriorityRecord]:
        """Return all priority records for a TIC ID (oldest first)."""
        return [
            PriorityRecord(**r)
            for r in self._data
            if r["tic_id"] == int(tic_id)
        ]

    def top_priority(self, n: int = 5) -> list[PriorityRecord]:
        """Return the most recent record per TIC ID, sorted by priority asc.

        Args:
            n: Maximum number of records to return.

        Returns:
            List of :class:`PriorityRecord` sorted by priority (1 best).
        """
        seen: dict[int, PriorityRecord] = {}
        # Iterate newest first (last appended wins)
        for r in reversed(self._data):
            rec = PriorityRecord(**r)
            if rec.tic_id not in seen:
                seen[rec.tic_id] = rec
        ranked = sorted(seen.values(), key=lambda r: r.priority)
        return ranked[:n]


def format_priority_tracker(tracker: FollowUpPriorityTracker, n: int = 10) -> str:
    """Format top-priority targets as Markdown."""
    top = tracker.top_priority(n)
    lines = [
        "## Follow-Up Priority Tracker",
        "",
        f"- Tracking {n} top targets",
        "",
        "| TIC ID | Priority | Reason | Updated |",
        "|--------|----------|--------|---------|",
    ]
    for rec in top:
        lines.append(
            f"| {rec.tic_id} | {rec.priority} | {rec.reason} | {rec.updated_at[:10]} |"
        )
    return "\n".join(lines)


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", help="Path to priority tracker JSON file")
    sub = p.add_subparsers(dest="cmd")

    up = sub.add_parser("update")
    up.add_argument("tic_id", type=int)
    up.add_argument("priority", type=int)
    up.add_argument("reason")

    hist = sub.add_parser("history")
    hist.add_argument("tic_id", type=int)

    top_p = sub.add_parser("top")
    top_p.add_argument("--n", type=int, default=5)

    args = p.parse_args(argv)
    tracker = FollowUpPriorityTracker(args.path)

    if args.cmd == "update":
        rec = tracker.update(args.tic_id, args.priority, args.reason)
        print(f"Updated TIC {rec.tic_id}: priority {rec.priority}")
    elif args.cmd == "history":
        for rec in tracker.get_history(args.tic_id):
            print(f"{rec.updated_at[:10]}  P{rec.priority}  {rec.reason}")
    else:
        print(format_priority_tracker(tracker, getattr(args, "n", 10)))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
