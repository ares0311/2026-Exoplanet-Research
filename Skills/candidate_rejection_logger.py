"""Persistent atomic-write log of rejected candidates with reason codes.

Public API:
    RejectionEntry  -- frozen dataclass
    RejectionLog    -- mutable, file-backed log
    format_rejection_summary(log) -> str
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class RejectionEntry:
    tic_id: int
    reason_code: str
    fpp: float | None
    note: str
    rejected_at: str


class RejectionLog:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def _load(self) -> list[dict]:
        if not self._path.exists():
            return []
        with open(self._path) as fh:
            data = json.load(fh)
        return data if isinstance(data, list) else []

    def _save(self, entries: list[dict]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self._path.parent)
        try:
            with os.fdopen(fd, "w") as fh:
                json.dump(entries, fh, indent=2)
            os.replace(tmp, self._path)
        except Exception:
            os.unlink(tmp)
            raise

    def record(self, tic_id: int, reason_code: str, *,
                fpp: float | None = None, note: str = "") -> RejectionEntry:
        entry = RejectionEntry(
            tic_id=tic_id,
            reason_code=reason_code,
            fpp=fpp,
            note=note,
            rejected_at=datetime.now(tz=UTC).isoformat(),
        )
        entries = self._load()
        entries.append({
            "tic_id": entry.tic_id,
            "reason_code": entry.reason_code,
            "fpp": entry.fpp,
            "note": entry.note,
            "rejected_at": entry.rejected_at,
        })
        self._save(entries)
        return entry

    def get(self, tic_id: int) -> list[RejectionEntry]:
        return [
            RejectionEntry(**e) for e in self._load() if e["tic_id"] == tic_id
        ]

    def all_entries(self) -> list[RejectionEntry]:
        return [RejectionEntry(**e) for e in self._load()]

    def summary(self) -> dict:
        entries = self._load()
        reason_counts: dict[str, int] = {}
        for e in entries:
            reason_counts[e["reason_code"]] = reason_counts.get(e["reason_code"], 0) + 1
        return {"n_rejected": len(entries), "reason_counts": reason_counts}

    def export_csv(self) -> str:
        entries = self._load()
        lines = ["tic_id,reason_code,fpp,note,rejected_at"]
        for e in entries:
            fpp_str = str(e["fpp"]) if e["fpp"] is not None else ""
            note = str(e["note"]).replace(",", ";")
            lines.append(f"{e['tic_id']},{e['reason_code']},{fpp_str},{note},{e['rejected_at']}")
        return "\n".join(lines)


def format_rejection_summary(log: RejectionLog) -> str:
    s = log.summary()
    lines = [
        "## Rejection Log Summary",
        "",
        f"Total Rejected: {s['n_rejected']}",
        "",
        "| Reason Code | Count |",
        "|-------------|-------|",
    ]
    for code, count in sorted(s["reason_counts"].items()):
        lines.append(f"| {code} | {count} |")
    return "\n".join(lines)


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Candidate rejection logger.")
    sub = parser.add_subparsers(dest="cmd")
    rec = sub.add_parser("record")
    rec.add_argument("log_path")
    rec.add_argument("tic_id", type=int)
    rec.add_argument("reason_code")
    rec.add_argument("--fpp", type=float, default=None)
    rec.add_argument("--note", default="")
    lst = sub.add_parser("summary")
    lst.add_argument("log_path")
    args = parser.parse_args()
    if args.cmd == "record":
        log = RejectionLog(args.log_path)
        log.record(args.tic_id, args.reason_code, fpp=args.fpp, note=args.note)
        print(f"Recorded rejection: TIC {args.tic_id} ({args.reason_code})")
    elif args.cmd == "summary":
        log = RejectionLog(args.log_path)
        print(format_rejection_summary(log))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
