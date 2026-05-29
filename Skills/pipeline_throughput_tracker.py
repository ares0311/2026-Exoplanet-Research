"""Track pipeline throughput — targets processed, timing, and success rates.

Records run events to a JSON log and computes aggregate throughput statistics.

Public API
----------
ThroughputEntry(tic_id, status, duration_s, run_at)
ThroughputStats(n_total, n_success, n_error, n_no_data, mean_duration_s,
                targets_per_hour, flag)
ThroughputTracker(log_path)
    .record(tic_id, status, duration_s) -> None
    .stats() -> ThroughputStats
    .clear() -> None
format_throughput_stats(stats) -> str
"""
from __future__ import annotations

import contextlib
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ThroughputEntry:
    tic_id: int | None
    status: str   # "success" | "error" | "no_data"
    duration_s: float
    run_at: float  # Unix timestamp


@dataclass(frozen=True)
class ThroughputStats:
    n_total: int
    n_success: int
    n_error: int
    n_no_data: int
    mean_duration_s: float | None
    targets_per_hour: float | None
    flag: str  # "OK" | "HIGH_ERROR_RATE" | "EMPTY"


class ThroughputTracker:
    """Persistent throughput log backed by a JSON file."""

    def __init__(self, log_path: str | Path) -> None:
        self._path = Path(log_path)
        self._entries: list[ThroughputEntry] = self._load()

    def _load(self) -> list[ThroughputEntry]:
        if not self._path.exists():
            return []
        try:
            raw = json.loads(self._path.read_text())
            entries = []
            for r in raw:
                with contextlib.suppress(Exception):
                    entries.append(ThroughputEntry(
                        tic_id=r.get("tic_id"),
                        status=str(r["status"]),
                        duration_s=float(r["duration_s"]),
                        run_at=float(r["run_at"]),
                    ))
            return entries
        except Exception:
            return []

    def _save(self) -> None:
        rows = [
            {"tic_id": e.tic_id, "status": e.status,
             "duration_s": e.duration_s, "run_at": e.run_at}
            for e in self._entries
        ]
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = None
        try:
            fd, tmp = tempfile.mkstemp(dir=self._path.parent, suffix=".tmp")
            with os.fdopen(fd, "w") as fh:
                json.dump(rows, fh)
            os.replace(tmp, self._path)
        except Exception:
            with contextlib.suppress(OSError):
                if tmp:
                    os.unlink(tmp)
            raise

    def record(self, tic_id: int | None, status: str, duration_s: float) -> None:
        """Record a completed pipeline run."""
        entry = ThroughputEntry(
            tic_id=tic_id,
            status=status,
            duration_s=max(0.0, duration_s),
            run_at=time.time(),
        )
        self._entries.append(entry)
        self._save()

    def stats(self) -> ThroughputStats:
        """Compute aggregate throughput statistics."""
        entries = self._entries
        if not entries:
            return ThroughputStats(
                n_total=0, n_success=0, n_error=0, n_no_data=0,
                mean_duration_s=None, targets_per_hour=None, flag="EMPTY"
            )

        n_total = len(entries)
        n_success = sum(1 for e in entries if e.status == "success")
        n_error = sum(1 for e in entries if e.status == "error")
        n_no_data = sum(1 for e in entries if e.status == "no_data")
        durations = [e.duration_s for e in entries]
        mean_dur = sum(durations) / len(durations)

        # Compute targets per hour from time span
        times = [e.run_at for e in entries]
        time_span_h = (max(times) - min(times)) / 3600
        tph = n_total / time_span_h if time_span_h > 0 else None

        error_rate = n_error / n_total if n_total > 0 else 0.0
        flag = "HIGH_ERROR_RATE" if error_rate > 0.2 else "OK"

        return ThroughputStats(
            n_total=n_total,
            n_success=n_success,
            n_error=n_error,
            n_no_data=n_no_data,
            mean_duration_s=round(mean_dur, 2),
            targets_per_hour=round(tph, 2) if tph is not None else None,
            flag=flag,
        )

    def clear(self) -> None:
        """Clear all recorded entries."""
        self._entries = []
        self._save()


def format_throughput_stats(stats: ThroughputStats) -> str:
    """Format throughput statistics as Markdown.

    Args:
        stats: ThroughputStats to format.

    Returns:
        Markdown string.
    """
    dur_str = f"{stats.mean_duration_s:.1f} s" if stats.mean_duration_s is not None else "—"
    tph_str = (f"{stats.targets_per_hour:.1f}/hr"
               if stats.targets_per_hour is not None else "—")
    lines = [
        "## Pipeline Throughput\n",
        f"**Status**: `{stats.flag}`\n",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Total targets | {stats.n_total} |",
        f"| Success | {stats.n_success} |",
        f"| Error | {stats.n_error} |",
        f"| No data | {stats.n_no_data} |",
        f"| Mean duration | {dur_str} |",
        f"| Throughput | {tph_str} |",
    ]
    return "\n".join(lines)


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Pipeline throughput tracker.")
    sub = parser.add_subparsers(dest="cmd")

    rec = sub.add_parser("record", help="Record a pipeline run.")
    rec.add_argument("--log", required=True)
    rec.add_argument("--tic-id", type=int, default=None)
    rec.add_argument("--status", default="success")
    rec.add_argument("--duration", type=float, default=0.0)

    rep = sub.add_parser("report", help="Print throughput report.")
    rep.add_argument("--log", required=True)

    args = parser.parse_args(argv)
    if args.cmd == "record":
        tracker = ThroughputTracker(args.log)
        tracker.record(args.tic_id, args.status, args.duration)
        print("Recorded.")
    elif args.cmd == "report":
        tracker = ThroughputTracker(args.log)
        stats = tracker.stats()
        print(format_throughput_stats(stats))
    else:
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
