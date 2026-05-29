"""Schedule ground-based follow-up observations for multiple targets.

Assigns time slots to targets based on priority score, transit window
availability, and a simple non-overlapping greedy scheduler.

Public API
----------
TargetWindow(tic_id, start_btjd, end_btjd, priority, duration_hours)
ScheduleEntry(tic_id, slot_start, slot_end, duration_hours, priority, note)
ScheduleResult(date_label, entries, n_scheduled, n_skipped,
               total_hours, flag)
schedule_targets(windows, *, night_start, night_end,
                 min_priority) -> ScheduleResult
format_schedule_result(result) -> str
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TargetWindow:
    tic_id: int
    start_btjd: float
    end_btjd: float
    priority: float    # 0–1; higher = more important
    duration_hours: float  # requested observation duration


@dataclass(frozen=True)
class ScheduleEntry:
    tic_id: int
    slot_start: float   # BTJD
    slot_end: float     # BTJD
    duration_hours: float
    priority: float
    note: str


@dataclass(frozen=True)
class ScheduleResult:
    date_label: str
    entries: tuple[ScheduleEntry, ...]
    n_scheduled: int
    n_skipped: int
    total_hours: float
    flag: str  # "OK" | "PARTIAL" | "EMPTY" | "OVERSUBSCRIBED"


def schedule_targets(
    windows: list[TargetWindow],
    *,
    night_start: float,
    night_end: float,
    min_priority: float = 0.0,
    date_label: str = "",
) -> ScheduleResult:
    """Greedy priority-ordered scheduler for a single night.

    Args:
        windows: List of target observation windows.
        night_start: Start of observable night in BTJD.
        night_end: End of observable night in BTJD.
        min_priority: Skip targets with priority below this threshold.
        date_label: Human-readable label for the night.

    Returns:
        ScheduleResult with non-overlapping assigned slots.
    """
    night_hours = (night_end - night_start) * 24.0

    if night_start >= night_end or not windows:
        return ScheduleResult(
            date_label=date_label,
            entries=(),
            n_scheduled=0,
            n_skipped=len(windows),
            total_hours=0.0,
            flag="EMPTY",
        )

    # Filter by priority and night overlap, then sort by priority desc
    candidates = [
        w for w in windows
        if w.priority >= min_priority
        and w.start_btjd < night_end
        and w.end_btjd > night_start
    ]
    candidates.sort(key=lambda w: w.priority, reverse=True)

    n_skipped = len(windows) - len(candidates)
    entries: list[ScheduleEntry] = []
    cursor = night_start   # current free-time cursor

    for w in candidates:
        dur_days = w.duration_hours / 24.0
        # Earliest we can start: max(cursor, window_start, night_start)
        earliest = max(cursor, w.start_btjd, night_start)
        latest_end = min(w.end_btjd, night_end)
        slot_start = earliest
        slot_end = slot_start + dur_days

        if slot_end > latest_end:
            n_skipped += 1
            continue

        entries.append(ScheduleEntry(
            tic_id=w.tic_id,
            slot_start=round(slot_start, 6),
            slot_end=round(slot_end, 6),
            duration_hours=w.duration_hours,
            priority=w.priority,
            note="",
        ))
        cursor = slot_end

    total_hours = sum(e.duration_hours for e in entries)
    n_scheduled = len(entries)

    if n_scheduled == 0:
        flag = "EMPTY"
    elif n_skipped > 0 and total_hours >= night_hours * 0.90:
        flag = "OVERSUBSCRIBED"
    elif n_skipped > 0:
        flag = "PARTIAL"
    else:
        flag = "OK"

    return ScheduleResult(
        date_label=date_label,
        entries=tuple(entries),
        n_scheduled=n_scheduled,
        n_skipped=n_skipped,
        total_hours=round(total_hours, 3),
        flag=flag,
    )


def format_schedule_result(result: ScheduleResult) -> str:
    """Format schedule result as Markdown.

    Args:
        result: ScheduleResult to format.

    Returns:
        Markdown string.
    """
    label = result.date_label or "—"
    lines = [
        f"## Multi-Target Schedule — {label}\n",
        f"**Status**: `{result.flag}` | "
        f"Scheduled: {result.n_scheduled} | Skipped: {result.n_skipped} | "
        f"Total: {result.total_hours:.2f} h\n",
    ]
    if not result.entries:
        lines.append("\n_No targets scheduled._")
        return "\n".join(lines)

    lines += [
        "",
        "| # | TIC ID | Start (BTJD) | End (BTJD) | Duration (h) | Priority |",
        "|---|---|---|---|---|---|",
    ]
    for i, e in enumerate(result.entries, 1):
        lines.append(
            f"| {i} | {e.tic_id} | {e.slot_start:.4f} | "
            f"{e.slot_end:.4f} | {e.duration_hours:.2f} | {e.priority:.2f} |"
        )
    return "\n".join(lines)


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Schedule multi-target observations.")
    parser.add_argument("windows", help="JSON file with list of target windows.")
    parser.add_argument("--night-start", type=float, required=True)
    parser.add_argument("--night-end", type=float, required=True)
    parser.add_argument("--min-priority", type=float, default=0.0)
    parser.add_argument("--date", default="")
    args = parser.parse_args(argv)

    from pathlib import Path
    raw = json.loads(Path(args.windows).read_text())
    wlist = [
        TargetWindow(
            tic_id=w["tic_id"],
            start_btjd=w["start_btjd"],
            end_btjd=w["end_btjd"],
            priority=w.get("priority", 0.5),
            duration_hours=w.get("duration_hours", 1.0),
        )
        for w in raw
    ]
    result = schedule_targets(
        wlist,
        night_start=args.night_start,
        night_end=args.night_end,
        min_priority=args.min_priority,
        date_label=args.date,
    )
    print(format_schedule_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
