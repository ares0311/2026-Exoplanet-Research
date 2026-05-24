"""Merge overlapping or adjacent observation time windows.

Given a list of (start, end) intervals, returns a minimal set of
non-overlapping contiguous windows.  Useful for combining observation
schedules from multiple nights or instruments into a clean baseline.
Distinct from ``follow_up_scheduler`` (generates schedules) and
``transit_survey_planner`` (plans transit events).

Public API
----------
TimeWindow(start, end, duration)
MergedWindowsResult(n_input, n_merged, total_duration,
                    merged_windows, flag)
merge_windows(windows, *, gap_threshold_days) -> MergedWindowsResult
format_merged_windows(result) -> str
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TimeWindow:
    start: float   # BJD or any consistent time unit
    end: float
    duration: float


@dataclass(frozen=True)
class MergedWindowsResult:
    n_input: int
    n_merged: int
    total_duration: float
    merged_windows: tuple[TimeWindow, ...]
    flag: str  # "OK" | "EMPTY" | "INVALID"


def merge_windows(
    windows: list[tuple[float, float]],
    *,
    gap_threshold_days: float = 0.0,
) -> MergedWindowsResult:
    """Merge overlapping/adjacent time windows.

    Args:
        windows: List of ``(start, end)`` tuples (days or BJD).
        gap_threshold_days: Treat gaps smaller than this as merged.
            Use ``0.0`` for exact overlap/touching only.

    Returns:
        :class:`MergedWindowsResult`.
    """
    if not isinstance(windows, list):
        return MergedWindowsResult(0, 0, 0.0, (), "INVALID")
    if not windows:
        return MergedWindowsResult(0, 0, 0.0, (), "EMPTY")

    # Validate all windows
    for w in windows:
        if len(w) < 2 or w[0] > w[1]:
            return MergedWindowsResult(len(windows), 0, 0.0, (), "INVALID")

    sorted_wins = sorted(windows, key=lambda w: w[0])
    merged: list[list[float]] = [list(sorted_wins[0])]

    for start, end in sorted_wins[1:]:
        last = merged[-1]
        if start <= last[1] + gap_threshold_days:
            last[1] = max(last[1], end)
        else:
            merged.append([start, end])

    result_windows = tuple(
        TimeWindow(start=w[0], end=w[1], duration=round(w[1] - w[0], 8))
        for w in merged
    )
    total = sum(w.duration for w in result_windows)

    return MergedWindowsResult(
        n_input=len(windows),
        n_merged=len(result_windows),
        total_duration=round(total, 8),
        merged_windows=result_windows,
        flag="OK",
    )


def format_merged_windows(result: MergedWindowsResult) -> str:
    """Format merged windows result as Markdown."""
    lines = [
        "## Observation Window Merger",
        "",
        f"- Input windows: {result.n_input}",
        f"- **Merged windows: {result.n_merged}**",
        f"- Total coverage: {result.total_duration:.4f} days",
        f"- **Flag: {result.flag}**",
    ]
    if result.merged_windows:
        lines += ["", "| Start | End | Duration (days) |", "|---|---|---|"]
        for w in result.merged_windows[:10]:
            lines.append(f"| {w.start:.4f} | {w.end:.4f} | {w.duration:.4f} |")
        if len(result.merged_windows) > 10:
            lines.append(f"| … | … | ({len(result.merged_windows) - 10} more) |")
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(
        prog="observation_window_merger",
        description="Merge overlapping observation time windows.",
    )
    parser.add_argument("--windows", type=str, default=None,
                        help='JSON array of [start, end] pairs')
    parser.add_argument("--gap-threshold", type=float, default=0.0)
    args = parser.parse_args(argv)

    wins = [tuple(w) for w in json.loads(args.windows)] if args.windows else []
    result = merge_windows(wins, gap_threshold_days=args.gap_threshold)
    print(format_merged_windows(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
