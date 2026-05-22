"""Estimate the number of observable transits in a set of time intervals.

Given an orbital period, reference epoch, and a list of (start, end) time
intervals, counts how many transit mid-times fall inside the combined
intervals and how many are complete (ingress and egress both within the
window).

Public API
----------
TransitCountResult(period_days, n_total_predicted, n_in_window,
                   n_complete, coverage_fraction, flag)
estimate_transit_count(period_days, epoch_bjd, intervals, *,
                       duration_hours) -> TransitCountResult
format_transit_count_result(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TransitCountResult:
    period_days: float
    n_total_predicted: int       # across the full span of intervals
    n_in_window: int             # mid-time inside at least one interval
    n_complete: int              # both ingress and egress inside an interval
    coverage_fraction: float     # n_in_window / n_total_predicted
    flag: str  # "OK" | "NO_INTERVALS" | "INVALID"


def estimate_transit_count(
    period_days: float,
    epoch_bjd: float,
    intervals: list[tuple[float, float]],
    *,
    duration_hours: float = 2.0,
) -> TransitCountResult:
    """Count transit mid-times falling inside time intervals.

    Args:
        period_days: Orbital period in days.
        epoch_bjd: Reference mid-transit epoch (BJD or same time base).
        intervals: List of (start, end) time intervals in the same time base.
        duration_hours: Transit duration (used to check completeness).

    Returns:
        :class:`TransitCountResult`.
    """
    if period_days <= 0:
        return TransitCountResult(period_days, 0, 0, 0, 0.0, "INVALID")
    if not intervals:
        return TransitCountResult(period_days, 0, 0, 0, 0.0, "NO_INTERVALS")

    # Filter degenerate intervals
    valid = [(s, e) for s, e in intervals if e > s]
    if not valid:
        return TransitCountResult(period_days, 0, 0, 0, 0.0, "NO_INTERVALS")

    t_min = min(s for s, _ in valid)
    t_max = max(e for _, e in valid)

    half_dur = (duration_hours / 24.0) / 2.0

    n_start = math.floor((t_min - epoch_bjd) / period_days)
    n_end = math.ceil((t_max - epoch_bjd) / period_days)

    n_total = 0
    n_in = 0
    n_complete = 0

    for k in range(n_start, n_end + 1):
        mid = epoch_bjd + k * period_days
        if mid < t_min or mid > t_max:
            continue
        n_total += 1
        for s, e in valid:
            if s <= mid <= e:
                n_in += 1
                if mid - half_dur >= s and mid + half_dur <= e:
                    n_complete += 1
                break

    cov = n_in / n_total if n_total > 0 else 0.0

    return TransitCountResult(
        period_days=period_days,
        n_total_predicted=n_total,
        n_in_window=n_in,
        n_complete=n_complete,
        coverage_fraction=round(cov, 4),
        flag="OK",
    )


def format_transit_count_result(result: TransitCountResult) -> str:
    """Format transit count result as Markdown."""
    lines = [
        "## Transit Count Estimate",
        "",
        f"- Period: {result.period_days:.4f} days",
        f"- Transits in span: {result.n_total_predicted}",
        f"- Transits in window: {result.n_in_window}",
        f"- Complete transits: {result.n_complete}",
        f"- Coverage fraction: {result.coverage_fraction:.2%}",
        f"- **Flag: {result.flag}**",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="transit_count_estimator",
        description="Count observable transits within time intervals.",
    )
    parser.add_argument("period_days", type=float)
    parser.add_argument("epoch_bjd", type=float)
    parser.add_argument("--duration-hours", type=float, default=2.0)
    args = parser.parse_args(argv)

    result = estimate_transit_count(
        args.period_days, args.epoch_bjd, [],
        duration_hours=args.duration_hours,
    )
    print(format_transit_count_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
