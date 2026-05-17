"""Identify transit events that fell inside data gaps.

Given a transit ephemeris and a list of observed-sector time ranges,
reports which predicted transits are fully covered, partially covered,
or missed due to data gaps.

Public API
----------
find_sector_gaps(period_days, epoch_bjd, sector_times, *, n_periods, duration_days)
    -> GapReport
format_gap_report(report) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class TransitCoverage:
    transit_number: int
    mid_bjd: float
    covered: bool
    partial: bool


@dataclass
class GapReport:
    total_transits: int
    covered_transits: int
    partial_transits: int
    missed_transits: int
    gap_fraction: float
    coverages: list[TransitCoverage] = field(default_factory=list)


def find_sector_gaps(
    period_days: float,
    epoch_bjd: float,
    sector_times: list[tuple[float, float]],
    *,
    n_periods: int | None = None,
    duration_days: float = 0.0833,
) -> GapReport:
    """Find which transits fall in data gaps.

    Args:
        period_days: Orbital period in days.
        epoch_bjd: Reference transit mid-time (BJD).
        sector_times: List of ``(start_bjd, end_bjd)`` observed time ranges.
            Ranges may overlap; the union is used.
        n_periods: Number of orbital periods to evaluate.  Defaults to the
            number of periods that fit within the span of ``sector_times``.
        duration_days: Transit duration in days used to assess partial coverage.
            Default 2 hours (0.0833 d).

    Returns:
        :class:`GapReport` with per-transit coverage flags.

    Raises:
        ValueError: If ``period_days`` ≤ 0 or no sector times provided.
    """
    if period_days <= 0:
        raise ValueError(f"period_days must be positive, got {period_days}")
    if not sector_times:
        raise ValueError("sector_times must not be empty")

    t_min = min(s for s, _ in sector_times)
    t_max = max(e for _, e in sector_times)

    if n_periods is None:
        span = t_max - t_min
        n_periods = max(1, int(span / period_days))

    half_dur = duration_days / 2.0
    n_start = math.ceil((t_min - epoch_bjd) / period_days)

    coverages: list[TransitCoverage] = []
    for i in range(n_periods):
        k = n_start + i
        mid = epoch_bjd + k * period_days
        t_in = mid - half_dur
        t_out = mid + half_dur

        # Check if mid-time is within any sector
        in_sector = any(s <= mid <= e for s, e in sector_times)
        # Check if full transit window is within some sector
        fully_covered = any(s <= t_in and t_out <= e for s, e in sector_times)
        partial = in_sector and not fully_covered

        coverages.append(TransitCoverage(
            transit_number=k,
            mid_bjd=mid,
            covered=fully_covered,
            partial=partial,
        ))

    total = len(coverages)
    covered = sum(1 for c in coverages if c.covered)
    partial = sum(1 for c in coverages if c.partial)
    missed = total - covered - partial
    gap_frac = missed / total if total > 0 else 0.0

    return GapReport(
        total_transits=total,
        covered_transits=covered,
        partial_transits=partial,
        missed_transits=missed,
        gap_fraction=gap_frac,
        coverages=coverages,
    )


def format_gap_report(report: GapReport) -> str:
    """Format a gap report as a short Markdown summary.

    Args:
        report: From :func:`find_sector_gaps`.

    Returns:
        Markdown string.
    """
    lines = [
        "## Sector Gap Report",
        "",
        "| Total transits | Covered | Partial | Missed | Gap fraction |",
        "| --- | --- | --- | --- | --- |",
        f"| {report.total_transits} | {report.covered_transits} "
        f"| {report.partial_transits} | {report.missed_transits} "
        f"| {report.gap_fraction:.2%} |",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(
        prog="sector_gap_finder",
        description="Identify transit events that fell in data gaps.",
    )
    parser.add_argument("--period", type=float, required=True, metavar="DAYS")
    parser.add_argument("--epoch", type=float, required=True, metavar="BJD")
    parser.add_argument("--sectors", type=str, required=True, metavar="JSON",
                        help='JSON list of [start_bjd, end_bjd] pairs.')
    parser.add_argument("--duration", type=float, default=0.0833, metavar="DAYS",
                        help="Transit duration in days (default: 0.0833 ≈ 2h).")
    args = parser.parse_args(argv)

    sector_times = [tuple(s) for s in json.loads(args.sectors)]
    report = find_sector_gaps(
        args.period, args.epoch, sector_times, duration_days=args.duration
    )
    print(format_gap_report(report), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
