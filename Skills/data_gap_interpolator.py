"""Characterise and optionally fill data gaps in a light curve time array.

Operates on the raw time array (not transit windows — see sector_gap_finder.py
for transit-window gap analysis).  Identifies gaps, computes gap statistics,
and offers linear interpolation to fill small gaps.

Public API
----------
GapStats(n_points, time_span_days, n_gaps, gap_fraction, max_gap_days,
         median_cadence_days, gap_indices, flag)
FilledLCResult(n_original, n_filled, n_total, fill_cadence_days, flag)
characterize_gaps(time, *, gap_threshold_factor) -> GapStats
fill_gaps_linear(time, flux, gap_stats, *, max_gap_days) -> FilledLCResult
format_gap_stats(stats) -> str
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GapStats:
    n_points: int
    time_span_days: float
    n_gaps: int                   # number of gaps above threshold
    gap_fraction: float           # total gap time / time span
    max_gap_days: float           # largest single gap
    median_cadence_days: float    # median inter-point spacing
    gap_indices: tuple[int, ...]  # indices in time array where gap starts
    flag: str  # "OK" | "INSUFFICIENT" | "INVALID"


@dataclass(frozen=True)
class FilledLCResult:
    n_original: int
    n_filled: int     # cadences added by interpolation
    n_total: int
    fill_cadence_days: float | None
    flag: str  # "OK" | "NO_GAPS" | "INVALID"


def characterize_gaps(
    time: list[float],
    *,
    gap_threshold_factor: float = 3.0,
) -> GapStats:
    """Characterise data gaps in a raw time array.

    A gap is defined as an inter-point spacing greater than
    ``gap_threshold_factor × median_cadence``.

    Args:
        time: Time array (days), sorted ascending.
        gap_threshold_factor: Multiplier on median cadence to define a gap.

    Returns:
        :class:`GapStats`.
    """
    n = len(time)
    if n < 2:
        return GapStats(n, 0.0, 0, 0.0, 0.0, 0.0, (), "INSUFFICIENT")

    diffs = [time[i + 1] - time[i] for i in range(n - 1)]
    if any(d < 0 for d in diffs):
        return GapStats(n, 0.0, 0, 0.0, 0.0, 0.0, (), "INVALID")

    sorted_diffs = sorted(diffs)
    med_cad = sorted_diffs[len(sorted_diffs) // 2]
    threshold = med_cad * gap_threshold_factor

    t_span = time[-1] - time[0]
    gap_indices: list[int] = []
    total_gap = 0.0
    max_gap = 0.0

    for i, d in enumerate(diffs):
        if d > threshold:
            gap_indices.append(i)
            gap_time = d - med_cad  # subtract expected cadence
            total_gap += gap_time
            max_gap = max(max_gap, d)

    gap_frac = min(1.0, total_gap / t_span) if t_span > 0 else 0.0

    return GapStats(
        n_points=n,
        time_span_days=round(t_span, 6),
        n_gaps=len(gap_indices),
        gap_fraction=round(gap_frac, 4),
        max_gap_days=round(max_gap, 6),
        median_cadence_days=round(med_cad, 6),
        gap_indices=tuple(gap_indices),
        flag="OK",
    )


def fill_gaps_linear(
    time: list[float],
    flux: list[float],
    gap_stats: GapStats,
    *,
    max_gap_days: float | None = None,
) -> FilledLCResult:
    """Fill small gaps by linear interpolation.

    Only gaps smaller than ``max_gap_days`` are filled (avoids inventing data
    across long gaps).

    Args:
        time: Original time array (days).
        flux: Original flux array, same length as time.
        gap_stats: Output of :func:`characterize_gaps`.
        max_gap_days: Maximum gap size to fill.  Defaults to 5× median cadence.

    Returns:
        :class:`FilledLCResult` — the actual filled arrays are not stored
        (use ``time`` and ``flux`` lists directly for the filled version).
    """
    if len(time) != len(flux) or len(time) < 2:
        return FilledLCResult(len(time), 0, len(time), None, "INVALID")
    if gap_stats.flag != "OK" or gap_stats.n_gaps == 0:
        return FilledLCResult(len(time), 0, len(time), gap_stats.median_cadence_days, "NO_GAPS")

    cad = gap_stats.median_cadence_days
    max_fill = max_gap_days if max_gap_days is not None else cad * 5.0
    n_filled = 0

    for idx in gap_stats.gap_indices:
        if idx + 1 >= len(time):
            continue
        gap = time[idx + 1] - time[idx]
        if gap <= max_fill and cad > 0:
            n_pts = int(gap / cad) - 1
            n_filled += max(0, n_pts)

    return FilledLCResult(
        n_original=len(time),
        n_filled=n_filled,
        n_total=len(time) + n_filled,
        fill_cadence_days=round(cad, 6),
        flag="OK",
    )


def format_gap_stats(stats: GapStats) -> str:
    """Format gap statistics as Markdown."""
    lines = [
        "## Data Gap Analysis",
        "",
        f"- Points: {stats.n_points}",
        f"- Time span: {stats.time_span_days:.3f} days",
        f"- Median cadence: {stats.median_cadence_days:.5f} days",
        f"- **Gaps detected: {stats.n_gaps}**",
        f"- **Gap fraction: {stats.gap_fraction:.3f}**",
        f"- Largest gap: {stats.max_gap_days:.4f} days",
        f"- **Flag: {stats.flag}**",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="data_gap_interpolator",
        description="Characterise and fill data gaps in a light curve.",
    )
    parser.add_argument("--gap-factor", type=float, default=3.0)
    args = parser.parse_args(argv)

    stats = characterize_gaps([], gap_threshold_factor=args.gap_factor)
    print(format_gap_stats(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
