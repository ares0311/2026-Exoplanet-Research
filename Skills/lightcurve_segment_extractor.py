"""Extract time-series segments centred on transit mid-times.

Given time and flux arrays plus a list of mid-transit times, extracts
symmetric windows around each transit for detailed inspection.

Public API
----------
TransitSegment(index, mid_time, time, flux, flux_err, n_points,
               coverage_fraction, flag)
extract_transit_segments(time, flux, *, mid_times, half_window_days,
                         flux_err) -> list[TransitSegment]
format_segment_summary(segments) -> str
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TransitSegment:
    index: int
    mid_time: float
    time: tuple[float, ...]
    flux: tuple[float, ...]
    flux_err: tuple[float | None, ...]
    n_points: int
    coverage_fraction: float  # observed / expected given typical cadence
    flag: str  # "OK" | "SPARSE" | "EMPTY"


def extract_transit_segments(
    time: list[float] | tuple[float, ...],
    flux: list[float] | tuple[float, ...],
    *,
    mid_times: list[float],
    half_window_days: float = 0.5,
    flux_err: list[float | None] | tuple[float | None, ...] | None = None,
    cadence_days: float | None = None,
    min_coverage: float = 0.5,
) -> list[TransitSegment]:
    """Extract symmetric windows around transit mid-times.

    Args:
        time: Array of observation timestamps (days).
        flux: Corresponding flux values.
        mid_times: List of transit mid-time timestamps.
        half_window_days: Half-width of the extraction window in days.
        flux_err: Optional flux uncertainty array.
        cadence_days: Assumed cadence for computing coverage fraction.
                      If None, estimated from data.
        min_coverage: Coverage fraction below which segment is flagged SPARSE.

    Returns:
        List of TransitSegment, one per mid-time.
    """
    t = list(time)
    f = list(flux)
    err: list[float | None] = list(flux_err) if flux_err is not None else [None] * len(t)

    if cadence_days is None and len(t) >= 2:
        diffs = [abs(t[i + 1] - t[i]) for i in range(min(100, len(t) - 1))]
        cadence_days = sorted(diffs)[len(diffs) // 2]

    window = half_window_days
    expected_points = (2 * window / cadence_days) if cadence_days else 1.0

    segments: list[TransitSegment] = []
    for idx, mid in enumerate(mid_times):
        lo = mid - window
        hi = mid + window
        sel_t: list[float] = []
        sel_f: list[float] = []
        sel_e: list[float | None] = []
        for ti, fi, ei in zip(t, f, err, strict=False):
            if lo <= ti <= hi:
                sel_t.append(ti)
                sel_f.append(fi)
                sel_e.append(ei)

        n_pts = len(sel_t)
        cov = min(n_pts / max(expected_points, 1), 1.0)

        if n_pts == 0:
            flag = "EMPTY"
        elif cov < min_coverage:
            flag = "SPARSE"
        else:
            flag = "OK"

        segments.append(TransitSegment(
            index=idx,
            mid_time=mid,
            time=tuple(sel_t),
            flux=tuple(sel_f),
            flux_err=tuple(sel_e),
            n_points=n_pts,
            coverage_fraction=round(cov, 4),
            flag=flag,
        ))

    return segments


def format_segment_summary(segments: list[TransitSegment]) -> str:
    """Format segment extraction summary as Markdown.

    Args:
        segments: List of TransitSegment from extract_transit_segments.

    Returns:
        Markdown string.
    """
    if not segments:
        return "## Transit Segment Extractor\n\n_No segments extracted._"

    n_ok = sum(1 for s in segments if s.flag == "OK")
    n_sparse = sum(1 for s in segments if s.flag == "SPARSE")
    n_empty = sum(1 for s in segments if s.flag == "EMPTY")

    lines = [
        "## Transit Segment Extractor\n",
        f"**Segments**: {len(segments)} | OK: {n_ok} | Sparse: {n_sparse} | Empty: {n_empty}\n",
        "",
        "| # | Mid-time (d) | N points | Coverage | Flag |",
        "|---|---|---|---|---|",
    ]
    for s in segments:
        lines.append(
            f"| {s.index} | {s.mid_time:.4f} | {s.n_points} | "
            f"{s.coverage_fraction:.2%} | `{s.flag}` |"
        )
    return "\n".join(lines)


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Extract transit segments from a light curve.")
    parser.add_argument("lc", help="Light curve JSON with 'time' and 'flux' arrays.")
    parser.add_argument("--mid-times", required=True,
                        help="Comma-separated list of mid-times in days.")
    parser.add_argument("--half-window", type=float, default=0.5)
    args = parser.parse_args(argv)

    from pathlib import Path
    data = json.loads(Path(args.lc).read_text())
    mid_times = [float(x) for x in args.mid_times.split(",")]
    segments = extract_transit_segments(
        data["time"], data["flux"],
        mid_times=mid_times,
        half_window_days=args.half_window,
        flux_err=data.get("flux_err"),
    )
    print(format_segment_summary(segments))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
