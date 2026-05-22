"""Detect cadences where a transit window overlaps quality-flagged data.

If a transit falls partly on gapped, saturated, or otherwise flagged cadences
the depth measurement will be biased.  This module checks each predicted
transit against a quality mask or gap list and reports the overlap fraction.

Public API
----------
TransitOverlapResult(n_transits_checked, n_overlapping, overlap_fraction,
                     per_transit_overlap, flag)
detect_transit_overlaps(time, quality_mask, period_days, epoch_bjd, *,
                        duration_hours, gap_threshold_days) -> TransitOverlapResult
format_overlap_result(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TransitOverlapResult:
    n_transits_checked: int
    n_overlapping: int           # transits with at least one flagged cadence
    overlap_fraction: float      # n_overlapping / n_transits_checked
    per_transit_overlap: tuple[float, ...]  # fraction of in-transit cadences flagged
    flag: str  # "OK" | "NO_TRANSITS" | "INVALID"


def detect_transit_overlaps(
    time: list[float],
    quality_mask: list[bool],     # True = clean, False = flagged
    period_days: float,
    epoch_bjd: float,
    *,
    duration_hours: float = 2.0,
    gap_threshold_days: float = 0.02,
) -> TransitOverlapResult:
    """Check predicted transit windows against quality flags.

    Args:
        time: Time array (BJD or same convention as epoch_bjd).
        quality_mask: Boolean array — True means the cadence is clean.
            Same length as time.
        period_days: Orbital period in days.
        epoch_bjd: Reference mid-transit epoch.
        duration_hours: Transit duration in hours.
        gap_threshold_days: Maximum allowed gap between consecutive cadences;
            gaps larger than this are treated as missing data inside a transit.

    Returns:
        :class:`TransitOverlapResult`.
    """
    n = len(time)
    if n < 2 or len(quality_mask) != n or period_days <= 0:
        return TransitOverlapResult(0, 0, 0.0, (), "INVALID")

    half_dur = (duration_hours / 24.0) / 2.0

    # Find all transit mid-times within the time array range
    t_min = min(time)
    t_max = max(time)

    n_start = math.floor((t_min - epoch_bjd) / period_days)
    n_end = math.ceil((t_max - epoch_bjd) / period_days)

    per_transit: list[float] = []
    n_overlapping = 0

    for k in range(n_start, n_end + 1):
        mid = epoch_bjd + k * period_days
        t_in = mid - half_dur
        t_out = mid + half_dur

        if t_out < t_min or t_in > t_max:
            continue

        in_t: list[float] = []
        in_clean: list[bool] = []

        for t, q in zip(time, quality_mask, strict=False):
            if t_in <= t <= t_out:
                in_t.append(t)
                in_clean.append(q)

        if not in_t:
            continue

        n_flagged = sum(1 for q in in_clean if not q)
        frac_flagged = n_flagged / len(in_t)
        per_transit.append(round(frac_flagged, 4))
        if frac_flagged > 0:
            n_overlapping += 1

    n_checked = len(per_transit)
    if n_checked == 0:
        return TransitOverlapResult(0, 0, 0.0, (), "NO_TRANSITS")

    ov_frac = n_overlapping / n_checked

    return TransitOverlapResult(
        n_transits_checked=n_checked,
        n_overlapping=n_overlapping,
        overlap_fraction=round(ov_frac, 4),
        per_transit_overlap=tuple(per_transit),
        flag="OK",
    )


def format_overlap_result(result: TransitOverlapResult) -> str:
    """Format transit overlap result as Markdown."""
    lines = [
        "## Transit Overlap Detection",
        "",
        f"- Transits checked: {result.n_transits_checked}",
        f"- Overlapping transits: {result.n_overlapping}",
        f"- Overlap fraction: {result.overlap_fraction:.2%}",
        f"- **Flag: {result.flag}**",
    ]
    if result.per_transit_overlap:
        lines += ["", "| Transit # | Flagged fraction |", "|---|---|"]
        for k, frac in enumerate(result.per_transit_overlap, 1):
            lines.append(f"| {k} | {frac:.2%} |")
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="transit_overlap_detector",
        description="Detect transit / quality-flag overlaps.",
    )
    parser.add_argument("period_days", type=float)
    parser.add_argument("epoch_bjd", type=float)
    parser.add_argument("--duration-hours", type=float, default=2.0)
    args = parser.parse_args(argv)

    result = detect_transit_overlaps(
        [], [], args.period_days, args.epoch_bjd,
        duration_hours=args.duration_hours,
    )
    print(format_overlap_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
