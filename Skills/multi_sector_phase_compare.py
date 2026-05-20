"""Compare phase-folded transit shape across individual sectors.

This offline utility helps spot sector-specific behavior before candidates are
advanced for follow-up. It folds each sector independently, estimates a simple
transit depth and phase centroid, then flags depth or timing disagreement.

Public API
----------
SectorPhaseMetrics(sector, n_points, n_in_transit, depth_ppm, centroid_phase, flag)
PhaseComparisonResult(period_days, epoch_bjd, duration_days, sectors, flag)
compare_sector_phase_folds(sector_lcs, period, epoch, *, duration_days,
                           depth_tolerance_ppm, phase_tolerance) -> PhaseComparisonResult
format_phase_comparison(result) -> str
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SectorPhaseMetrics:
    sector: int
    n_points: int
    n_in_transit: int
    depth_ppm: float | None
    centroid_phase: float | None
    flag: str


@dataclass(frozen=True)
class PhaseComparisonResult:
    period_days: float
    epoch_bjd: float
    duration_days: float
    sectors: tuple[SectorPhaseMetrics, ...]
    max_depth_delta_ppm: float | None
    max_centroid_delta_phase: float | None
    flag: str


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    mid = len(sorted_values) // 2
    if len(sorted_values) % 2:
        return sorted_values[mid]
    return (sorted_values[mid - 1] + sorted_values[mid]) / 2.0


def _phase(time: float, period: float, epoch: float) -> float:
    folded = ((time - epoch) % period) / period
    return folded - 1.0 if folded >= 0.5 else folded


def _sector_metrics(
    lc: dict,
    *,
    index: int,
    period: float,
    epoch: float,
    duration_days: float,
    min_in_transit_points: int,
) -> SectorPhaseMetrics:
    time = [float(value) for value in lc.get("time", [])]
    flux = [float(value) for value in lc.get("flux", [])]
    sector = int(lc.get("sector", index + 1))

    if period <= 0 or duration_days <= 0 or not time or not flux:
        return SectorPhaseMetrics(sector, len(time), 0, None, None, "INSUFFICIENT")

    phases = [_phase(value, period, epoch) for value in time]
    half_duration_phase = min(0.5, duration_days / period / 2.0)
    oot_start_phase = min(0.5, half_duration_phase * 3.0)

    in_flux: list[float] = []
    out_flux: list[float] = []
    weighted_phase_sum = 0.0
    deficit_sum = 0.0

    baseline = _median(flux)
    if baseline is None:
        return SectorPhaseMetrics(sector, len(time), 0, None, None, "INSUFFICIENT")

    for ph, fl in zip(phases, flux, strict=False):
        if abs(ph) <= half_duration_phase:
            in_flux.append(fl)
            deficit = max(0.0, baseline - fl)
            weighted_phase_sum += ph * deficit
            deficit_sum += deficit
        elif abs(ph) >= oot_start_phase:
            out_flux.append(fl)

    if len(in_flux) < min_in_transit_points or not out_flux:
        return SectorPhaseMetrics(
            sector,
            len(time),
            len(in_flux),
            None,
            None,
            "INSUFFICIENT",
        )

    in_median = _median(in_flux)
    out_median = _median(out_flux)
    if in_median is None or out_median is None or out_median == 0.0:
        return SectorPhaseMetrics(
            sector,
            len(time),
            len(in_flux),
            None,
            None,
            "INSUFFICIENT",
        )

    depth_ppm = (out_median - in_median) / out_median * 1_000_000.0
    centroid_phase = weighted_phase_sum / deficit_sum if deficit_sum > 0.0 else 0.0
    flag = "OK" if depth_ppm > 0.0 else "WEAK_OR_INVERTED"
    return SectorPhaseMetrics(
        sector,
        len(time),
        len(in_flux),
        depth_ppm,
        centroid_phase,
        flag,
    )


def _max_pairwise_delta(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    return max(values) - min(values)


def compare_sector_phase_folds(
    sector_lcs: list[dict],
    period: float,
    epoch: float,
    *,
    duration_days: float,
    depth_tolerance_ppm: float = 500.0,
    phase_tolerance: float = 0.02,
    min_in_transit_points: int = 2,
) -> PhaseComparisonResult:
    """Compare per-sector phase-folded transit depth and phase centroid.

    Args:
        sector_lcs: List of dicts with ``time``, ``flux``, and optional
            ``sector`` fields.
        period: Transit period in days.
        epoch: Transit epoch in BJD-like units matching ``time``.
        duration_days: Transit duration in days.
        depth_tolerance_ppm: Maximum allowed sector-to-sector depth spread.
        phase_tolerance: Maximum allowed sector-to-sector centroid spread.
        min_in_transit_points: Minimum cadences required inside transit.

    Returns:
        :class:`PhaseComparisonResult`.
    """
    sectors = tuple(
        _sector_metrics(
            lc,
            index=index,
            period=period,
            epoch=epoch,
            duration_days=duration_days,
            min_in_transit_points=min_in_transit_points,
        )
        for index, lc in enumerate(sector_lcs)
    )
    depths = [metric.depth_ppm for metric in sectors if metric.depth_ppm is not None]
    centroids = [
        metric.centroid_phase for metric in sectors if metric.centroid_phase is not None
    ]
    max_depth_delta = _max_pairwise_delta(depths)
    max_centroid_delta = _max_pairwise_delta(centroids)

    if not sectors or not depths:
        flag = "INSUFFICIENT_SECTORS"
    elif any(metric.flag == "INSUFFICIENT" for metric in sectors):
        flag = "PARTIAL"
    elif len(depths) < 2:
        flag = "INSUFFICIENT_SECTORS"
    elif any(metric.flag == "WEAK_OR_INVERTED" for metric in sectors):
        flag = "WEAK_OR_INVERTED"
    elif max_depth_delta is not None and max_depth_delta > depth_tolerance_ppm:
        flag = "DEPTH_MISMATCH"
    elif max_centroid_delta is not None and max_centroid_delta > phase_tolerance:
        flag = "PHASE_SHIFT"
    else:
        flag = "CONSISTENT"

    return PhaseComparisonResult(
        period_days=period,
        epoch_bjd=epoch,
        duration_days=duration_days,
        sectors=sectors,
        max_depth_delta_ppm=max_depth_delta,
        max_centroid_delta_phase=max_centroid_delta,
        flag=flag,
    )


def _format_optional(value: float | None, precision: int) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{precision}f}"


def format_phase_comparison(result: PhaseComparisonResult) -> str:
    """Format a phase-fold comparison as Markdown."""
    lines = [
        "## Multi-Sector Phase-Fold Comparison",
        "",
        f"- Period: {result.period_days:.6f} d",
        f"- Epoch: {result.epoch_bjd:.6f}",
        f"- Duration: {result.duration_days:.6f} d",
        f"- Sectors compared: {len(result.sectors)}",
        f"- Overall flag: **{result.flag}**",
        f"- Max depth delta: {_format_optional(result.max_depth_delta_ppm, 1)} ppm",
        f"- Max centroid delta: {_format_optional(result.max_centroid_delta_phase, 5)} phase",
        "",
        "| Sector | Points | In transit | Depth (ppm) | Centroid phase | Flag |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for metric in result.sectors:
        lines.append(
            f"| {metric.sector} | {metric.n_points} | {metric.n_in_transit} | "
            f"{_format_optional(metric.depth_ppm, 1)} | "
            f"{_format_optional(metric.centroid_phase, 5)} | {metric.flag} |"
        )
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(
        prog="multi_sector_phase_compare",
        description="Compare phase-folded transit metrics across sectors.",
    )
    parser.add_argument("--lcs", nargs="+", required=True, metavar="JSON")
    parser.add_argument("--period", type=float, required=True)
    parser.add_argument("--epoch", type=float, required=True)
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--depth-tolerance-ppm", type=float, default=500.0)
    parser.add_argument("--phase-tolerance", type=float, default=0.02)
    args = parser.parse_args(argv)

    sector_lcs = [json.loads(Path(path).read_text()) for path in args.lcs]
    result = compare_sector_phase_folds(
        sector_lcs,
        args.period,
        args.epoch,
        duration_days=args.duration,
        depth_tolerance_ppm=args.depth_tolerance_ppm,
        phase_tolerance=args.phase_tolerance,
    )
    print(format_phase_comparison(result))
    return 0 if result.flag in {"CONSISTENT", "PARTIAL"} else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
