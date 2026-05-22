"""Refine a period estimate by minimising O-C (observed minus computed) RMS.

Given a list of measured transit midpoints and an initial period estimate,
scans a fine grid around the estimate and returns the period that minimises
the RMS of the O-C residuals.

Public API
----------
PeriodRefinementResult(initial_period_days, refined_period_days, oc_rms_minutes,
                       period_uncertainty_days, n_transits, flag)
refine_period_from_oc(midpoints, initial_period_days, initial_epoch_bjd, *,
                      search_half_width_frac,
                      n_grid) -> PeriodRefinementResult
format_refinement_result(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PeriodRefinementResult:
    initial_period_days: float
    refined_period_days: float | None
    oc_rms_minutes: float | None      # RMS of O-C residuals in minutes
    period_uncertainty_days: float | None
    n_transits: int
    flag: str  # "OK" | "INSUFFICIENT" | "INVALID"


def _oc_rms(midpoints: list[float], period: float, epoch: float) -> float:
    """Compute RMS of O-C residuals in days."""
    rms_sq = 0.0
    for t in midpoints:
        n = round((t - epoch) / period)
        oc = t - (epoch + n * period)
        rms_sq += oc ** 2
    return math.sqrt(rms_sq / len(midpoints))


def refine_period_from_oc(
    midpoints: list[float],
    initial_period_days: float,
    initial_epoch_bjd: float,
    *,
    search_half_width_frac: float = 0.01,
    n_grid: int = 100,
) -> PeriodRefinementResult:
    """Refine period by minimising O-C RMS over a fine grid.

    Args:
        midpoints: List of measured transit mid-times (BJD).
        initial_period_days: Initial period estimate (days).
        initial_epoch_bjd: Reference epoch (BJD).
        search_half_width_frac: Half-width of grid as fraction of period.
        n_grid: Number of grid points to evaluate.

    Returns:
        :class:`PeriodRefinementResult`.
    """
    if len(midpoints) < 2:
        return PeriodRefinementResult(
            initial_period_days, None, None, None, len(midpoints), "INSUFFICIENT"
        )
    if initial_period_days <= 0:
        return PeriodRefinementResult(initial_period_days, None, None, None, 0, "INVALID")

    half_w = initial_period_days * search_half_width_frac
    p_min = initial_period_days - half_w
    p_max = initial_period_days + half_w
    dp = (p_max - p_min) / max(n_grid - 1, 1)

    best_period = initial_period_days
    best_rms = _oc_rms(midpoints, initial_period_days, initial_epoch_bjd)

    for i in range(n_grid):
        p = p_min + i * dp
        if p <= 0:
            continue
        rms = _oc_rms(midpoints, p, initial_epoch_bjd)
        if rms < best_rms:
            best_rms = rms
            best_period = p

    # Period uncertainty from parabolic interpolation around minimum
    idx_best = round((best_period - p_min) / dp)
    uncertainty: float | None = None
    if 1 <= idx_best < n_grid - 1:
        p_lo = p_min + (idx_best - 1) * dp
        p_hi = p_min + (idx_best + 1) * dp
        rms_lo = _oc_rms(midpoints, p_lo, initial_epoch_bjd)
        rms_hi = _oc_rms(midpoints, p_hi, initial_epoch_bjd)
        denom = rms_lo - 2 * best_rms + rms_hi
        if abs(denom) > 1e-20:
            uncertainty = abs(dp * math.sqrt(abs(best_rms / denom))) if best_rms > 0 else dp

    return PeriodRefinementResult(
        initial_period_days=round(initial_period_days, 6),
        refined_period_days=round(best_period, 8),
        oc_rms_minutes=round(best_rms * 1440.0, 4),
        period_uncertainty_days=round(uncertainty, 8) if uncertainty is not None else None,
        n_transits=len(midpoints),
        flag="OK",
    )


def format_refinement_result(result: PeriodRefinementResult) -> str:
    """Format period refinement result as Markdown."""
    lines = [
        "## Period Refinement (O-C)",
        "",
        f"- Initial period: {result.initial_period_days} days",
        f"- Refined period: {result.refined_period_days} days",
        f"- O-C RMS: {result.oc_rms_minutes} minutes",
        f"- Period uncertainty: {result.period_uncertainty_days} days",
        f"- Transit midpoints used: {result.n_transits}",
        f"- **Flag: {result.flag}**",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="period_refinement_calculator",
        description="Refine period estimate by minimising O-C RMS.",
    )
    parser.add_argument("period_days", type=float)
    parser.add_argument("epoch_bjd", type=float)
    args = parser.parse_args(argv)

    result = refine_period_from_oc([], args.period_days, args.epoch_bjd)
    print(format_refinement_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
