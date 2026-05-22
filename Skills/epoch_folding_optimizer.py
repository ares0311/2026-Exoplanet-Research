"""Optimise the transit reference epoch by minimising O-C RMS.

Given a set of measured transit mid-times and a fixed period, grid-searches
the reference epoch T0 to find the value that minimises the RMS of the
O-C (observed minus computed) residuals.

Complements ``period_refinement_calculator.py`` which refines the period
holding the epoch fixed.

Public API
----------
EpochOptResult(initial_epoch_bjd, optimised_epoch_bjd, oc_rms_minutes,
               epoch_uncertainty_days, n_transits, flag)
optimize_epoch(midpoints, period_days, epoch_init, *,
               n_grid, half_width_days) -> EpochOptResult
format_epoch_opt_result(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class EpochOptResult:
    initial_epoch_bjd: float
    optimised_epoch_bjd: float | None
    oc_rms_minutes: float | None
    epoch_uncertainty_days: float | None
    n_transits: int
    flag: str  # "OK" | "INSUFFICIENT" | "INVALID"


def _oc_rms_epoch(midpoints: list[float], period: float, epoch: float) -> float:
    """Compute O-C RMS (days) for given epoch."""
    rms_sq = 0.0
    for t in midpoints:
        n = round((t - epoch) / period)
        oc = t - (epoch + n * period)
        rms_sq += oc ** 2
    return math.sqrt(rms_sq / len(midpoints))


def optimize_epoch(
    midpoints: list[float],
    period_days: float,
    epoch_init: float,
    *,
    n_grid: int = 100,
    half_width_days: float | None = None,
) -> EpochOptResult:
    """Optimise reference epoch by minimising O-C RMS over a grid.

    Args:
        midpoints: Measured transit mid-times (BJD).
        period_days: Fixed orbital period (days).
        epoch_init: Initial epoch estimate (BJD).
        n_grid: Number of grid points to evaluate.
        half_width_days: Half-width of search grid (days).
            Defaults to half the period.

    Returns:
        :class:`EpochOptResult`.
    """
    if len(midpoints) < 2:
        return EpochOptResult(
            epoch_init, None, None, None, len(midpoints), "INSUFFICIENT"
        )
    if period_days <= 0:
        return EpochOptResult(epoch_init, None, None, None, 0, "INVALID")

    hw = half_width_days if half_width_days is not None else period_days / 2.0
    de = 2.0 * hw / max(n_grid - 1, 1)

    best_epoch = epoch_init
    best_rms = _oc_rms_epoch(midpoints, period_days, epoch_init)

    for i in range(n_grid):
        ep = epoch_init - hw + i * de
        rms = _oc_rms_epoch(midpoints, period_days, ep)
        if rms < best_rms:
            best_rms = rms
            best_epoch = ep

    # Parabolic interpolation for uncertainty
    idx_best = round((best_epoch - (epoch_init - hw)) / de)
    uncertainty: float | None = None
    if 1 <= idx_best < n_grid - 1:
        e_lo = epoch_init - hw + (idx_best - 1) * de
        e_hi = epoch_init - hw + (idx_best + 1) * de
        rms_lo = _oc_rms_epoch(midpoints, period_days, e_lo)
        rms_hi = _oc_rms_epoch(midpoints, period_days, e_hi)
        denom = rms_lo - 2 * best_rms + rms_hi
        if abs(denom) > 1e-20 and best_rms > 0:
            uncertainty = abs(de * math.sqrt(abs(best_rms / denom)))

    return EpochOptResult(
        initial_epoch_bjd=round(epoch_init, 8),
        optimised_epoch_bjd=round(best_epoch, 8),
        oc_rms_minutes=round(best_rms * 1440.0, 4),
        epoch_uncertainty_days=round(uncertainty, 8) if uncertainty is not None else None,
        n_transits=len(midpoints),
        flag="OK",
    )


def format_epoch_opt_result(result: EpochOptResult) -> str:
    """Format epoch optimisation result as Markdown."""
    lines = [
        "## Epoch Folding Optimizer",
        "",
        f"- Initial epoch: {result.initial_epoch_bjd} BJD",
        f"- Optimised epoch: {result.optimised_epoch_bjd} BJD",
        f"- O-C RMS: {result.oc_rms_minutes} minutes",
        f"- Epoch uncertainty: {result.epoch_uncertainty_days} days",
        f"- Transit midpoints used: {result.n_transits}",
        f"- **Flag: {result.flag}**",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="epoch_folding_optimizer",
        description="Optimise transit reference epoch by minimising O-C RMS.",
    )
    parser.add_argument("period_days", type=float)
    parser.add_argument("epoch_init", type=float)
    parser.add_argument("--n-grid", type=int, default=100)
    args = parser.parse_args(argv)

    result = optimize_epoch([], args.period_days, args.epoch_init, n_grid=args.n_grid)
    print(format_epoch_opt_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
