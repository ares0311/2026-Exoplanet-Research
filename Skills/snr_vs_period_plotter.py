"""Compute SNR as a function of trial period for a grid of BLS tests.

Builds a period-vs-SNR profile without requiring matplotlib — returns
structured data that can be plotted externally or used for diagnostics.

Public API
----------
PeriodSNRPoint(period_days, snr, n_transits_expected)
PeriodSNRResult(n_periods, peak_period_days, peak_snr,
                points, flag)
compute_period_snr(time, flux, period_days_grid, *, flux_err,
                   duration_hours, transit_half_width) -> PeriodSNRResult
format_period_snr_result(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PeriodSNRPoint:
    period_days: float
    snr: float | None
    n_transits_expected: int


@dataclass(frozen=True)
class PeriodSNRResult:
    n_periods: int
    peak_period_days: float | None
    peak_snr: float | None
    points: tuple[PeriodSNRPoint, ...]
    flag: str  # "OK" | "INSUFFICIENT" | "INVALID"


def _phase_fold(time: list[float], epoch: float, period: float) -> list[float]:
    phases = []
    for t in time:
        ph = ((t - epoch) % period) / period
        if ph >= 0.5:
            ph -= 1.0
        phases.append(ph)
    return phases


def compute_period_snr(
    time: list[float],
    flux: list[float],
    period_days_grid: list[float],
    *,
    flux_err: list[float] | None = None,
    duration_hours: float = 2.0,
    transit_half_width: float = 0.05,
) -> PeriodSNRResult:
    """Compute SNR at each trial period via phase-bin depth / scatter.

    Args:
        time: Time array (days).
        flux: Normalised flux array, same length as time.
        period_days_grid: Trial periods to evaluate (days).
        flux_err: Per-point uncertainties.  Defaults to MAD-based estimate.
        duration_hours: Expected transit duration (hours).
        transit_half_width: Phase half-width for in-transit bin.

    Returns:
        :class:`PeriodSNRResult`.
    """
    n = len(time)
    if n < 5 or len(flux) != n:
        return PeriodSNRResult(0, None, None, (), "INVALID")
    if not period_days_grid:
        return PeriodSNRResult(0, None, None, (), "INVALID")

    # Default flux_err from MAD
    if flux_err is None or len(flux_err) != n:
        med = sorted(flux)[n // 2]
        mad = sorted(abs(f - med) for f in flux)[n // 2]
        sigma = max(mad * 1.4826, 1e-9)
        flux_err = [sigma] * n

    time_span = max(time) - min(time) if max(time) > min(time) else 1.0

    points: list[PeriodSNRPoint] = []

    for p in period_days_grid:
        if p <= 0:
            points.append(PeriodSNRPoint(p, None, 0))
            continue

        n_transits = max(1, int(time_span / p))
        epoch = time[0]
        phases = _phase_fold(time, epoch, p)

        in_flux: list[float] = []
        in_err_sq: list[float] = []
        oot_flux: list[float] = []
        oot_err_sq: list[float] = []
        hw = transit_half_width

        for i, ph in enumerate(phases):
            if abs(ph) <= hw:
                in_flux.append(flux[i])
                in_err_sq.append(flux_err[i] ** 2)
            else:
                oot_flux.append(flux[i])
                oot_err_sq.append(flux_err[i] ** 2)

        if len(in_flux) < 1 or len(oot_flux) < 2:
            points.append(PeriodSNRPoint(p, None, n_transits))
            continue

        in_mean = sum(in_flux) / len(in_flux)
        oot_mean = sum(oot_flux) / len(oot_flux)
        depth = oot_mean - in_mean

        noise_in = math.sqrt(sum(in_err_sq)) / len(in_flux)
        noise_oot = math.sqrt(sum(oot_err_sq)) / len(oot_flux)
        noise = math.sqrt(noise_in ** 2 + noise_oot ** 2)

        snr = depth / noise if noise > 1e-20 else None
        points.append(PeriodSNRPoint(p, round(snr, 4) if snr is not None else None, n_transits))

    valid = [pt for pt in points if pt.snr is not None]
    if not valid:
        return PeriodSNRResult(len(period_days_grid), None, None, tuple(points), "INSUFFICIENT")

    best = max(valid, key=lambda pt: pt.snr)  # type: ignore[arg-type]
    return PeriodSNRResult(
        n_periods=len(period_days_grid),
        peak_period_days=best.period_days,
        peak_snr=best.snr,
        points=tuple(points),
        flag="OK",
    )


def format_period_snr_result(result: PeriodSNRResult) -> str:
    """Format period-SNR result as Markdown."""
    lines = [
        "## Period vs SNR Profile",
        "",
        f"- Periods tested: {result.n_periods}",
        f"- Peak period: {result.peak_period_days} days",
        f"- Peak SNR: {result.peak_snr}",
        f"- **Flag: {result.flag}**",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="snr_vs_period_plotter",
        description="Compute SNR vs trial period grid.",
    )
    parser.add_argument("--period-min", type=float, default=1.0)
    parser.add_argument("--period-max", type=float, default=10.0)
    parser.add_argument("--n-periods", type=int, default=20)
    args = parser.parse_args(argv)

    step = (args.period_max - args.period_min) / max(args.n_periods - 1, 1)
    grid = [args.period_min + i * step for i in range(args.n_periods)]
    result = compute_period_snr([], [], grid)
    print(format_period_snr_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
