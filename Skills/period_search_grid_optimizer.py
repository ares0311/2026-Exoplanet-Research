"""Compute an optimal period search grid for BLS / Lomb-Scargle period searches."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PeriodGridResult:
    baseline_days: float
    period_min_days: float
    period_max_days: float
    n_periods: int
    frequency_resolution_per_day: float   # Δf = 1 / (π * T)
    period_spacing_at_min: float          # δP at P_min
    period_spacing_at_max: float          # δP at P_max
    oversampling_factor: float
    flag: str


def compute_period_search_grid(
    baseline_days: float,
    period_min_days: float = 0.5,
    period_max_days: float = 15.0,
    oversampling: float = 1.0,
) -> PeriodGridResult:
    """
    Compute the optimal period grid spacing for a BLS transit search.

    Independent frequency spacing (Kovács et al. 2002):
      Δf = 1 / (π * T_baseline)
    So the period spacing at period P:
      δP(P) = P² * Δf = P² / (π * T_baseline)

    Total number of trial periods (integrating over frequency range):
      N = π * T_baseline * (1/P_min − 1/P_max) * oversampling

    Parameters
    ----------
    baseline_days:   Total observation baseline in days.
    period_min_days: Minimum trial period in days.
    period_max_days: Maximum trial period in days.
    oversampling:    Oversampling factor (default 1.0 = Nyquist).
    """
    if not math.isfinite(baseline_days) or baseline_days <= 0:
        return PeriodGridResult(baseline_days, period_min_days, period_max_days,
                                0, float("nan"), float("nan"), float("nan"),
                                oversampling, "INVALID_BASELINE")
    if not math.isfinite(period_min_days) or period_min_days <= 0:
        return PeriodGridResult(baseline_days, period_min_days, period_max_days,
                                0, float("nan"), float("nan"), float("nan"),
                                oversampling, "INVALID_PERIOD_MIN")
    if not math.isfinite(period_max_days) or period_max_days <= period_min_days:
        return PeriodGridResult(baseline_days, period_min_days, period_max_days,
                                0, float("nan"), float("nan"), float("nan"),
                                oversampling, "INVALID_PERIOD_MAX")

    delta_f = 1.0 / (math.pi * baseline_days)

    dp_at_min = period_min_days ** 2 * delta_f
    dp_at_max = period_max_days ** 2 * delta_f

    n_periods = int(math.ceil(
        math.pi * baseline_days * oversampling * (1.0 / period_min_days - 1.0 / period_max_days)
    ))

    return PeriodGridResult(
        baseline_days=baseline_days,
        period_min_days=period_min_days,
        period_max_days=period_max_days,
        n_periods=n_periods,
        frequency_resolution_per_day=round(delta_f, 12),
        period_spacing_at_min=round(dp_at_min, 8),
        period_spacing_at_max=round(dp_at_max, 6),
        oversampling_factor=oversampling,
        flag="OK",
    )


def format_period_grid_result(r: PeriodGridResult) -> str:
    def _f(v: float, fmt: str = ".6f") -> str:
        return format(v, fmt) if math.isfinite(v) else "N/A"

    return (
        f"| Parameter | Value |\n|---|---|\n"
        f"| Baseline (days) | {_f(r.baseline_days, '.2f')} |\n"
        f"| Period range (days) | "
        f"{_f(r.period_min_days, '.4f')} – {_f(r.period_max_days, '.4f')} |\n"
        f"| N trial periods | {r.n_periods} |\n"
        f"| Frequency resolution (1/day) | {_f(r.frequency_resolution_per_day)} |\n"
        f"| δP at P_min (days) | {_f(r.period_spacing_at_min)} |\n"
        f"| δP at P_max (days) | {_f(r.period_spacing_at_max)} |\n"
        f"| Oversampling | {_f(r.oversampling_factor, '.2f')} |\n"
        f"| Flag | {r.flag} |\n"
    )


def _cli() -> int:
    p = argparse.ArgumentParser(description="Compute optimal BLS period search grid.")
    p.add_argument("baseline_days", type=float)
    p.add_argument("--period-min", type=float, default=0.5)
    p.add_argument("--period-max", type=float, default=15.0)
    p.add_argument("--oversampling", type=float, default=1.0)
    args = p.parse_args()
    r = compute_period_search_grid(
        args.baseline_days, args.period_min, args.period_max, args.oversampling,
    )
    print(format_period_grid_result(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
