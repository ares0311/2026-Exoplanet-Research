"""Calculate recommended period search range from light curve baseline and cadence."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PeriodSearchRangeResult:
    baseline_days: float
    cadence_minutes: float
    period_min_days: float
    period_max_days: float
    n_periods: int
    frequency_resolution_per_day: float
    flag: str


def compute_search_range(
    baseline_days: float,
    cadence_minutes: float,
    min_transit_duration_hours: float = 0.5,
    max_period_fraction: float = 0.5,
    oversample_factor: float = 3.0,
) -> PeriodSearchRangeResult:
    """
    Compute recommended BLS period search range.

    period_min: at least 2 × min_transit_duration (need to sample the transit).
    period_max: max_period_fraction × baseline (need at least 2 transits).
    n_periods: oversample_factor × baseline × freq_resolution grid points.

    Nyquist-limited: period_min >= 2 × cadence.
    """
    if not math.isfinite(baseline_days) or baseline_days <= 0.0:
        return PeriodSearchRangeResult(
            baseline_days=baseline_days, cadence_minutes=cadence_minutes,
            period_min_days=float("nan"), period_max_days=float("nan"),
            n_periods=0, frequency_resolution_per_day=float("nan"),
            flag="INVALID_BASELINE",
        )
    if not math.isfinite(cadence_minutes) or cadence_minutes <= 0.0:
        return PeriodSearchRangeResult(
            baseline_days=baseline_days, cadence_minutes=cadence_minutes,
            period_min_days=float("nan"), period_max_days=float("nan"),
            n_periods=0, frequency_resolution_per_day=float("nan"),
            flag="INVALID_CADENCE",
        )

    cadence_days = cadence_minutes / 1440.0
    nyquist_min = 2.0 * cadence_days

    # Minimum period: based on transit duration requirement
    transit_min_days = min_transit_duration_hours / 24.0
    period_min_duration = 2.0 * transit_min_days
    period_min = max(period_min_duration, nyquist_min, 0.5)

    # Maximum period: must observe at least 2 transits
    period_max = max_period_fraction * baseline_days
    if period_max <= period_min:
        period_max = baseline_days * 0.5
    period_max = max(period_max, period_min * 2.0)

    # Frequency resolution: df = 1 / baseline
    freq_res = 1.0 / baseline_days

    # Number of grid points: oversample the frequency grid
    f_min = 1.0 / period_max
    f_max = 1.0 / period_min
    n_periods = max(int(oversample_factor * (f_max - f_min) / freq_res), 100)

    return PeriodSearchRangeResult(
        baseline_days=baseline_days,
        cadence_minutes=cadence_minutes,
        period_min_days=round(period_min, 5),
        period_max_days=round(period_max, 3),
        n_periods=n_periods,
        frequency_resolution_per_day=round(freq_res, 6),
        flag="OK",
    )


def format_search_range(r: PeriodSearchRangeResult) -> str:
    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| Baseline (days) | {r.baseline_days:.2f} |\n"
        f"| Cadence (min) | {r.cadence_minutes:.1f} |\n"
        f"| Period min (days) | {r.period_min_days:.5f} |\n"
        f"| Period max (days) | {r.period_max_days:.3f} |\n"
        f"| N periods in grid | {r.n_periods} |\n"
        f"| Freq resolution (1/day) | {r.frequency_resolution_per_day:.6f} |\n"
        f"| Flag | {r.flag} |\n"
    )


def _cli() -> int:
    p = argparse.ArgumentParser(description="Compute BLS period search range.")
    p.add_argument("baseline_days", type=float)
    p.add_argument("cadence_minutes", type=float)
    p.add_argument("--min-transit-duration-hours", type=float, default=0.5)
    p.add_argument("--max-period-fraction", type=float, default=0.5)
    args = p.parse_args()
    r = compute_search_range(
        args.baseline_days, args.cadence_minutes,
        min_transit_duration_hours=args.min_transit_duration_hours,
        max_period_fraction=args.max_period_fraction,
    )
    print(format_search_range(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
