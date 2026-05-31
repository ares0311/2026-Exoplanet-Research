"""Analyze transit timing O-C residuals for trends and TTV signatures."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TimingResidualResult:
    n_transits: int
    mean_minutes: float
    rms_minutes: float
    max_abs_minutes: float
    trend_slope_min_per_transit: float
    n_outliers: int
    flag: str


def analyze_timing_residuals(
    oc_minutes: list[float],
    transit_numbers: list[int] | None = None,
    outlier_sigma: float = 3.0,
    trend_threshold_min_per_transit: float = 0.5,
) -> TimingResidualResult:
    """
    Analyze O-C residuals: mean, RMS, max, linear trend, outlier count.

    Flags: FLAT (well-behaved) / TREND (significant linear drift) /
           TTV_CANDIDATE (large scatter without trend).
    """
    n = len(oc_minutes)
    if n < 3:
        return TimingResidualResult(
            n_transits=n, mean_minutes=float("nan"), rms_minutes=float("nan"),
            max_abs_minutes=float("nan"), trend_slope_min_per_transit=float("nan"),
            n_outliers=0, flag="INSUFFICIENT_DATA",
        )

    ns = list(transit_numbers) if transit_numbers is not None else list(range(n))

    mean_oc = sum(oc_minutes) / n
    rms = math.sqrt(sum((v - mean_oc) ** 2 for v in oc_minutes) / n)
    max_abs = max(abs(v) for v in oc_minutes)

    # Linear trend: O-C = slope * n + intercept (least-squares)
    n_bar = sum(ns) / n
    oc_bar = mean_oc
    snn = sum((k - n_bar) ** 2 for k in ns)
    sno = sum((ns[i] - n_bar) * (oc_minutes[i] - oc_bar) for i in range(n))
    slope = sno / snn if snn > 0.0 else 0.0
    intercept = oc_bar - slope * n_bar

    # Detrended residuals
    detrended = [oc_minutes[i] - (slope * ns[i] + intercept) for i in range(n)]
    rms_detrended = math.sqrt(sum(v ** 2 for v in detrended) / n)

    # Outliers: |detrended| > outlier_sigma * rms_detrended
    if rms_detrended > 0.0:
        n_outliers = sum(1 for v in detrended if abs(v) > outlier_sigma * rms_detrended)
    else:
        n_outliers = 0

    # Classification
    has_trend = abs(slope) > trend_threshold_min_per_transit

    if has_trend:
        flag = "TREND"
    elif rms > 5.0 or n_outliers >= 2:
        flag = "TTV_CANDIDATE"
    else:
        flag = "FLAT"

    return TimingResidualResult(
        n_transits=n,
        mean_minutes=round(mean_oc, 4),
        rms_minutes=round(rms, 4),
        max_abs_minutes=round(max_abs, 4),
        trend_slope_min_per_transit=round(slope, 6),
        n_outliers=n_outliers,
        flag=flag,
    )


def format_timing_residual_result(r: TimingResidualResult) -> str:
    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| N transits | {r.n_transits} |\n"
        f"| Mean O-C (min) | {r.mean_minutes:.4f} |\n"
        f"| RMS O-C (min) | {r.rms_minutes:.4f} |\n"
        f"| Max |O-C| (min) | {r.max_abs_minutes:.4f} |\n"
        f"| Trend (min/transit) | {r.trend_slope_min_per_transit:.6f} |\n"
        f"| Outliers (>{r.n_transits > 0 and 3}σ) | {r.n_outliers} |\n"
        f"| Flag | {r.flag} |\n"
    )


def _cli() -> int:
    p = argparse.ArgumentParser(description="Analyze transit timing O-C residuals.")
    p.add_argument("oc_minutes", type=float, nargs="+", help="O-C residuals in minutes")
    p.add_argument("--outlier-sigma", type=float, default=3.0)
    p.add_argument("--trend-threshold", type=float, default=0.5)
    args = p.parse_args()
    r = analyze_timing_residuals(
        args.oc_minutes,
        outlier_sigma=args.outlier_sigma,
        trend_threshold_min_per_transit=args.trend_threshold,
    )
    print(format_timing_residual_result(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
