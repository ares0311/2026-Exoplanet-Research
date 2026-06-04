"""Detect linear and quadratic RV trends indicative of outer companions."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RvTrendResult:
    n_points: int
    linear_slope_ms_per_day: float
    quadratic_coeff_ms_per_day2: float
    rms_residual_ms: float
    trend_significance: float
    trend_detected: bool
    flag: str


def _least_squares_linear(x: list[float], y: list[float]) -> tuple[float, float, float]:
    """Return (slope, intercept, rms_residual) for a linear fit."""
    n = len(x)
    sx = sum(x)
    sy = sum(y)
    sxx = sum(xi**2 for xi in x)
    sxy = sum(xi * yi for xi, yi in zip(x, y, strict=False))
    det = n * sxx - sx**2
    if abs(det) < 1e-12:
        return 0.0, sy / n if n > 0 else 0.0, 0.0
    slope = (n * sxy - sx * sy) / det
    intercept = (sy - slope * sx) / n
    residuals = [yi - (slope * xi + intercept) for xi, yi in zip(x, y, strict=False)]
    rms = math.sqrt(sum(r**2 for r in residuals) / n) if n > 0 else 0.0
    return slope, intercept, rms


def _least_squares_quadratic(
    x: list[float], y: list[float]
) -> tuple[float, float, float, float]:
    """Return (a2, a1, a0, rms_residual) for ax²+bx+c fit via normal equations."""
    n = len(x)
    if n < 3:
        return 0.0, 0.0, 0.0, 0.0

    s0 = float(n)
    s1 = sum(x)
    s2 = sum(xi**2 for xi in x)
    s3 = sum(xi**3 for xi in x)
    s4 = sum(xi**4 for xi in x)
    t0 = sum(y)
    t1 = sum(xi * yi for xi, yi in zip(x, y, strict=False))
    t2 = sum(xi**2 * yi for xi, yi in zip(x, y, strict=False))

    # Solve 3x3 normal equations via Cramer's rule
    def det3(a: list[list[float]]) -> float:
        return (
            a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
            - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
            + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
        )

    M = [[s4, s3, s2], [s3, s2, s1], [s2, s1, s0]]
    d = det3(M)
    if abs(d) < 1e-20:
        slope, intercept, rms = _least_squares_linear(x, y)
        return 0.0, slope, intercept, rms

    def replace_col(col: int, rhs: list[float]) -> list[list[float]]:
        m2 = [row[:] for row in M]
        for i in range(3):
            m2[i][col] = rhs[i]
        return m2

    rhs = [t2, t1, t0]
    a2 = det3(replace_col(0, rhs)) / d
    a1 = det3(replace_col(1, rhs)) / d
    a0 = det3(replace_col(2, rhs)) / d

    residuals = [yi - (a2 * xi**2 + a1 * xi + a0) for xi, yi in zip(x, y, strict=False)]
    rms = math.sqrt(sum(r**2 for r in residuals) / n)
    return a2, a1, a0, rms


def detect_rv_trend(
    times_bjd: list[float],
    rv_ms: list[float],
    rv_err_ms: list[float] | None = None,
    significance_threshold: float = 3.0,
) -> RvTrendResult:
    """
    Detect linear or quadratic RV trends from a time series.

    Fits a quadratic ax²+bx+c; compares slope to noise floor.
    trend_detected when |linear_slope| / (rms / sqrt(n)) > threshold.
    """
    n = len(times_bjd)
    if n < 3:
        return RvTrendResult(
            n_points=n, linear_slope_ms_per_day=float("nan"),
            quadratic_coeff_ms_per_day2=float("nan"), rms_residual_ms=float("nan"),
            trend_significance=float("nan"), trend_detected=False,
            flag="INSUFFICIENT_DATA",
        )
    if len(rv_ms) != n:
        return RvTrendResult(
            n_points=n, linear_slope_ms_per_day=float("nan"),
            quadratic_coeff_ms_per_day2=float("nan"), rms_residual_ms=float("nan"),
            trend_significance=float("nan"), trend_detected=False,
            flag="LENGTH_MISMATCH",
        )

    # Center times to improve numerical stability
    t0 = sum(times_bjd) / n
    tx = [t - t0 for t in times_bjd]

    a2, a1, _a0, rms = _least_squares_quadratic(tx, rv_ms)

    # Significance of linear slope
    noise_floor = rms / math.sqrt(n) if n > 0 else float("inf")
    significance = abs(a1) / noise_floor if noise_floor > 0 else 0.0
    trend_detected = significance > significance_threshold

    return RvTrendResult(
        n_points=n,
        linear_slope_ms_per_day=round(a1, 6),
        quadratic_coeff_ms_per_day2=round(a2, 8),
        rms_residual_ms=round(rms, 4),
        trend_significance=round(significance, 3),
        trend_detected=trend_detected,
        flag="TREND_DETECTED" if trend_detected else "OK",
    )


def format_rv_trend_result(r: RvTrendResult) -> str:
    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| N RV points | {r.n_points} |\n"
        f"| Linear slope (m/s/day) | {r.linear_slope_ms_per_day:.6f} |\n"
        f"| Quadratic coeff (m/s/day²) | {r.quadratic_coeff_ms_per_day2:.8f} |\n"
        f"| RMS residual (m/s) | {r.rms_residual_ms:.4f} |\n"
        f"| Trend significance (σ) | {r.trend_significance:.3f} |\n"
        f"| Trend detected | {r.trend_detected} |\n"
        f"| Flag | {r.flag} |\n"
    )


def _cli() -> int:
    p = argparse.ArgumentParser(description="Detect RV trend from time series.")
    p.add_argument("times_json", help="JSON array of BJD times")
    p.add_argument("rv_json", help="JSON array of RV values (m/s)")
    args = p.parse_args()
    import json
    times = json.loads(args.times_json)
    rv = json.loads(args.rv_json)
    r = detect_rv_trend(times, rv)
    print(format_rv_trend_result(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
