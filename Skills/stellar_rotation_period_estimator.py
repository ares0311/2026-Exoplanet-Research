"""Estimate stellar rotation period from light-curve autocorrelation.

Public API:
    RotationEstimateResult  -- frozen dataclass
    estimate_rotation_period(flux, cadence_days, ...) -> RotationEstimateResult
    format_rotation_estimate(result) -> str
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RotationEstimateResult:
    period_days: float
    acf_peak_lag: float
    alias_check: bool
    quality: str
    flag: str


def _pearson_r(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n < 2:
        return 0.0
    mx = sum(x) / n
    my = sum(y) / n
    dx = [v - mx for v in x]
    dy = [v - my for v in y]
    num = sum(a * b for a, b in zip(dx, dy, strict=False))
    sx = math.sqrt(sum(a * a for a in dx))
    sy = math.sqrt(sum(b * b for b in dy))
    if sx == 0.0 or sy == 0.0:
        return 0.0
    return num / (sx * sy)


def estimate_rotation_period(
    flux: list[float],
    cadence_days: float,
    *,
    max_lag_days: float = 30.0,
    min_period_days: float = 0.5,
) -> RotationEstimateResult:
    if len(flux) < 10:
        return RotationEstimateResult(
            period_days=0.0, acf_peak_lag=0.0, alias_check=False,
            quality="poor", flag="INSUFFICIENT_DATA",
        )

    max_lag_steps = min(len(flux) - 1, int(max_lag_days / cadence_days))
    acf: dict[int, float] = {}
    for k in range(1, max_lag_steps + 1):
        r = _pearson_r(flux[:-k], flux[k:])
        acf[k] = r

    # Find first local maximum with positive ACF after lag 1
    peak_k = -1
    peak_r = -999.0
    for k in range(2, max_lag_steps - 1):
        if k not in acf or (k - 1) not in acf or (k + 1) not in acf:
            continue
        if acf[k] > acf[k - 1] and acf[k] > acf[k + 1] and acf[k] > 0.0:
            peak_k = k
            peak_r = acf[k]
            break

    if peak_k == -1:
        return RotationEstimateResult(
            period_days=0.0, acf_peak_lag=0.0, alias_check=False,
            quality="poor", flag="NO_PERIOD_FOUND",
        )

    acf_peak_lag = peak_k * cadence_days
    period_days = acf_peak_lag

    if period_days < min_period_days:
        return RotationEstimateResult(
            period_days=period_days, acf_peak_lag=acf_peak_lag, alias_check=False,
            quality="poor", flag="PERIOD_TOO_SHORT",
        )

    # Alias check: look for peak near 2× period that is larger
    double_k = int(peak_k * 2)
    alias_check = False
    for dk in range(max(1, double_k - 2), min(max_lag_steps, double_k + 3)):
        if dk in acf and acf[dk] > peak_r:
            alias_check = True
            break

    if peak_r > 0.5:
        quality = "good"
    elif peak_r > 0.2:
        quality = "fair"
    else:
        quality = "poor"

    return RotationEstimateResult(
        period_days=period_days,
        acf_peak_lag=acf_peak_lag,
        alias_check=alias_check,
        quality=quality,
        flag="OK",
    )


def format_rotation_estimate(result: RotationEstimateResult) -> str:
    lines = [
        "## Stellar Rotation Period Estimate",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Period (days) | {result.period_days:.4f} |",
        f"| ACF Peak Lag (days) | {result.acf_peak_lag:.4f} |",
        f"| Alias Check | {result.alias_check} |",
        f"| Quality | {result.quality} |",
        f"| Flag | {result.flag} |",
    ]
    return "\n".join(lines)


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Estimate stellar rotation period.")
    parser.add_argument("flux_file", help="JSON file with flux values.")
    parser.add_argument("cadence_days", type=float)
    parser.add_argument("--max-lag-days", type=float, default=30.0)
    parser.add_argument("--min-period-days", type=float, default=0.5)
    args = parser.parse_args()
    with open(args.flux_file) as fh:
        flux = json.load(fh)
    result = estimate_rotation_period(
        flux, args.cadence_days,
        max_lag_days=args.max_lag_days,
        min_period_days=args.min_period_days,
    )
    print(format_rotation_estimate(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
