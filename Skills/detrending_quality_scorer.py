"""Score detrending quality via residual autocorrelation.

Public API:
    DetrendingQualityResult  -- frozen dataclass
    score_detrending_quality(residuals) -> DetrendingQualityResult
    format_detrending_quality(result) -> str
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class DetrendingQualityResult:
    lag1_acf: float
    durbin_watson: float
    rms_residual: float
    quality_score: float
    flag: str


def score_detrending_quality(residuals: list[float]) -> DetrendingQualityResult:
    if len(residuals) < 4:
        return DetrendingQualityResult(
            lag1_acf=0.0, durbin_watson=2.0, rms_residual=0.0,
            quality_score=0.0, flag="INSUFFICIENT_DATA",
        )
    n = len(residuals)
    mean_r = sum(residuals) / n
    centered = [r - mean_r for r in residuals]
    ss = sum(c * c for c in centered)
    rms_residual = math.sqrt(ss / n)

    if ss == 0.0:
        lag1_acf = 0.0
        durbin_watson = 2.0
    else:
        lag1_acf = sum(centered[i] * centered[i + 1] for i in range(n - 1)) / ss
        diff_sq = sum((residuals[i + 1] - residuals[i]) ** 2 for i in range(n - 1))
        durbin_watson = diff_sq / ss if ss > 0 else 2.0

    abs_acf = abs(lag1_acf)
    dw_deviation = abs(durbin_watson - 2.0) / 2.0
    quality_score = max(0.0, 1.0 - abs_acf - 0.5 * dw_deviation)

    if abs_acf > 0.3:
        flag = "HIGH_AUTOCORRELATION"
    elif abs_acf > 0.1:
        flag = "MODERATE_AUTOCORRELATION"
    else:
        flag = "OK"

    return DetrendingQualityResult(
        lag1_acf=lag1_acf,
        durbin_watson=durbin_watson,
        rms_residual=rms_residual,
        quality_score=quality_score,
        flag=flag,
    )


def format_detrending_quality(result: DetrendingQualityResult) -> str:
    lines = [
        "## Detrending Quality Score",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Lag-1 ACF | {result.lag1_acf:.4f} |",
        f"| Durbin-Watson | {result.durbin_watson:.4f} |",
        f"| RMS Residual | {result.rms_residual:.6f} |",
        f"| Quality Score | {result.quality_score:.4f} |",
        f"| Flag | {result.flag} |",
    ]
    return "\n".join(lines)


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Score detrending quality.")
    parser.add_argument("residuals_file", help="JSON file of residual values.")
    args = parser.parse_args()
    with open(args.residuals_file) as fh:
        residuals = json.load(fh)
    result = score_detrending_quality(residuals)
    print(format_detrending_quality(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
