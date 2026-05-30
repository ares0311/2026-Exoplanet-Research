"""Estimate BLS period-search completeness for given parameters.

Public API:
    CompletenessResult  -- frozen dataclass
    estimate_search_completeness(period_days, depth_ppm, baseline_days,
                                 noise_ppm, gap_fraction) -> CompletenessResult
    format_completeness_result(result) -> str
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CompletenessResult:
    period_days: float
    depth_ppm: float
    expected_n_transits: float
    expected_snr: float
    completeness_fraction: float
    flag: str


def estimate_search_completeness(
    period_days: float,
    depth_ppm: float,
    baseline_days: float,
    noise_ppm: float,
    *,
    gap_fraction: float = 0.0,
    snr_threshold: float = 7.1,
) -> CompletenessResult:
    if period_days <= 0:
        return CompletenessResult(
            period_days=period_days, depth_ppm=depth_ppm,
            expected_n_transits=0.0, expected_snr=0.0,
            completeness_fraction=0.0, flag="INVALID_PERIOD",
        )
    if noise_ppm <= 0:
        return CompletenessResult(
            period_days=period_days, depth_ppm=depth_ppm,
            expected_n_transits=0.0, expected_snr=0.0,
            completeness_fraction=0.0, flag="INVALID_NOISE",
        )
    effective_baseline = baseline_days * (1.0 - gap_fraction)
    expected_n_transits = effective_baseline / period_days
    expected_snr = depth_ppm / noise_ppm * math.sqrt(max(expected_n_transits, 0.0))
    if expected_snr <= 0:
        completeness_fraction = 0.0
    elif expected_snr >= snr_threshold * 2:
        completeness_fraction = 1.0
    else:
        completeness_fraction = min(1.0, max(0.0, expected_snr / (snr_threshold * 2)))
    flag = "OK" if expected_snr >= snr_threshold else "BELOW_SNR_THRESHOLD"
    return CompletenessResult(
        period_days=period_days,
        depth_ppm=depth_ppm,
        expected_n_transits=expected_n_transits,
        expected_snr=expected_snr,
        completeness_fraction=completeness_fraction,
        flag=flag,
    )


def format_completeness_result(result: CompletenessResult) -> str:
    lines = [
        "## Period Search Completeness Estimate",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Period (days) | {result.period_days:.4f} |",
        f"| Depth (ppm) | {result.depth_ppm:.1f} |",
        f"| Expected N Transits | {result.expected_n_transits:.2f} |",
        f"| Expected SNR | {result.expected_snr:.2f} |",
        f"| Completeness Fraction | {result.completeness_fraction:.4f} |",
        f"| Flag | {result.flag} |",
    ]
    return "\n".join(lines)


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Estimate BLS search completeness.")
    parser.add_argument("period_days", type=float)
    parser.add_argument("depth_ppm", type=float)
    parser.add_argument("baseline_days", type=float)
    parser.add_argument("noise_ppm", type=float)
    parser.add_argument("--gap-fraction", type=float, default=0.0)
    parser.add_argument("--snr-threshold", type=float, default=7.1)
    args = parser.parse_args()
    result = estimate_search_completeness(
        args.period_days, args.depth_ppm, args.baseline_days, args.noise_ppm,
        gap_fraction=args.gap_fraction, snr_threshold=args.snr_threshold,
    )
    print(format_completeness_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
