"""Validate period / semi-major axis / stellar mass self-consistency via Kepler's 3rd law.

Public API:
    KeplerCheckResult  -- frozen dataclass
    check_kepler_third_law(period_days, semi_major_axis_au, stellar_mass_msun) -> KeplerCheckResult
    format_kepler_check_result(result) -> str
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

# G in AU^3 / (M_sun * day^2)
_G_AU3_MSUN_DAY2 = 2.959122e-4


@dataclass(frozen=True)
class KeplerCheckResult:
    period_days: float
    semi_major_axis_au: float
    stellar_mass_msun: float
    predicted_period_days: float
    predicted_sma_au: float
    period_residual_days: float
    sma_residual_au: float
    consistent: bool
    flag: str


def check_kepler_third_law(
    period_days: float,
    semi_major_axis_au: float,
    stellar_mass_msun: float,
    tolerance: float = 0.05,
) -> KeplerCheckResult:
    if period_days <= 0:
        return KeplerCheckResult(
            period_days=period_days, semi_major_axis_au=semi_major_axis_au,
            stellar_mass_msun=stellar_mass_msun, predicted_period_days=0.0,
            predicted_sma_au=0.0, period_residual_days=0.0, sma_residual_au=0.0,
            consistent=False, flag="INVALID_PERIOD",
        )
    if semi_major_axis_au <= 0:
        return KeplerCheckResult(
            period_days=period_days, semi_major_axis_au=semi_major_axis_au,
            stellar_mass_msun=stellar_mass_msun, predicted_period_days=0.0,
            predicted_sma_au=0.0, period_residual_days=0.0, sma_residual_au=0.0,
            consistent=False, flag="INVALID_SMA",
        )
    if stellar_mass_msun <= 0:
        return KeplerCheckResult(
            period_days=period_days, semi_major_axis_au=semi_major_axis_au,
            stellar_mass_msun=stellar_mass_msun, predicted_period_days=0.0,
            predicted_sma_au=0.0, period_residual_days=0.0, sma_residual_au=0.0,
            consistent=False, flag="INVALID_STELLAR_MASS",
        )
    # P^2 = (4 pi^2 / G M) a^3  =>  P = 2 pi sqrt(a^3 / G M)
    predicted_period = (
        2.0 * math.pi * math.sqrt(semi_major_axis_au ** 3 / (_G_AU3_MSUN_DAY2 * stellar_mass_msun))
    )
    # a^3 = G M P^2 / (4 pi^2)
    predicted_sma = (
        (
            _G_AU3_MSUN_DAY2 * stellar_mass_msun * period_days ** 2
            / (4.0 * math.pi ** 2)
        ) ** (1.0 / 3.0)
    )
    period_res = abs(period_days - predicted_period)
    sma_res = abs(semi_major_axis_au - predicted_sma)
    rel_period = period_res / predicted_period if predicted_period > 0 else 0.0
    consistent = rel_period <= tolerance
    flag = "CONSISTENT" if consistent else "INCONSISTENT"
    return KeplerCheckResult(
        period_days=period_days,
        semi_major_axis_au=semi_major_axis_au,
        stellar_mass_msun=stellar_mass_msun,
        predicted_period_days=predicted_period,
        predicted_sma_au=predicted_sma,
        period_residual_days=period_res,
        sma_residual_au=sma_res,
        consistent=consistent,
        flag=flag,
    )


def format_kepler_check_result(result: KeplerCheckResult) -> str:
    lines = [
        "## Kepler Third Law Check",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Period (days) | {result.period_days:.4f} |",
        f"| Predicted Period (days) | {result.predicted_period_days:.4f} |",
        f"| Period Residual (days) | {result.period_residual_days:.4f} |",
        f"| SMA (AU) | {result.semi_major_axis_au:.4f} |",
        f"| Predicted SMA (AU) | {result.predicted_sma_au:.4f} |",
        f"| SMA Residual (AU) | {result.sma_residual_au:.4f} |",
        f"| Consistent | {result.consistent} |",
        f"| Flag | {result.flag} |",
    ]
    return "\n".join(lines)


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Check Kepler 3rd law consistency.")
    parser.add_argument("period_days", type=float)
    parser.add_argument("semi_major_axis_au", type=float)
    parser.add_argument("stellar_mass_msun", type=float)
    parser.add_argument("--tolerance", type=float, default=0.05)
    args = parser.parse_args()
    result = check_kepler_third_law(
        args.period_days, args.semi_major_axis_au, args.stellar_mass_msun, args.tolerance,
    )
    print(format_kepler_check_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
