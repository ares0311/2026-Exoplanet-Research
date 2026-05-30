"""Derive transit impact parameter b from T14, T23, and stellar radius.

Uses Seager & Mallén-Ornelas (2003) geometry.

Public API:
    ImpactParameterResult  -- frozen dataclass
    compute_impact_parameter(t14_hours, t23_hours, stellar_radius_rsun,
                              period_days, depth_ppm) -> ImpactParameterResult
    format_impact_parameter_result(result) -> str
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

_RSUN_AU = 0.00465047


@dataclass(frozen=True)
class ImpactParameterResult:
    t14_hours: float
    t23_hours: float
    impact_parameter: float
    rp_over_rstar: float
    a_over_rstar: float
    inclination_deg: float
    flag: str


def compute_impact_parameter(
    t14_hours: float,
    t23_hours: float,
    stellar_radius_rsun: float,
    period_days: float,
    depth_ppm: float,
) -> ImpactParameterResult:
    if t14_hours <= 0:
        return ImpactParameterResult(
            t14_hours=t14_hours, t23_hours=t23_hours, impact_parameter=0.0,
            rp_over_rstar=0.0, a_over_rstar=0.0, inclination_deg=0.0, flag="INVALID_T14",
        )
    if t23_hours < 0 or t23_hours >= t14_hours:
        return ImpactParameterResult(
            t14_hours=t14_hours, t23_hours=t23_hours, impact_parameter=0.0,
            rp_over_rstar=0.0, a_over_rstar=0.0, inclination_deg=0.0, flag="INVALID_T23",
        )
    if stellar_radius_rsun <= 0:
        return ImpactParameterResult(
            t14_hours=t14_hours, t23_hours=t23_hours, impact_parameter=0.0,
            rp_over_rstar=0.0, a_over_rstar=0.0, inclination_deg=0.0, flag="INVALID_STELLAR_RADIUS",
        )
    if period_days <= 0:
        return ImpactParameterResult(
            t14_hours=t14_hours, t23_hours=t23_hours, impact_parameter=0.0,
            rp_over_rstar=0.0, a_over_rstar=0.0, inclination_deg=0.0, flag="INVALID_PERIOD",
        )
    if depth_ppm <= 0:
        return ImpactParameterResult(
            t14_hours=t14_hours, t23_hours=t23_hours, impact_parameter=0.0,
            rp_over_rstar=0.0, a_over_rstar=0.0, inclination_deg=0.0, flag="INVALID_DEPTH",
        )
    rp_over_rstar = math.sqrt(depth_ppm / 1e6)
    # sin(pi*T14/P) and sin(pi*T23/P) in fraction of period
    t14_frac = t14_hours / 24.0
    t23_frac = t23_hours / 24.0
    sin14 = math.sin(math.pi * t14_frac / period_days)
    sin23 = math.sin(math.pi * t23_frac / period_days)
    # a/R* from T14: a/R* = sqrt((1+k)^2 - b^2) / sin14 * pi/P  -- iterative approach
    # Direct formula: (1+k)^2 = (a/R*)^2 sin14^2 + b^2
    #                 (1-k)^2 = (a/R*)^2 sin23^2 + b^2
    # Subtracting: 4k = (a/R*)^2 (sin14^2 - sin23^2)
    k = rp_over_rstar
    denom = sin14 ** 2 - sin23 ** 2
    if denom <= 0:
        # fallback: assume b=0
        a_over_rstar = (1.0 + k) / sin14 if sin14 > 0 else 0.0
        impact_parameter = 0.0
    else:
        a_over_rstar_sq = 4.0 * k / denom
        a_over_rstar = math.sqrt(a_over_rstar_sq) if a_over_rstar_sq > 0 else 0.0
        b_sq = (1.0 + k) ** 2 - a_over_rstar ** 2 * sin14 ** 2
        impact_parameter = math.sqrt(max(0.0, b_sq))
    if a_over_rstar > 0:
        cos_i = impact_parameter / a_over_rstar
        inclination_deg = math.degrees(math.acos(min(1.0, cos_i)))
    else:
        inclination_deg = 90.0
    return ImpactParameterResult(
        t14_hours=t14_hours,
        t23_hours=t23_hours,
        impact_parameter=impact_parameter,
        rp_over_rstar=rp_over_rstar,
        a_over_rstar=a_over_rstar,
        inclination_deg=inclination_deg,
        flag="OK",
    )


def format_impact_parameter_result(result: ImpactParameterResult) -> str:
    lines = [
        "## Impact Parameter from Duration",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| T14 (hr) | {result.t14_hours:.3f} |",
        f"| T23 (hr) | {result.t23_hours:.3f} |",
        f"| Impact Parameter b | {result.impact_parameter:.4f} |",
        f"| Rp/R* | {result.rp_over_rstar:.4f} |",
        f"| a/R* | {result.a_over_rstar:.2f} |",
        f"| Inclination (deg) | {result.inclination_deg:.2f} |",
        f"| Flag | {result.flag} |",
    ]
    return "\n".join(lines)


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Compute impact parameter from transit durations.")
    parser.add_argument("t14_hours", type=float)
    parser.add_argument("t23_hours", type=float)
    parser.add_argument("stellar_radius_rsun", type=float)
    parser.add_argument("period_days", type=float)
    parser.add_argument("depth_ppm", type=float)
    args = parser.parse_args()
    result = compute_impact_parameter(
        args.t14_hours, args.t23_hours, args.stellar_radius_rsun,
        args.period_days, args.depth_ppm,
    )
    print(format_impact_parameter_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
