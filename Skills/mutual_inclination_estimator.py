"""Estimate mutual inclination between two transiting planets."""
from __future__ import annotations

import math
from dataclasses import dataclass

_G = 6.674e-11
_MSUN_KG = 1.989e30
_RSUN_M = 6.957e8


@dataclass(frozen=True)
class MutualInclinationResult:
    inclination1_deg: float
    inclination2_deg: float
    mutual_inclination_deg: float
    impact_parameter1: float
    impact_parameter2: float
    flag: str


def estimate_mutual_inclination(
    period1_days: float,
    depth1_ppm: float,
    duration1_hours: float,
    period2_days: float,
    depth2_ppm: float,
    duration2_hours: float,
    stellar_radius_rsun: float = 1.0,
    stellar_mass_msun: float = 1.0,
) -> MutualInclinationResult:
    """Estimate mutual inclination from two transiting planets' durations.

    For each planet, derive inclination i_j from:
      cos(i_j) = b_j * Rs / a_j
    where b_j = sqrt(max(((a_j/Rs)² * (1 - (π*T_j*k_j/(P_j))²), 0))

    Mutual inclination: Δi = |i₁ - i₂|  (coplanar lower bound)

    Args:
        period1_days: period of planet 1 (days)
        depth1_ppm: transit depth of planet 1 (ppm)
        duration1_hours: transit duration of planet 1 (hours)
        period2_days: period of planet 2 (days)
        depth2_ppm: transit depth of planet 2 (ppm)
        duration2_hours: transit duration of planet 2 (hours)
        stellar_radius_rsun: stellar radius (solar radii)
        stellar_mass_msun: stellar mass (solar masses)
    """
    for val, name in [(period1_days, "PERIOD"), (period2_days, "PERIOD"),
                      (depth1_ppm, "DEPTH"), (depth2_ppm, "DEPTH"),
                      (duration1_hours, "DURATION"), (duration2_hours, "DURATION")]:
        if val <= 0.0:
            return MutualInclinationResult(float("nan"), float("nan"),
                                            float("nan"), float("nan"),
                                            float("nan"), f"INVALID_{name}")

    rs_m = stellar_radius_rsun * _RSUN_M
    ms_kg = stellar_mass_msun * _MSUN_KG

    def _incl(period_days: float, duration_hours: float) -> tuple[float, float]:
        p_s = period_days * 86400.0
        a_m = (_G * ms_kg * p_s**2 / (4.0 * math.pi**2)) ** (1.0 / 3.0)
        a_rs = a_m / rs_m
        # sin(θ) where θ = π * T / P (half-angle of transit)
        sin_theta = math.pi * duration_hours / 24.0 / period_days
        sin_theta = min(sin_theta, 1.0)
        # b = sqrt((a/Rs)^2 - (a/Rs * sin(theta))^2)
        b_val = math.sqrt(max(a_rs**2 - (a_rs * sin_theta)**2, 0.0))
        cos_i = b_val / a_rs if a_rs > 0 else 0.0
        cos_i = max(min(cos_i, 1.0), -1.0)
        incl_deg = math.degrees(math.acos(cos_i))
        return incl_deg, b_val

    i1, b1 = _incl(period1_days, duration1_hours)
    i2, b2 = _incl(period2_days, duration2_hours)

    delta_i = abs(i1 - i2)

    return MutualInclinationResult(
        inclination1_deg=i1,
        inclination2_deg=i2,
        mutual_inclination_deg=delta_i,
        impact_parameter1=b1,
        impact_parameter2=b2,
        flag="OK",
    )


def format_mutual_inclination_result(r: MutualInclinationResult) -> str:
    if r.flag != "OK":
        return f"MutualInclination | flag={r.flag}"
    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| Inclination planet 1 | {r.inclination1_deg:.3f} ° |\n"
        f"| Inclination planet 2 | {r.inclination2_deg:.3f} ° |\n"
        f"| Mutual inclination | {r.mutual_inclination_deg:.3f} ° |\n"
        f"| b₁ | {r.impact_parameter1:.3f} |\n"
        f"| b₂ | {r.impact_parameter2:.3f} |\n"
        f"| Flag | {r.flag} |"
    )


def _cli() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Mutual inclination estimator")
    p.add_argument("period1", type=float)
    p.add_argument("depth1_ppm", type=float)
    p.add_argument("duration1_hours", type=float)
    p.add_argument("period2", type=float)
    p.add_argument("depth2_ppm", type=float)
    p.add_argument("duration2_hours", type=float)
    args = p.parse_args()
    r = estimate_mutual_inclination(args.period1, args.depth1_ppm, args.duration1_hours,
                                     args.period2, args.depth2_ppm, args.duration2_hours)
    print(format_mutual_inclination_result(r))


if __name__ == "__main__":
    _cli()
