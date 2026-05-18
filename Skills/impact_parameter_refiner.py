"""Refine impact parameter from transit duration, period, and stellar density.

Uses the Seager & Mallén-Ornelas (2003) transit geometry formulas to
infer b (impact parameter) from observables.

Public API
----------
ImpactParameterResult(b, b_err, a_over_rstar, inclination_deg,
                      transit_duration_hours, period_days, flag)
refine_impact_parameter(period_days, duration_hours, depth_ppm,
                        stellar_radius_rsun, stellar_mass_msun, *,
                        duration_err_hours) -> ImpactParameterResult
format_impact_parameter_result(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass

_G = 6.674e-11
_MSUN_KG = 1.989e30
_RSUN_M = 6.957e8
_DAY_S = 86400.0


@dataclass(frozen=True)
class ImpactParameterResult:
    b: float                      # impact parameter [0, 1+k] where k = Rp/R*
    b_err: float | None
    a_over_rstar: float           # semi-major axis / stellar radius
    inclination_deg: float
    transit_duration_hours: float
    period_days: float
    flag: str                     # "CENTRAL", "GRAZING", "INVALID"


def _compute_b(a_over_rstar: float, inc_deg: float) -> float:
    return a_over_rstar * math.cos(math.radians(inc_deg))


def refine_impact_parameter(
    period_days: float,
    duration_hours: float,
    depth_ppm: float,
    stellar_radius_rsun: float,
    stellar_mass_msun: float,
    *,
    duration_err_hours: float | None = None,
) -> ImpactParameterResult:
    """Estimate impact parameter from transit geometry.

    Uses circular-orbit Kepler's 3rd law to get a/R*, then solves for b
    from the transit duration.

    Args:
        period_days: Orbital period in days.
        duration_hours: Transit duration (first to last contact) in hours.
        depth_ppm: Transit depth in parts per million.
        stellar_radius_rsun: Stellar radius in solar radii.
        stellar_mass_msun: Stellar mass in solar masses.
        duration_err_hours: 1-sigma duration uncertainty.

    Returns:
        :class:`ImpactParameterResult`.
    """
    if period_days <= 0 or duration_hours <= 0 or stellar_radius_rsun <= 0:
        return ImpactParameterResult(0.0, None, 0.0, 90.0, duration_hours, period_days, "INVALID")

    P = period_days * _DAY_S
    r_star = stellar_radius_rsun * _RSUN_M
    m_star = stellar_mass_msun * _MSUN_KG

    # Semi-major axis from Kepler's 3rd law: a³ = G M* P² / (4π²)
    a = ((_G * m_star * P ** 2) / (4 * math.pi ** 2)) ** (1.0 / 3.0)
    a_over_rstar = a / r_star

    # Radius ratio
    k = math.sqrt(max(depth_ppm, 0.0) / 1e6)

    # Transit duration formula (Seager & Mallén-Ornelas 2003, eq. 9):
    # T = (P/π) * arcsin( (R*/a) * sqrt((1+k)² - b²) / sin(i) )
    # For grazing/central approximation: solve for b using T and a/R*
    #
    # sin(π T / P) = (1/a_r) * sqrt((1+k)^2 - b^2) / sin(i)
    # Combined with b = a_r cos(i):
    # sin^2(πT/P) * a_r^2 = (1+k)^2 - b^2 + b^2/a_r^2 * (a_r^2 - 1)
    # Simplified (valid for b=a_r cos i):
    # sin^2(πT/P) = ((1+k)^2 - b^2) / a_r^2

    dur_days = duration_hours / 24.0
    sin_val = math.sin(math.pi * dur_days / period_days)
    rhs = (1.0 + k) ** 2 - (a_over_rstar * sin_val) ** 2
    if rhs < 0:
        # Duration is too long given a/R*; clamp to grazing
        b = 1.0 + k
        flag = "GRAZING"
    else:
        b = math.sqrt(max(rhs, 0.0))
        flag = "GRAZING" if b >= 0.9 else "CENTRAL"

    # Inclination from b = a/R* * cos(i)
    cos_i = b / a_over_rstar if a_over_rstar > 0 else 0.0
    cos_i = max(-1.0, min(1.0, cos_i))
    inc_deg = math.degrees(math.acos(cos_i))

    b_err: float | None = None
    if duration_err_hours is not None and duration_err_hours > 0:
        dur_hi = duration_hours + duration_err_hours
        dur_lo = max(duration_hours - duration_err_hours, 0.01)

        sin_hi = math.sin(math.pi * dur_hi / 24.0 / period_days)
        rhs_hi = (1.0 + k) ** 2 - (a_over_rstar * sin_hi) ** 2
        b_hi = math.sqrt(max(rhs_hi, 0.0)) if rhs_hi >= 0 else 1.0 + k

        sin_lo = math.sin(math.pi * dur_lo / 24.0 / period_days)
        rhs_lo = (1.0 + k) ** 2 - (a_over_rstar * sin_lo) ** 2
        b_lo = math.sqrt(max(rhs_lo, 0.0)) if rhs_lo >= 0 else 1.0 + k

        b_err = abs(b_hi - b_lo) / 2.0

    return ImpactParameterResult(
        b=round(b, 4),
        b_err=round(b_err, 4) if b_err is not None else None,
        a_over_rstar=round(a_over_rstar, 3),
        inclination_deg=round(inc_deg, 3),
        transit_duration_hours=duration_hours,
        period_days=period_days,
        flag=flag,
    )


def format_impact_parameter_result(result: ImpactParameterResult) -> str:
    """Format impact parameter result as Markdown."""
    err_str = f" ± {result.b_err:.4f}" if result.b_err is not None else ""
    lines = [
        "## Impact Parameter Refinement",
        "",
        f"- Period: {result.period_days:.4f} d",
        f"- Duration: {result.transit_duration_hours:.3f} h",
        f"- a/R★: {result.a_over_rstar:.2f}",
        f"- **b = {result.b:.4f}{err_str}**",
        f"- Inclination: {result.inclination_deg:.2f}°",
        f"- Flag: **{result.flag}**",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="impact_parameter_refiner",
        description="Refine impact parameter from transit geometry.",
    )
    parser.add_argument("period_days", type=float)
    parser.add_argument("duration_hours", type=float)
    parser.add_argument("depth_ppm", type=float)
    parser.add_argument("stellar_radius_rsun", type=float)
    parser.add_argument("stellar_mass_msun", type=float)
    args = parser.parse_args(argv)

    result = refine_impact_parameter(
        args.period_days, args.duration_hours, args.depth_ppm,
        args.stellar_radius_rsun, args.stellar_mass_msun,
    )
    print(format_impact_parameter_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
