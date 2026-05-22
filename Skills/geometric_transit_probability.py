"""Compute the geometric probability that a planet transits its host star.

Uses Kepler's third law and the stellar density (or stellar radius + period)
to derive the semi-major axis, then returns P_tr = (R★ + Rp) / a.

Public API
----------
TransitProbResult(period_days, stellar_density_gcc, stellar_radius_rsun,
                  rp_over_rs, semi_major_axis_au, semi_major_axis_rs,
                  transit_probability, flag)
compute_transit_probability(period_days, stellar_density_gcc,
                            stellar_radius_rsun, *, rp_over_rs) -> TransitProbResult
format_transit_prob_result(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass

_RSUN_AU = 0.00465047       # solar radius in AU
_RSUN_CGS = 6.957e10        # solar radius in cm
_G_CGS = 6.674e-8           # gravitational constant cm³/g/s²
_DAY_S = 86400.0


@dataclass(frozen=True)
class TransitProbResult:
    period_days: float
    stellar_density_gcc: float | None
    stellar_radius_rsun: float | None
    rp_over_rs: float        # planet-to-star radius ratio (0 = point planet)
    semi_major_axis_au: float | None
    semi_major_axis_rs: float | None   # a / R★
    transit_probability: float | None  # P_tr in [0, 1]
    flag: str  # "OK" | "INVALID"


def compute_transit_probability(
    period_days: float,
    stellar_density_gcc: float | None = None,
    stellar_radius_rsun: float | None = None,
    *,
    rp_over_rs: float = 0.0,
) -> TransitProbResult:
    """Compute geometric transit probability P_tr ≈ (R★ + Rp) / a.

    Requires either ``stellar_density_gcc`` (preferred — gives a/R★ directly)
    or ``stellar_radius_rsun`` alone (uses solar density as fallback).

    Args:
        period_days: Orbital period (days).
        stellar_density_gcc: Stellar mean density (g/cm³).
        stellar_radius_rsun: Stellar radius (solar radii).
        rp_over_rs: Rp/R★ radius ratio (default 0 = point planet).

    Returns:
        :class:`TransitProbResult`.
    """
    if period_days <= 0:
        return TransitProbResult(
            period_days, stellar_density_gcc, stellar_radius_rsun,
            rp_over_rs, None, None, None, "INVALID"
        )

    P_s = period_days * _DAY_S

    # Derive a/R★ from stellar density via Kepler's 3rd law:
    # a/R★ = (G ρ★ P² / (3π))^(1/3)  (in stellar radii)
    a_over_rs: float | None = None
    a_au: float | None = None

    if stellar_density_gcc is not None and stellar_density_gcc > 0:
        a_over_rs = ((_G_CGS * stellar_density_gcc * P_s ** 2) / (3.0 * math.pi)) ** (1.0 / 3.0)
        if stellar_radius_rsun is not None and stellar_radius_rsun > 0:
            a_au = a_over_rs * stellar_radius_rsun * _RSUN_AU
    elif stellar_radius_rsun is not None and stellar_radius_rsun > 0:
        # Fall back to solar density ≈ 1.41 g/cm³
        rho_sun = 1.409
        a_over_rs = ((_G_CGS * rho_sun * P_s ** 2) / (3.0 * math.pi)) ** (1.0 / 3.0)
        a_au = a_over_rs * stellar_radius_rsun * _RSUN_AU

    if a_over_rs is None or a_over_rs <= 0:
        return TransitProbResult(
            period_days, stellar_density_gcc, stellar_radius_rsun,
            rp_over_rs, a_au, a_over_rs, None, "INSUFFICIENT"
        )

    p_tr = min(1.0, (1.0 + rp_over_rs) / a_over_rs)

    return TransitProbResult(
        period_days=period_days,
        stellar_density_gcc=stellar_density_gcc,
        stellar_radius_rsun=stellar_radius_rsun,
        rp_over_rs=rp_over_rs,
        semi_major_axis_au=round(a_au, 6) if a_au is not None else None,
        semi_major_axis_rs=round(a_over_rs, 4),
        transit_probability=round(p_tr, 6),
        flag="OK",
    )


def format_transit_prob_result(result: TransitProbResult) -> str:
    """Format transit probability result as Markdown."""
    lines = [
        "## Geometric Transit Probability",
        "",
        f"- Period: {result.period_days} days",
        f"- Stellar density: {result.stellar_density_gcc} g/cm³",
        f"- Stellar radius: {result.stellar_radius_rsun} R☉",
        f"- Rp/R★: {result.rp_over_rs}",
        f"- a/R★: {result.semi_major_axis_rs}",
        f"- a (AU): {result.semi_major_axis_au}",
        f"- **P_transit: {result.transit_probability}**",
        f"- **Flag: {result.flag}**",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="geometric_transit_probability",
        description="Compute geometric transit probability from period and stellar parameters.",
    )
    parser.add_argument("period_days", type=float)
    parser.add_argument("--stellar-density", type=float, default=None)
    parser.add_argument("--stellar-radius", type=float, default=None)
    parser.add_argument("--rp-over-rs", type=float, default=0.0)
    args = parser.parse_args(argv)

    result = compute_transit_probability(
        args.period_days,
        args.stellar_density,
        args.stellar_radius,
        rp_over_rs=args.rp_over_rs,
    )
    print(format_transit_prob_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
