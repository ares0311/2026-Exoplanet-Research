"""Compute expected transit duration from orbital and stellar parameters.

Uses the Seager & Mallén-Ornelas (2003) approximation for the full and
flat-bottom transit durations of a small planet on a circular orbit.

Public API
----------
TransitDurationResult(period_days, stellar_radius_rsun, stellar_mass_msun,
                      impact_parameter, depth_ppm, t14_hours, t23_hours,
                      ingress_egress_hours, flag)
compute_transit_duration(period_days, stellar_radius_rsun, stellar_mass_msun,
                         *, impact_parameter, depth_ppm) -> TransitDurationResult
format_duration_result(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass

_G = 6.674e-11
_MSUN_KG = 1.989e30
_RSUN_M = 6.957e8


@dataclass(frozen=True)
class TransitDurationResult:
    period_days: float
    stellar_radius_rsun: float
    stellar_mass_msun: float
    impact_parameter: float
    depth_ppm: float
    t14_hours: float | None
    t23_hours: float | None
    ingress_egress_hours: float | None
    flag: str  # "OK", "GRAZING", "INVALID"


def compute_transit_duration(
    period_days: float,
    stellar_radius_rsun: float,
    stellar_mass_msun: float,
    *,
    impact_parameter: float = 0.0,
    depth_ppm: float = 0.0,
) -> TransitDurationResult:
    """Compute T14 and T23 transit durations.

    Args:
        period_days: Orbital period in days.
        stellar_radius_rsun: Stellar radius in solar radii.
        stellar_mass_msun: Stellar mass in solar masses.
        impact_parameter: Normalised impact parameter b in [0, 1).
        depth_ppm: Transit depth in ppm (used to derive Rp/R*).

    Returns:
        :class:`TransitDurationResult`.
    """
    if period_days <= 0 or stellar_radius_rsun <= 0 or stellar_mass_msun <= 0:
        return TransitDurationResult(
            period_days, stellar_radius_rsun, stellar_mass_msun,
            impact_parameter, depth_ppm, None, None, None, "INVALID",
        )

    b = abs(impact_parameter)
    p = period_days * 86400.0  # seconds
    rs = stellar_radius_rsun * _RSUN_M
    ms = stellar_mass_msun * _MSUN_KG

    # Semi-major axis via Kepler's third law
    a = ((_G * ms * p ** 2) / (4 * math.pi ** 2)) ** (1.0 / 3.0)
    a_over_rs = a / rs

    rp_over_rs = math.sqrt(max(depth_ppm / 1e6, 0.0))

    # Seager & MO (2003) eq. 3: duration factor
    sin_arg_14_sq = ((1.0 + rp_over_rs) ** 2 - b ** 2) / a_over_rs ** 2
    sin_arg_23_sq = ((1.0 - rp_over_rs) ** 2 - b ** 2) / a_over_rs ** 2

    if sin_arg_14_sq < 0:
        return TransitDurationResult(
            period_days, stellar_radius_rsun, stellar_mass_msun,
            impact_parameter, depth_ppm, None, None, None, "GRAZING",
        )

    t14_sec = (p / math.pi) * math.asin(math.sqrt(sin_arg_14_sq) / math.sin(math.pi / 2))
    t14_hours = t14_sec / 3600.0

    t23_hours: float | None = None
    ingress_egress_hours: float | None = None
    if sin_arg_23_sq >= 0:
        t23_sec = (p / math.pi) * math.asin(math.sqrt(sin_arg_23_sq) / math.sin(math.pi / 2))
        t23_hours = t23_sec / 3600.0
        ingress_egress_hours = (t14_hours - t23_hours) / 2.0

    flag = "GRAZING" if t23_hours is None else "OK"
    return TransitDurationResult(
        period_days=period_days,
        stellar_radius_rsun=stellar_radius_rsun,
        stellar_mass_msun=stellar_mass_msun,
        impact_parameter=b,
        depth_ppm=depth_ppm,
        t14_hours=round(t14_hours, 4),
        t23_hours=round(t23_hours, 4) if t23_hours is not None else None,
        ingress_egress_hours=(
            round(ingress_egress_hours, 4) if ingress_egress_hours is not None else None
        ),
        flag=flag,
    )


def format_duration_result(result: TransitDurationResult) -> str:
    """Format transit duration result as Markdown."""
    lines = [
        "## Transit Duration",
        "",
        f"- Period: {result.period_days:.4f} days",
        f"- R★: {result.stellar_radius_rsun:.3f} R☉",
        f"- M★: {result.stellar_mass_msun:.3f} M☉",
        f"- Impact parameter b: {result.impact_parameter:.3f}",
        f"- Depth: {result.depth_ppm:.1f} ppm",
    ]
    if result.flag == "INVALID":
        lines.append("- **Flag: INVALID** — non-positive inputs")
    elif result.flag == "GRAZING":
        lines.append("- **Flag: GRAZING** — transit does not fit stellar disk")
    else:
        lines += [
            f"- T₁₄ (full duration): {result.t14_hours:.4f} h",
            (f"- T₂₃ (flat-bottom): {result.t23_hours:.4f} h"
             if result.t23_hours is not None else "- T₂₃: N/A"),
            (f"- Ingress/egress: {result.ingress_egress_hours:.4f} h"
             if result.ingress_egress_hours is not None else "- Ingress/egress: N/A"),
            f"- **Flag: {result.flag}**",
        ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="transit_duration_calculator",
        description="Compute expected transit duration.",
    )
    parser.add_argument("period_days", type=float)
    parser.add_argument("stellar_radius_rsun", type=float)
    parser.add_argument("stellar_mass_msun", type=float)
    parser.add_argument("--impact-parameter", type=float, default=0.0)
    parser.add_argument("--depth-ppm", type=float, default=0.0)
    args = parser.parse_args(argv)

    result = compute_transit_duration(
        args.period_days, args.stellar_radius_rsun, args.stellar_mass_msun,
        impact_parameter=args.impact_parameter, depth_ppm=args.depth_ppm,
    )
    print(format_duration_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
