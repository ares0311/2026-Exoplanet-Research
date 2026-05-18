"""Calculate transit geometry parameters from measured observables.

Derives semi-major axis, orbital inclination, and planet/star radius ratio
from transit depth, duration, period, and impact parameter.

Public API
----------
TransitGeometryResult(period_days, depth_ppm, duration_hours, impact_parameter,
                      stellar_radius_rsun, stellar_mass_msun,
                      rp_over_rs, a_over_rs, a_au, inclination_deg, flag)
compute_transit_geometry(period_days, depth_ppm, duration_hours,
                         stellar_radius_rsun, stellar_mass_msun, *,
                         impact_parameter) -> TransitGeometryResult
format_geometry_result(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass

_G = 6.674e-11
_MSUN_KG = 1.989e30
_RSUN_M = 6.957e8
_AU_M = 1.496e11


@dataclass(frozen=True)
class TransitGeometryResult:
    period_days: float
    depth_ppm: float
    duration_hours: float
    impact_parameter: float
    stellar_radius_rsun: float
    stellar_mass_msun: float
    rp_over_rs: float | None
    a_over_rs: float | None
    a_au: float | None
    inclination_deg: float | None
    flag: str  # "OK", "INVALID", "UNPHYSICAL"


def compute_transit_geometry(
    period_days: float,
    depth_ppm: float,
    duration_hours: float,
    stellar_radius_rsun: float,
    stellar_mass_msun: float,
    *,
    impact_parameter: float = 0.0,
) -> TransitGeometryResult:
    """Compute transit geometry from observables.

    Args:
        period_days: Orbital period in days.
        depth_ppm: Transit depth in ppm.
        duration_hours: Full transit duration T₁₄ in hours.
        stellar_radius_rsun: Stellar radius in solar radii.
        stellar_mass_msun: Stellar mass in solar masses.
        impact_parameter: Normalised impact parameter b.

    Returns:
        :class:`TransitGeometryResult`.
    """
    if (period_days <= 0 or depth_ppm < 0 or duration_hours <= 0
            or stellar_radius_rsun <= 0 or stellar_mass_msun <= 0):
        return TransitGeometryResult(
            period_days, depth_ppm, duration_hours, abs(impact_parameter),
            stellar_radius_rsun, stellar_mass_msun,
            None, None, None, None, "INVALID",
        )

    b = abs(impact_parameter)
    p_sec = period_days * 86400.0
    t14_sec = duration_hours * 3600.0
    rs_m = stellar_radius_rsun * _RSUN_M
    ms_kg = stellar_mass_msun * _MSUN_KG

    # Rp/R* from depth
    rp_over_rs = math.sqrt(max(depth_ppm / 1e6, 0.0))

    # Semi-major axis from Kepler's third law
    a_m = ((_G * ms_kg * p_sec ** 2) / (4 * math.pi ** 2)) ** (1.0 / 3.0)
    a_over_rs = a_m / rs_m
    a_au = a_m / _AU_M

    # Check physical consistency
    sin_arg = math.pi * t14_sec / p_sec
    if sin_arg >= math.pi / 2:
        return TransitGeometryResult(
            period_days, depth_ppm, duration_hours, b,
            stellar_radius_rsun, stellar_mass_msun,
            round(rp_over_rs, 6), round(a_over_rs, 4), round(a_au, 6),
            None, "UNPHYSICAL",
        )

    # Inclination: cos(i) = b / (a/R*)
    if a_over_rs > 0:
        cos_i = b / a_over_rs
        cos_i = min(cos_i, 1.0)
        inclination_deg = math.degrees(math.acos(cos_i))
    else:
        inclination_deg = None

    return TransitGeometryResult(
        period_days=period_days,
        depth_ppm=depth_ppm,
        duration_hours=duration_hours,
        impact_parameter=b,
        stellar_radius_rsun=stellar_radius_rsun,
        stellar_mass_msun=stellar_mass_msun,
        rp_over_rs=round(rp_over_rs, 6),
        a_over_rs=round(a_over_rs, 4),
        a_au=round(a_au, 6),
        inclination_deg=round(inclination_deg, 4) if inclination_deg is not None else None,
        flag="OK",
    )


def format_geometry_result(result: TransitGeometryResult) -> str:
    """Format transit geometry result as Markdown."""
    lines = [
        "## Transit Geometry",
        "",
        f"- Period: {result.period_days:.4f} days",
        f"- Depth: {result.depth_ppm:.1f} ppm",
        f"- Duration (T₁₄): {result.duration_hours:.4f} h",
        f"- Impact parameter b: {result.impact_parameter:.3f}",
        f"- R★: {result.stellar_radius_rsun:.3f} R☉",
        f"- M★: {result.stellar_mass_msun:.3f} M☉",
    ]
    if result.flag == "INVALID":
        lines.append("- **Flag: INVALID** — non-positive inputs")
    elif result.flag == "UNPHYSICAL":
        lines.append("- **Flag: UNPHYSICAL** — geometry inconsistency")
    else:
        lines += [
            f"- Rp/R★: {result.rp_over_rs:.6f}",
            f"- a/R★: {result.a_over_rs:.4f}",
            f"- a: {result.a_au:.4f} AU",
            (f"- Inclination: {result.inclination_deg:.4f}°"
             if result.inclination_deg is not None else "- Inclination: N/A"),
            f"- **Flag: {result.flag}**",
        ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="transit_geometry_calculator",
        description="Compute transit geometry from observables.",
    )
    parser.add_argument("period_days", type=float)
    parser.add_argument("depth_ppm", type=float)
    parser.add_argument("duration_hours", type=float)
    parser.add_argument("stellar_radius_rsun", type=float)
    parser.add_argument("stellar_mass_msun", type=float)
    parser.add_argument("--impact-parameter", type=float, default=0.0)
    args = parser.parse_args(argv)

    result = compute_transit_geometry(
        args.period_days, args.depth_ppm, args.duration_hours,
        args.stellar_radius_rsun, args.stellar_mass_msun,
        impact_parameter=args.impact_parameter,
    )
    print(format_geometry_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
