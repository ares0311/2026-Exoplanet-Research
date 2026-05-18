"""Compute mean stellar density from transit observables.

Uses the photometric stellar density formula from Seager & Mallén-Ornelas (2003):
ρ★ = (3π / G·P²) · (a/R★)³

where a/R★ is derived from the transit duration and period (b=0 approximation).

Public API
----------
StellarDensityResult(period_days, duration_hours, depth_ppm,
                     a_over_rstar, rho_gcm3, rho_sun, flag)
compute_stellar_density(period_days, duration_hours, depth_ppm, *,
                        impact_parameter) -> StellarDensityResult
format_stellar_density_result(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass

_G = 6.674e-11
_RHO_SUN_GCMS = 1.408  # g cm⁻³


@dataclass(frozen=True)
class StellarDensityResult:
    period_days: float
    duration_hours: float
    depth_ppm: float
    impact_parameter: float
    a_over_rstar: float | None
    rho_gcm3: float | None
    rho_sun: float | None         # ρ★ / ρ☉
    flag: str                     # "OK", "INVALID", "UNPHYSICAL"


def compute_stellar_density(
    period_days: float,
    duration_hours: float,
    depth_ppm: float,
    *,
    impact_parameter: float = 0.0,
) -> StellarDensityResult:
    """Compute photometric stellar density.

    Args:
        period_days: Orbital period in days.
        duration_hours: Full transit duration T₁₄ in hours.
        depth_ppm: Transit depth in ppm.
        impact_parameter: Normalised impact parameter b.

    Returns:
        :class:`StellarDensityResult`.
    """
    if period_days <= 0 or duration_hours <= 0 or depth_ppm < 0:
        return StellarDensityResult(
            period_days, duration_hours, depth_ppm,
            abs(impact_parameter), None, None, None, "INVALID",
        )

    b = abs(impact_parameter)
    p_sec = period_days * 86400.0
    t14_sec = duration_hours * 3600.0
    k = math.sqrt(max(depth_ppm / 1e6, 0.0))

    # a/R* from duration formula (circular orbit):
    # T14 = (P/π) * arcsin(sqrt((1+k)² - b²) / (a/R*))
    # => (a/R*) = sqrt((1+k)² - b²) / sin(π*T14/P)
    if t14_sec >= p_sec:
        return StellarDensityResult(
            period_days, duration_hours, depth_ppm,
            b, None, None, None, "UNPHYSICAL",
        )

    sin_val = math.sin(math.pi * t14_sec / p_sec)
    if sin_val <= 0 or sin_val > 1:
        return StellarDensityResult(
            period_days, duration_hours, depth_ppm,
            b, None, None, None, "UNPHYSICAL",
        )

    numerator_sq = (1.0 + k) ** 2 - b ** 2
    if numerator_sq < 0:
        return StellarDensityResult(
            period_days, duration_hours, depth_ppm,
            b, None, None, None, "UNPHYSICAL",
        )

    a_over_rs = math.sqrt(numerator_sq) / sin_val

    # Stellar density: ρ★ = 3π(a/R★)³ / (G P²)  [kg/m³]
    rho_kgm3 = 3.0 * math.pi * a_over_rs ** 3 / (_G * p_sec ** 2)
    rho_gcm3 = rho_kgm3 * 1e-3  # kg/m³ → g/cm³
    rho_sun = rho_gcm3 / _RHO_SUN_GCMS

    return StellarDensityResult(
        period_days=period_days,
        duration_hours=duration_hours,
        depth_ppm=depth_ppm,
        impact_parameter=b,
        a_over_rstar=round(a_over_rs, 4),
        rho_gcm3=round(rho_gcm3, 4),
        rho_sun=round(rho_sun, 4),
        flag="OK",
    )


def format_stellar_density_result(result: StellarDensityResult) -> str:
    """Format stellar density result as Markdown."""
    lines = [
        "## Photometric Stellar Density",
        "",
        f"- Period: {result.period_days:.4f} days",
        f"- Duration (T₁₄): {result.duration_hours:.4f} h",
        f"- Depth: {result.depth_ppm:.1f} ppm",
        f"- Impact parameter b: {result.impact_parameter:.3f}",
    ]
    if result.flag == "INVALID":
        lines.append("- **Flag: INVALID** — non-positive inputs")
    elif result.flag == "UNPHYSICAL":
        lines.append("- **Flag: UNPHYSICAL** — duration exceeds orbital geometry")
    else:
        lines += [
            f"- a/R★: {result.a_over_rstar:.4f}",
            f"- ρ★: {result.rho_gcm3:.4f} g cm⁻³",
            f"- ρ★/ρ☉: {result.rho_sun:.4f}",
            f"- **Flag: {result.flag}**",
        ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="stellar_density_calculator",
        description="Compute photometric stellar density from transit observables.",
    )
    parser.add_argument("period_days", type=float)
    parser.add_argument("duration_hours", type=float)
    parser.add_argument("depth_ppm", type=float)
    parser.add_argument("--impact-parameter", type=float, default=0.0)
    args = parser.parse_args(argv)

    result = compute_stellar_density(
        args.period_days, args.duration_hours, args.depth_ppm,
        impact_parameter=args.impact_parameter,
    )
    print(format_stellar_density_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
