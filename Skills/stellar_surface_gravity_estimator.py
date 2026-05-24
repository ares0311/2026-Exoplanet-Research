"""Derive stellar surface gravity log g from stellar mass and radius.

Uses g = G M / R² to compute surface gravity in cgs units, then returns
log10(g).  Optionally propagates fractional uncertainties on mass and radius.
Complements ``stellar_density_calculator`` (which derives ρ★ from transit
duration) and ``stellar_params_fetcher`` (which reads params from catalogs).

Public API
----------
SurfaceGravityResult(mass_msun, radius_rsun, gravity_cgs, logg,
                     logg_uncertainty, flag)
estimate_surface_gravity(mass_msun, radius_rsun, *,
                         mass_err_msun, radius_err_rsun) -> SurfaceGravityResult
format_surface_gravity_result(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# Physical constants (CGS)
_G_CGS = 6.674e-8          # cm^3 g^-1 s^-2
_MSUN_CGS = 1.989e33       # g
_RSUN_CGS = 6.957e10       # cm


@dataclass(frozen=True)
class SurfaceGravityResult:
    mass_msun: float
    radius_rsun: float
    gravity_cgs: float | None    # cm s^-2
    logg: float | None           # log10(g / cm s^-2)
    logg_uncertainty: float | None
    flag: str  # "OK" | "INVALID"


def estimate_surface_gravity(
    mass_msun: float,
    radius_rsun: float,
    *,
    mass_err_msun: float | None = None,
    radius_err_rsun: float | None = None,
) -> SurfaceGravityResult:
    """Compute stellar surface gravity log g in cgs units.

    g = G × M / R²
    σ_logg = sqrt((σ_M/M)² + (2 σ_R/R)²) / ln(10)

    Args:
        mass_msun: Stellar mass (solar masses).
        radius_rsun: Stellar radius (solar radii).
        mass_err_msun: Uncertainty on mass (solar masses).
        radius_err_rsun: Uncertainty on radius (solar radii).

    Returns:
        :class:`SurfaceGravityResult`.
    """
    if not (math.isfinite(mass_msun) and math.isfinite(radius_rsun)):
        return SurfaceGravityResult(mass_msun, radius_rsun, None, None, None, "INVALID")
    if mass_msun <= 0 or radius_rsun <= 0:
        return SurfaceGravityResult(mass_msun, radius_rsun, None, None, None, "INVALID")

    m_cgs = mass_msun * _MSUN_CGS
    r_cgs = radius_rsun * _RSUN_CGS
    g_cgs = _G_CGS * m_cgs / r_cgs ** 2
    logg = math.log10(g_cgs)

    logg_err: float | None = None
    if (mass_err_msun is not None and radius_err_rsun is not None
            and mass_err_msun >= 0 and radius_err_rsun >= 0):
        frac_m = mass_err_msun / mass_msun
        frac_r = radius_err_rsun / radius_rsun
        logg_err = round(math.sqrt(frac_m ** 2 + (2 * frac_r) ** 2) / math.log(10), 4)

    return SurfaceGravityResult(
        mass_msun=mass_msun,
        radius_rsun=radius_rsun,
        gravity_cgs=round(g_cgs, 4),
        logg=round(logg, 4),
        logg_uncertainty=logg_err,
        flag="OK",
    )


def format_surface_gravity_result(result: SurfaceGravityResult) -> str:
    """Format surface gravity result as Markdown."""
    err_str = f" ± {result.logg_uncertainty}" if result.logg_uncertainty is not None else ""
    lines = [
        "## Stellar Surface Gravity Estimator",
        "",
        f"- Mass: {result.mass_msun} M_sun",
        f"- Radius: {result.radius_rsun} R_sun",
        f"- g: {result.gravity_cgs} cm s⁻²",
        f"- **log g: {result.logg}{err_str} (cgs)**",
        f"- **Flag: {result.flag}**",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="stellar_surface_gravity_estimator",
        description="Compute stellar log g from mass and radius.",
    )
    parser.add_argument("mass_msun", type=float)
    parser.add_argument("radius_rsun", type=float)
    parser.add_argument("--mass-err", type=float, default=None)
    parser.add_argument("--radius-err", type=float, default=None)
    args = parser.parse_args(argv)

    result = estimate_surface_gravity(
        args.mass_msun, args.radius_rsun,
        mass_err_msun=args.mass_err,
        radius_err_rsun=args.radius_err,
    )
    print(format_surface_gravity_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
