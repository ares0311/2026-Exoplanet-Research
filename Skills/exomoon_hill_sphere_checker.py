"""Check exomoon stability within the planet's Hill sphere."""
from __future__ import annotations

import math
from dataclasses import dataclass

_G = 6.674e-11
_MSUN_KG = 1.989e30
_MJUP_KG = 1.898e27
_AU_M = 1.495978707e11
_RJUP_M = 7.1492e7


@dataclass(frozen=True)
class ExomoonStabilityResult:
    hill_radius_au: float               # planet Hill sphere radius (AU)
    stable_orbit_limit_au: float        # ~0.5 * R_Hill (prograde) or ~0.33 * R_Hill
    stable_orbit_limit_rjup: float      # same in planet radii
    roche_limit_rjup: float             # Roche limit for moon
    max_moon_period_days: float         # period at stable orbit limit
    stability_class: str                # WIDE_STABLE / MARGINAL / ROCHE_LIMITED
    flag: str


def check_exomoon_stability(
    stellar_mass_msun: float,
    planet_mass_mjup: float,
    orbital_distance_au: float,
    moon_density_gcc: float = 3.0,
    moon_radius_rearth: float = 0.27,
    prograde: bool = True,
) -> ExomoonStabilityResult:
    """Check exomoon stability within the planetary Hill sphere.

    Stable limit (Domingos et al. 2006):
      r_stable ≈ 0.4895 * R_Hill   (prograde)
      r_stable ≈ 0.9309 * R_Hill   (retrograde)
    Hill radius: R_H = a * (Mp / (3*Ms))^(1/3)

    Roche limit (fluid Moon):
      d_Roche = R_planet * (2 * ρ_planet / ρ_moon)^(1/3)

    Args:
        stellar_mass_msun: stellar mass (solar masses)
        planet_mass_mjup: planet mass (Jupiter masses)
        orbital_distance_au: planet semi-major axis (AU)
        moon_density_gcc: moon bulk density (g/cm³)
        moon_radius_rearth: moon radius (Earth radii); for Roche limit calc
        prograde: True for prograde orbit (tighter limit)
    """
    if stellar_mass_msun <= 0.0:
        return ExomoonStabilityResult(float("nan"), float("nan"), float("nan"),
                                       float("nan"), float("nan"), "UNKNOWN",
                                       "INVALID_STELLAR_MASS")
    if planet_mass_mjup <= 0.0:
        return ExomoonStabilityResult(float("nan"), float("nan"), float("nan"),
                                       float("nan"), float("nan"), "UNKNOWN",
                                       "INVALID_PLANET_MASS")
    if orbital_distance_au <= 0.0:
        return ExomoonStabilityResult(float("nan"), float("nan"), float("nan"),
                                       float("nan"), float("nan"), "UNKNOWN",
                                       "INVALID_DISTANCE")

    _MEARTH_KG = 5.972e24
    _REARTH_M = 6.371e6

    ms_kg = stellar_mass_msun * _MSUN_KG
    mp_kg = planet_mass_mjup * _MJUP_KG
    a_m = orbital_distance_au * _AU_M
    rp_m = _RJUP_M  # use nominal Jupiter radius for Roche limit

    # Hill radius
    r_hill_m = a_m * (mp_kg / (3.0 * ms_kg)) ** (1.0 / 3.0)
    r_hill_au = r_hill_m / _AU_M

    # Stable orbit fraction (Domingos+2006)
    frac = 0.4895 if prograde else 0.9309
    r_stable_m = frac * r_hill_m
    r_stable_au = r_stable_m / _AU_M
    r_stable_rjup = r_stable_m / _RJUP_M

    # Roche limit (fluid satellite)
    rho_planet_cgs = mp_kg / (4.0 / 3.0 * math.pi * rp_m**3) * 1e-3  # g/cm³
    moon_density_gcc = max(moon_density_gcc, 0.1)
    d_roche_m = rp_m * (2.0 * rho_planet_cgs / moon_density_gcc) ** (1.0 / 3.0)
    d_roche_rjup = d_roche_m / _RJUP_M

    # Period at stable limit (Kepler's 3rd law around planet)
    p_stable_s = 2.0 * math.pi * math.sqrt(r_stable_m**3 / (_G * mp_kg))
    p_stable_days = p_stable_s / 86400.0

    # Stability class
    if r_stable_m > 5.0 * d_roche_m:
        stab = "WIDE_STABLE"
    elif r_stable_m > d_roche_m:
        stab = "MARGINAL"
    else:
        stab = "ROCHE_LIMITED"

    return ExomoonStabilityResult(
        hill_radius_au=r_hill_au,
        stable_orbit_limit_au=r_stable_au,
        stable_orbit_limit_rjup=r_stable_rjup,
        roche_limit_rjup=d_roche_rjup,
        max_moon_period_days=p_stable_days,
        stability_class=stab,
        flag="OK",
    )


def format_exomoon_stability_result(r: ExomoonStabilityResult) -> str:
    if r.flag != "OK":
        return f"ExomoonStability | flag={r.flag}"
    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| Hill radius | {r.hill_radius_au:.4f} AU |\n"
        f"| Stable orbit limit | {r.stable_orbit_limit_rjup:.2f} R_Jup |\n"
        f"| Roche limit | {r.roche_limit_rjup:.2f} R_Jup |\n"
        f"| Max moon period | {r.max_moon_period_days:.2f} d |\n"
        f"| Stability class | {r.stability_class} |\n"
        f"| Flag | {r.flag} |"
    )


def _cli() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Exomoon Hill sphere stability checker")
    p.add_argument("stellar_mass_msun", type=float)
    p.add_argument("planet_mass_mjup", type=float)
    p.add_argument("orbital_distance_au", type=float)
    p.add_argument("--retrograde", action="store_true")
    args = p.parse_args()
    r = check_exomoon_stability(args.stellar_mass_msun, args.planet_mass_mjup,
                                 args.orbital_distance_au, prograde=not args.retrograde)
    print(format_exomoon_stability_result(r))


if __name__ == "__main__":
    _cli()
