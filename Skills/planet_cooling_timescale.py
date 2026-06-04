"""Estimate Kelvin-Helmholtz contraction cooling timescale for gas giant planets."""
from __future__ import annotations

import math
from dataclasses import dataclass

_G = 6.674e-11
_MSUN_KG = 1.989e30
_MJUP_KG = 1.898e27
_RJUP_M = 7.1492e7
_L_SUN = 3.828e26
_SEC_PER_GYR = 3.156e16


@dataclass(frozen=True)
class PlanetCoolingResult:
    gravitational_energy_j: float
    intrinsic_luminosity_w: float
    kelvin_helmholtz_timescale_gyr: float
    cooling_class: str   # YOUNG_INFLATED / CONTRACTING / EVOLVED
    flag: str


def compute_planet_cooling_timescale(
    planet_mass_mjup: float,
    planet_radius_rjup: float,
    intrinsic_luminosity_lsun: float | None = None,
    planet_age_gyr: float | None = None,
) -> PlanetCoolingResult:
    """Estimate Kelvin-Helmholtz cooling timescale for a gas giant.

    KH timescale: τ_KH = G Mp² / (Rp × L_int)

    If L_int not supplied, it is estimated from Baraffe et al. (2003) cooling tracks:
      log(L_int/L_sun) ≈ -5.0 - 1.3*log10(age_gyr) for Jupiter-mass planets.

    Args:
        planet_mass_mjup: planet mass (Jupiter masses)
        planet_radius_rjup: planet radius (Jupiter radii)
        intrinsic_luminosity_lsun: known intrinsic luminosity (solar units); optional
        planet_age_gyr: planet age (Gyr); used to estimate L_int if not provided
    """
    if planet_mass_mjup <= 0.0:
        return PlanetCoolingResult(float("nan"), float("nan"), float("nan"),
                                    "UNKNOWN", "INVALID_MASS")
    if planet_radius_rjup <= 0.0:
        return PlanetCoolingResult(float("nan"), float("nan"), float("nan"),
                                    "UNKNOWN", "INVALID_RADIUS")

    mp_kg = planet_mass_mjup * _MJUP_KG
    rp_m = planet_radius_rjup * _RJUP_M
    grav_energy_j = _G * mp_kg**2 / rp_m

    if intrinsic_luminosity_lsun is not None:
        if intrinsic_luminosity_lsun <= 0.0:
            return PlanetCoolingResult(grav_energy_j, float("nan"), float("nan"),
                                        "UNKNOWN", "INVALID_LUMINOSITY")
        l_int_w = intrinsic_luminosity_lsun * _L_SUN
    elif planet_age_gyr is not None:
        if planet_age_gyr <= 0.0:
            return PlanetCoolingResult(grav_energy_j, float("nan"), float("nan"),
                                        "UNKNOWN", "INVALID_AGE")
        log_l = -5.0 - 1.3 * math.log10(planet_age_gyr)
        l_int_w = 10.0**log_l * _L_SUN * (planet_mass_mjup ** 1.5)
    else:
        # Use Jupiter's current intrinsic luminosity as default: ~8.7e17 W
        l_int_w = 8.7e17 * planet_mass_mjup

    tau_s = grav_energy_j / l_int_w
    tau_gyr = tau_s / _SEC_PER_GYR

    if tau_gyr < 1.0:
        cooling_class = "YOUNG_INFLATED"
    elif tau_gyr < 10.0:
        cooling_class = "CONTRACTING"
    else:
        cooling_class = "EVOLVED"

    return PlanetCoolingResult(
        gravitational_energy_j=grav_energy_j,
        intrinsic_luminosity_w=l_int_w,
        kelvin_helmholtz_timescale_gyr=tau_gyr,
        cooling_class=cooling_class,
        flag="OK",
    )


def format_planet_cooling_result(r: PlanetCoolingResult) -> str:
    if r.flag != "OK":
        return f"PlanetCooling | flag={r.flag}"
    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| Gravitational energy | {r.gravitational_energy_j:.3e} J |\n"
        f"| Intrinsic luminosity | {r.intrinsic_luminosity_w:.3e} W |\n"
        f"| KH cooling timescale | {r.kelvin_helmholtz_timescale_gyr:.2f} Gyr |\n"
        f"| Cooling class | {r.cooling_class} |\n"
        f"| Flag | {r.flag} |"
    )


def _cli() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Gas giant Kelvin-Helmholtz cooling timescale")
    p.add_argument("mass_mjup", type=float, help="Planet mass (Jupiter masses)")
    p.add_argument("radius_rjup", type=float, help="Planet radius (Jupiter radii)")
    p.add_argument("--lint", type=float, default=None, help="Intrinsic luminosity (L_sun)")
    p.add_argument("--age", type=float, default=None, help="Planet age (Gyr)")
    args = p.parse_args()
    r = compute_planet_cooling_timescale(args.mass_mjup, args.radius_rjup,
                                          intrinsic_luminosity_lsun=args.lint,
                                          planet_age_gyr=args.age)
    print(format_planet_cooling_result(r))


if __name__ == "__main__":
    _cli()
