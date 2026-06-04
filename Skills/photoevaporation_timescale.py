"""Energy-limited XUV photoevaporation timescale for exoplanet atmospheres."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

_G = 6.674e-11          # m³ kg⁻¹ s⁻²
_M_EARTH_KG = 5.972e24
_R_EARTH_M = 6.371e6
_R_JUP_M = 7.149e7
_M_JUP_KG = 1.898e27
_AU_M = 1.496e11
_S_TO_GYR = 1.0 / 3.156e16  # seconds → Gyr


@dataclass(frozen=True)
class PhotoevaporationResult:
    planet_radius_rearth: float
    planet_mass_mearth: float
    orbital_distance_au: float
    xuv_flux_wm2: float
    efficiency: float
    mass_loss_rate_gs: float      # g/s
    timescale_gyr: float          # atmosphere loss timescale
    stability_class: str          # STABLE / MARGINAL / UNSTABLE
    flag: str


def compute_photoevaporation_timescale(
    planet_radius_rearth: float,
    planet_mass_mearth: float,
    orbital_distance_au: float,
    xuv_flux_wm2: float,
    efficiency: float = 0.10,
    atmosphere_fraction: float = 0.01,
) -> PhotoevaporationResult:
    """
    Energy-limited XUV atmospheric escape timescale.

    Ṁ = η * π * Rp³ * F_XUV / (G * Mp * K)
    where K ≈ 1 (tidal correction, set to 1 for simplicity).
    Timescale τ = f_atm * Mp / Ṁ.

    Parameters
    ----------
    planet_radius_rearth:  Planet radius in Earth radii.
    planet_mass_mearth:    Planet mass in Earth masses.
    orbital_distance_au:   Semi-major axis in AU.
    xuv_flux_wm2:          Stellar XUV flux at the planet (W/m²).
    efficiency:            Heating efficiency η (default 0.10).
    atmosphere_fraction:   Atmospheric mass as fraction of total planet mass.
    """
    if not math.isfinite(planet_radius_rearth) or planet_radius_rearth <= 0:
        return PhotoevaporationResult(
            planet_radius_rearth, planet_mass_mearth, orbital_distance_au, xuv_flux_wm2,
            efficiency, float("nan"), float("nan"), "UNKNOWN", "INVALID_RADIUS",
        )
    if not math.isfinite(planet_mass_mearth) or planet_mass_mearth <= 0:
        return PhotoevaporationResult(
            planet_radius_rearth, planet_mass_mearth, orbital_distance_au, xuv_flux_wm2,
            efficiency, float("nan"), float("nan"), "UNKNOWN", "INVALID_MASS",
        )
    if not math.isfinite(xuv_flux_wm2) or xuv_flux_wm2 <= 0:
        return PhotoevaporationResult(
            planet_radius_rearth, planet_mass_mearth, orbital_distance_au, xuv_flux_wm2,
            efficiency, float("nan"), float("nan"), "UNKNOWN", "INVALID_XUV_FLUX",
        )

    rp_m = planet_radius_rearth * _R_EARTH_M
    mp_kg = planet_mass_mearth * _M_EARTH_KG

    mass_loss_kgs = efficiency * math.pi * rp_m ** 3 * xuv_flux_wm2 / (_G * mp_kg)
    mass_loss_gs = mass_loss_kgs * 1e3

    atm_mass_kg = atmosphere_fraction * mp_kg
    timescale_s = atm_mass_kg / mass_loss_kgs
    timescale_gyr = timescale_s * _S_TO_GYR

    if timescale_gyr >= 10.0:
        stability = "STABLE"
    elif timescale_gyr >= 1.0:
        stability = "MARGINAL"
    else:
        stability = "UNSTABLE"

    return PhotoevaporationResult(
        planet_radius_rearth=planet_radius_rearth,
        planet_mass_mearth=planet_mass_mearth,
        orbital_distance_au=orbital_distance_au,
        xuv_flux_wm2=xuv_flux_wm2,
        efficiency=efficiency,
        mass_loss_rate_gs=round(mass_loss_gs, 4),
        timescale_gyr=round(timescale_gyr, 4),
        stability_class=stability,
        flag="OK",
    )


def format_photoevaporation_result(r: PhotoevaporationResult) -> str:
    def _f(v: float, fmt: str = ".4f") -> str:
        return format(v, fmt) if math.isfinite(v) else "N/A"

    return (
        f"| Parameter | Value |\n|---|---|\n"
        f"| Planet radius (R_earth) | {_f(r.planet_radius_rearth)} |\n"
        f"| Planet mass (M_earth) | {_f(r.planet_mass_mearth)} |\n"
        f"| XUV flux (W/m²) | {_f(r.xuv_flux_wm2, '.4e')} |\n"
        f"| Mass loss rate (g/s) | {_f(r.mass_loss_rate_gs, '.4e')} |\n"
        f"| Timescale (Gyr) | {_f(r.timescale_gyr)} |\n"
        f"| Stability class | {r.stability_class} |\n"
        f"| Flag | {r.flag} |\n"
    )


def _cli() -> int:
    p = argparse.ArgumentParser(description="Estimate XUV photoevaporation timescale.")
    p.add_argument("planet_radius_rearth", type=float)
    p.add_argument("planet_mass_mearth", type=float)
    p.add_argument("orbital_distance_au", type=float)
    p.add_argument("xuv_flux_wm2", type=float)
    p.add_argument("--efficiency", type=float, default=0.10)
    p.add_argument("--atmosphere-fraction", type=float, default=0.01)
    args = p.parse_args()
    r = compute_photoevaporation_timescale(
        args.planet_radius_rearth, args.planet_mass_mearth,
        args.orbital_distance_au, args.xuv_flux_wm2,
        efficiency=args.efficiency, atmosphere_fraction=args.atmosphere_fraction,
    )
    print(format_photoevaporation_result(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
