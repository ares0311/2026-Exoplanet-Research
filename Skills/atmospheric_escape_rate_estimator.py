"""Estimate atmospheric escape rate via energy-limited photoionization."""
from __future__ import annotations

import math
from dataclasses import dataclass

_G = 6.674e-11
_MEARTH_KG = 5.972e24
_REARTH_M = 6.371e6
_AU_M = 1.495978707e11
_L_SUN = 3.828e26
_SEC_PER_GYR = 3.1557e16


@dataclass(frozen=True)
class AtmosphericEscapeResult:
    xuv_flux_wm2: float                # XUV flux at planet
    escape_rate_kg_s: float            # dM/dt (kg/s)
    mass_loss_timescale_gyr: float     # envelope lost in this time
    escape_class: str                  # SAFE / MODERATE / RAPID / CATASTROPHIC
    flag: str


def estimate_escape_rate(
    planet_radius_rearth: float,
    planet_mass_mearth: float,
    orbital_distance_au: float,
    stellar_xuv_luminosity_lsun: float = 1e-4,
    efficiency: float = 0.15,
    envelope_fraction: float = 0.01,
) -> AtmosphericEscapeResult:
    """Estimate energy-limited atmospheric escape rate.

    Energy-limited mass loss (Watson et al. 1981; Erkaev et al. 2007):
      dM/dt = η × π × Rp³ × F_XUV / (G × Mp × K_tide)
    where K_tide ≈ 1 (tidal correction, set to 1 for simplicity).

    Args:
        planet_radius_rearth: planet radius (Earth radii)
        planet_mass_mearth: planet mass (Earth masses)
        orbital_distance_au: orbital distance (AU)
        stellar_xuv_luminosity_lsun: stellar XUV luminosity (solar luminosities)
        efficiency: heating efficiency η (typically 0.10–0.20)
        envelope_fraction: envelope mass fraction (of total planet mass)
    """
    if planet_radius_rearth <= 0.0:
        return AtmosphericEscapeResult(float("nan"), float("nan"),
                                       float("nan"), "UNKNOWN", "INVALID_RADIUS")
    if planet_mass_mearth <= 0.0:
        return AtmosphericEscapeResult(float("nan"), float("nan"),
                                       float("nan"), "UNKNOWN", "INVALID_MASS")
    if orbital_distance_au <= 0.0:
        return AtmosphericEscapeResult(float("nan"), float("nan"),
                                       float("nan"), "UNKNOWN", "INVALID_DISTANCE")

    rp_m = planet_radius_rearth * _REARTH_M
    mp_kg = planet_mass_mearth * _MEARTH_KG
    a_m = orbital_distance_au * _AU_M
    l_xuv = stellar_xuv_luminosity_lsun * _L_SUN

    f_xuv = l_xuv / (4.0 * math.pi * a_m**2)

    k_tide = 1.0
    dm_dt = efficiency * math.pi * rp_m**3 * f_xuv / (_G * mp_kg * k_tide)

    envelope_mass_kg = envelope_fraction * mp_kg
    tau_s = envelope_mass_kg / dm_dt if dm_dt > 0 else float("inf")
    tau_gyr = tau_s / _SEC_PER_GYR

    dm_dt_norm = dm_dt / 1e6  # normalise to 10^6 kg/s (Earth oceans/Gyr scale)
    if dm_dt_norm < 0.01:
        escape_class = "SAFE"
    elif dm_dt_norm < 0.1:
        escape_class = "MODERATE"
    elif dm_dt_norm < 1.0:
        escape_class = "RAPID"
    else:
        escape_class = "CATASTROPHIC"

    return AtmosphericEscapeResult(
        xuv_flux_wm2=f_xuv,
        escape_rate_kg_s=dm_dt,
        mass_loss_timescale_gyr=tau_gyr,
        escape_class=escape_class,
        flag="OK",
    )


def format_escape_rate_result(r: AtmosphericEscapeResult) -> str:
    if r.flag != "OK":
        return f"AtmosphericEscape | flag={r.flag}"
    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| XUV flux | {r.xuv_flux_wm2:.3e} W/m² |\n"
        f"| Escape rate | {r.escape_rate_kg_s:.3e} kg/s |\n"
        f"| Envelope loss timescale | {r.mass_loss_timescale_gyr:.3e} Gyr |\n"
        f"| Escape class | {r.escape_class} |\n"
        f"| Flag | {r.flag} |"
    )


def _cli() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Atmospheric escape rate estimator")
    p.add_argument("planet_radius_rearth", type=float)
    p.add_argument("planet_mass_mearth", type=float)
    p.add_argument("orbital_distance_au", type=float)
    p.add_argument("--xuv", type=float, default=1e-4)
    args = p.parse_args()
    r = estimate_escape_rate(args.planet_radius_rearth, args.planet_mass_mearth,
                              args.orbital_distance_au, stellar_xuv_luminosity_lsun=args.xuv)
    print(format_escape_rate_result(r))


if __name__ == "__main__":
    _cli()
