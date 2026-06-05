"""Estimate stellar wind ram pressure and photoionization balance for exoplanet atmospheres."""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class WindPhotoionizationResult:
    wind_ram_pressure_pa: float         # stellar wind dynamic pressure at planet (Pa)
    xuv_flux_ergcm2s: float             # XUV flux at planet (erg/cm²/s)
    ionization_timescale_s: float       # characteristic photoionization timescale (s)
    mass_loss_rate_gs: float            # XUV-driven mass loss rate (g/s)
    magnetosphere_standoff_rp: float    # magnetopause standoff (in planet radii)
    atmosphere_class: str               # RETAINED / VULNERABLE / STRIPPED / UNKNOWN
    flag: str


_MSUN_KG = 1.989e30
_RSUN_M = 6.957e8
_MEARTH_KG = 5.972e24
_REARTH_M = 6.371e6
_AU_M = 1.496e11
_BOLTZMANN = 1.381e-23
_PROTON_MASS_KG = 1.673e-27


def estimate_wind_photoionization_balance(
    stellar_mass_msun: float,
    stellar_radius_rsun: float,
    stellar_age_gyr: float,
    orbital_distance_au: float,
    planet_mass_mearth: float,
    planet_radius_rearth: float,
    stellar_wind_velocity_kms: float = 400.0,
    planetary_magnetic_moment_earth: float = 1.0,
) -> WindPhotoionizationResult:
    """Estimate stellar wind ram pressure and XUV-driven atmospheric escape.

    Stellar wind density at distance d (Parker 1958 approximation):
      n_sw = n_sw0 × (R★/d)²    [cm⁻³]
    with n_sw0 ≈ 10 × (M★/M☉)^1.5  [cm⁻³] at 1 AU

    XUV luminosity scaling (Ribas+2005, Sanz-Forcada+2011):
      L_XUV / L☉ ≈ 10^(-3.5) × (age/Gyr)^(-1.5)  for solar-type stars

    Energy-limited mass loss (Watson+1981):
      dM/dt = η × π × Rp² × F_XUV / (G × Mp / Rp)  [g/s]

    Magnetopause standoff (Magnetopause balance):
      R_mp ≈ Rp × (B²/(8π × P_sw))^(1/6)

    Args:
        stellar_mass_msun: stellar mass (solar masses)
        stellar_radius_rsun: stellar radius (solar radii)
        stellar_age_gyr: stellar age (Gyr)
        orbital_distance_au: planet orbital distance (AU)
        planet_mass_mearth: planet mass (Earth masses)
        planet_radius_rearth: planet radius (Earth radii)
        stellar_wind_velocity_kms: stellar wind velocity (km/s)
        planetary_magnetic_moment_earth: planetary magnetic moment (Earth units)
    """
    if stellar_mass_msun <= 0.0:
        return WindPhotoionizationResult(float("nan"), float("nan"), float("nan"),
                                          float("nan"), float("nan"), "UNKNOWN",
                                          "INVALID_STELLAR_MASS")
    if orbital_distance_au <= 0.0:
        return WindPhotoionizationResult(float("nan"), float("nan"), float("nan"),
                                          float("nan"), float("nan"), "UNKNOWN",
                                          "INVALID_DISTANCE")
    if planet_mass_mearth <= 0.0:
        return WindPhotoionizationResult(float("nan"), float("nan"), float("nan"),
                                          float("nan"), float("nan"), "UNKNOWN",
                                          "INVALID_PLANET_MASS")
    if stellar_age_gyr <= 0.0:
        return WindPhotoionizationResult(float("nan"), float("nan"), float("nan"),
                                          float("nan"), float("nan"), "UNKNOWN",
                                          "INVALID_AGE")

    v_sw_ms = stellar_wind_velocity_kms * 1e3

    # Stellar wind number density at 1 AU (cm⁻³), scaled with stellar mass
    n_sw_1au_cm3 = 10.0 * stellar_mass_msun ** 1.5

    # Scale with distance: n ∝ r⁻²
    n_sw_cm3 = n_sw_1au_cm3 / (orbital_distance_au ** 2)
    n_sw_m3 = n_sw_cm3 * 1e6   # cm⁻³ → m⁻³

    # Ram pressure: P_ram = ρ × v² = n × m_p × v²
    rho_sw = n_sw_m3 * _PROTON_MASS_KG
    ram_pressure_pa = rho_sw * v_sw_ms ** 2

    # XUV luminosity scaling (Ribas+2005): L_XUV/L☉ ≈ 10^(-3.5) × age^(-1.5)
    l_xuv_lsun = 10.0 ** (-3.5) * stellar_mass_msun ** 2.0 * (stellar_age_gyr ** (-1.5))
    l_xuv_ergs = l_xuv_lsun * 3.828e33   # erg/s

    # XUV flux at planet
    d_cm = orbital_distance_au * _AU_M * 100.0   # AU → cm
    xuv_flux_ergcm2s = l_xuv_ergs / (4.0 * math.pi * d_cm ** 2)

    # Photoionization timescale: τ ≈ E_ion / F_xuv × (4πRp²)/(πRp²)
    # Simple: τ_ion ≈ N_ion × E_ion / (F_xuv × Rp²) — order of magnitude
    # For hydrogen: E_ion ≈ 13.6 eV = 2.18e-11 erg
    e_ion_erg = 2.18e-11
    # Column density proxy: N ≈ P_atm/(m_H × g) ~ 1e25 cm⁻² for Earth-like
    n_col_cm2 = 1.0e25 * (planet_mass_mearth / planet_radius_rearth ** 2)
    ion_timescale_s = (n_col_cm2 * e_ion_erg) / max(xuv_flux_ergcm2s, 1e-10)

    # Energy-limited mass loss (Watson+1981):
    # dM/dt = η × π × Rp² × F_XUV / (G × Mp/Rp)
    # efficiency η ~ 0.15 for moderate XUV
    eta = 0.15
    _G = 6.674e-11
    mp_kg = planet_mass_mearth * _MEARTH_KG
    rp_m = planet_radius_rearth * _REARTH_M
    f_xuv_si = xuv_flux_ergcm2s * 1e-3   # erg/cm²/s → W/m²

    gravitational_potential = _G * mp_kg / rp_m   # J/kg
    mass_loss_kgs = eta * math.pi * rp_m ** 2 * f_xuv_si / gravitational_potential
    mass_loss_gs = mass_loss_kgs * 1e3   # kg/s → g/s

    # Magnetopause standoff: R_mp / Rp ≈ (M/M_Earth)^(1/3) × 10
    r_mp_rp = 10.0 * planetary_magnetic_moment_earth ** (1.0 / 3.0)

    # Atmospheric classification based on mass loss timescale
    # Compare total atmosphere mass to loss rate
    m_atm_kg = mp_kg * 1e-4   # ~0.01% atmosphere by mass
    t_loss_yr = (m_atm_kg * 1e3) / mass_loss_gs / (3.156e7) if mass_loss_gs > 0 else float("inf")

    if t_loss_yr > 10.0e9:   # longer than Hubble time
        atm_class = "RETAINED"
    elif t_loss_yr > 1.0e9:
        atm_class = "VULNERABLE"
    elif t_loss_yr > 1.0e7:
        atm_class = "STRIPPED"
    else:
        atm_class = "STRIPPED"

    # Override: strong magnetic field protects atmosphere
    if planetary_magnetic_moment_earth >= 5.0 and atm_class == "VULNERABLE":
        atm_class = "RETAINED"

    return WindPhotoionizationResult(
        wind_ram_pressure_pa=ram_pressure_pa,
        xuv_flux_ergcm2s=xuv_flux_ergcm2s,
        ionization_timescale_s=ion_timescale_s,
        mass_loss_rate_gs=mass_loss_gs,
        magnetosphere_standoff_rp=r_mp_rp,
        atmosphere_class=atm_class,
        flag="OK",
    )


def format_wind_photoionization_result(r: WindPhotoionizationResult) -> str:
    if r.flag != "OK":
        return f"WindPhotoionization | flag={r.flag}"
    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| Wind ram pressure | {r.wind_ram_pressure_pa:.3e} Pa |\n"
        f"| XUV flux | {r.xuv_flux_ergcm2s:.3e} erg/cm²/s |\n"
        f"| Ionization timescale | {r.ionization_timescale_s:.3e} s |\n"
        f"| Mass loss rate | {r.mass_loss_rate_gs:.3e} g/s |\n"
        f"| Magnetosphere standoff | {r.magnetosphere_standoff_rp:.1f} Rp |\n"
        f"| Atmosphere class | {r.atmosphere_class} |\n"
        f"| Flag | {r.flag} |"
    )


def _cli() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Stellar wind / photoionization balance estimator")
    p.add_argument("stellar_mass_msun", type=float)
    p.add_argument("stellar_radius_rsun", type=float)
    p.add_argument("stellar_age_gyr", type=float)
    p.add_argument("orbital_distance_au", type=float)
    p.add_argument("planet_mass_mearth", type=float)
    p.add_argument("planet_radius_rearth", type=float)
    p.add_argument("--vwind", type=float, default=400.0, help="Wind velocity km/s")
    args = p.parse_args()
    r = estimate_wind_photoionization_balance(
        args.stellar_mass_msun, args.stellar_radius_rsun, args.stellar_age_gyr,
        args.orbital_distance_au, args.planet_mass_mearth, args.planet_radius_rearth,
        stellar_wind_velocity_kms=args.vwind,
    )
    print(format_wind_photoionization_result(r))


if __name__ == "__main__":
    _cli()
