"""Compute stellar wind ram pressure and magnetospheric standoff distance."""
from __future__ import annotations

import math
from dataclasses import dataclass

_AU_M = 1.495978707e11
_RSUN_M = 6.957e8
_MSUN_KG = 1.989e30
_MP_KG = 1.6726e-27    # proton mass
_MU0 = 1.2566370614e-6  # permeability of free space


@dataclass(frozen=True)
class StellarWindResult:
    orbital_distance_au: float
    wind_density_kg_m3: float
    wind_velocity_ms: float
    ram_pressure_pa: float
    magnetospheric_standoff_rp: float   # in units of planet radii
    space_weather_class: str            # QUIET / MODERATE / ACTIVE / EXTREME
    flag: str


def compute_stellar_wind_ram_pressure(
    orbital_distance_au: float,
    stellar_mass_loss_rate_msun_yr: float = 2e-14,
    wind_velocity_ms: float = 4e5,
    planet_radius_rearth: float = 1.0,
    planet_magnetic_moment_earth: float = 1.0,
) -> StellarWindResult:
    """Compute stellar wind ram pressure and magnetospheric standoff.

    Wind density from mass conservation: ρ(r) = Ṁ / (4π r² v_w)
    Ram pressure: P_ram = ρ v_w²
    Magnetospheric standoff: r_m where P_ram = B²/(2μ₀), B = B₀(R_p/r_m)³

    Args:
        orbital_distance_au: orbital distance (AU)
        stellar_mass_loss_rate_msun_yr: stellar mass loss rate (solar masses per year)
        wind_velocity_ms: stellar wind speed at planet (m/s); ~400 km/s for Sun
        planet_radius_rearth: planet radius (Earth radii)
        planet_magnetic_moment_earth: planetary magnetic moment in Earth units
    """
    _REARTH_M = 6.371e6
    _MDOT_SUN = 2e-14 * _MSUN_KG / 3.156e7  # kg/s at solar rate

    if orbital_distance_au <= 0.0:
        return StellarWindResult(orbital_distance_au, float("nan"), float("nan"),
                                  float("nan"), float("nan"), "UNKNOWN", "INVALID_DISTANCE")
    if stellar_mass_loss_rate_msun_yr <= 0.0:
        return StellarWindResult(orbital_distance_au, float("nan"), float("nan"),
                                  float("nan"), float("nan"), "UNKNOWN", "INVALID_MDOT")
    if wind_velocity_ms <= 0.0:
        return StellarWindResult(orbital_distance_au, float("nan"), float("nan"),
                                  float("nan"), float("nan"), "UNKNOWN", "INVALID_VELOCITY")

    mdot_kg_s = stellar_mass_loss_rate_msun_yr * _MSUN_KG / 3.156e7
    r_m = orbital_distance_au * _AU_M
    rho = mdot_kg_s / (4.0 * math.pi * r_m**2 * wind_velocity_ms)
    p_ram = rho * wind_velocity_ms**2

    rp_m = planet_radius_rearth * _REARTH_M
    # Earth surface field B₀ ≈ 3×10⁻⁵ T; scaled by magnetic moment
    b0_earth = 3e-5  # T
    b0 = b0_earth * planet_magnetic_moment_earth
    # Standoff: B²(r_m)/(2μ₀) = P_ram; B(r_m) = B₀(R_p/r_m)³ [dipole]
    # r_m = R_p × (B₀²/(2μ₀ P_ram))^(1/6)
    if p_ram > 0.0:
        r_m_standoff = rp_m * (b0**2 / (2.0 * _MU0 * p_ram)) ** (1.0 / 6.0)
        standoff_rp = r_m_standoff / rp_m
    else:
        standoff_rp = float("inf")

    p_ram_norm = p_ram / 2e-9  # normalised to 1 nPa (typical SW at Earth)
    if p_ram_norm < 0.5:
        sw_class = "QUIET"
    elif p_ram_norm < 2.0:
        sw_class = "MODERATE"
    elif p_ram_norm < 10.0:
        sw_class = "ACTIVE"
    else:
        sw_class = "EXTREME"

    return StellarWindResult(
        orbital_distance_au=orbital_distance_au,
        wind_density_kg_m3=rho,
        wind_velocity_ms=wind_velocity_ms,
        ram_pressure_pa=p_ram,
        magnetospheric_standoff_rp=standoff_rp,
        space_weather_class=sw_class,
        flag="OK",
    )


def format_stellar_wind_result(r: StellarWindResult) -> str:
    if r.flag != "OK":
        return f"StellarWindRamPressure | flag={r.flag}"
    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| Orbital distance | {r.orbital_distance_au:.3f} AU |\n"
        f"| Wind density | {r.wind_density_kg_m3:.3e} kg/m³ |\n"
        f"| Wind velocity | {r.wind_velocity_ms:.0f} m/s |\n"
        f"| Ram pressure | {r.ram_pressure_pa:.3e} Pa |\n"
        f"| Magnetospheric standoff | {r.magnetospheric_standoff_rp:.2f} Rp |\n"
        f"| Space weather class | {r.space_weather_class} |\n"
        f"| Flag | {r.flag} |"
    )


def _cli() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Stellar wind ram pressure calculator")
    p.add_argument("orbital_distance_au", type=float)
    p.add_argument("--mdot", type=float, default=2e-14, help="Mass loss rate (Msun/yr)")
    p.add_argument("--vwind", type=float, default=4e5, help="Wind velocity (m/s)")
    args = p.parse_args()
    r = compute_stellar_wind_ram_pressure(
        args.orbital_distance_au,
        stellar_mass_loss_rate_msun_yr=args.mdot,
        wind_velocity_ms=args.vwind,
    )
    print(format_stellar_wind_result(r))


if __name__ == "__main__":
    _cli()
