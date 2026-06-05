"""Compute apsidal precession rate from GR and tidal contributions."""
from __future__ import annotations

import math
from dataclasses import dataclass

_G = 6.674e-11
_C = 2.998e8
_MSUN_KG = 1.989e30
_MJUP_KG = 1.898e27
_RJUP_M = 7.1492e7
_RSUN_M = 6.957e8
_SEC_PER_YR = 3.1557e7


@dataclass(frozen=True)
class ApsidialPrecessionResult:
    gr_rate_deg_per_orbit: float       # GR contribution
    tidal_rate_deg_per_orbit: float    # tidal contribution
    total_rate_deg_per_orbit: float    # sum
    precession_period_yr: float        # full 360° precession period
    flag: str


def compute_apsidal_precession(
    period_days: float,
    eccentricity: float = 0.1,
    planet_mass_mjup: float = 1.0,
    stellar_mass_msun: float = 1.0,
    stellar_radius_rsun: float = 1.0,
    planet_radius_rjup: float = 1.0,
    love_number_k2_planet: float = 0.3,
    love_number_k2_star: float = 0.03,
) -> ApsidialPrecessionResult:
    """Compute apsidal precession from GR and tidal deformation.

    GR contribution (Einstein 1916):
      dω/dt_GR = (3 * G * Ms)^(3/2) / (c² * a^(5/2) * (1 - e²))  [rad/orbit]

    Tidal contribution (Sterne 1939):
      dω/dt_tidal = (k2_p + k2_s * (Ms/Mp) * (Rp/Rs)^5 * (Rs/a)^5) × f(e) per orbit

    Args:
        period_days: orbital period (days)
        eccentricity: orbital eccentricity
        planet_mass_mjup: planet mass (Jupiter masses)
        stellar_mass_msun: stellar mass (solar masses)
        stellar_radius_rsun: stellar radius (solar radii)
        planet_radius_rjup: planet radius (Jupiter radii)
        love_number_k2_planet: planetary Love number k₂
        love_number_k2_star: stellar Love number k₂
    """
    if period_days <= 0.0:
        return ApsidialPrecessionResult(float("nan"), float("nan"),
                                         float("nan"), float("nan"), "INVALID_PERIOD")
    if not (0.0 <= eccentricity < 1.0):
        return ApsidialPrecessionResult(float("nan"), float("nan"),
                                         float("nan"), float("nan"), "INVALID_ECCENTRICITY")
    if planet_mass_mjup <= 0.0:
        return ApsidialPrecessionResult(float("nan"), float("nan"),
                                         float("nan"), float("nan"), "INVALID_MASS")

    ms_kg = stellar_mass_msun * _MSUN_KG
    mp_kg = planet_mass_mjup * _MJUP_KG
    rs_m = stellar_radius_rsun * _RSUN_M
    rp_m = planet_radius_rjup * _RJUP_M
    p_s = period_days * 86400.0

    a_m = (_G * ms_kg * p_s**2 / (4.0 * math.pi**2)) ** (1.0 / 3.0)

    # GR rate (rad per second)
    gr_rate_rad_s = (3.0 * (_G * ms_kg) ** (3.0 / 2.0) /
                     (_C**2 * a_m**(5.0 / 2.0) * (1.0 - eccentricity**2)))
    gr_rate_rad_orbit = gr_rate_rad_s * p_s
    gr_deg_orbit = math.degrees(gr_rate_rad_orbit)

    # Tidal rate (Sterne 1939)
    # Eccentricity function f(e) = 1 + (3/2)e² + (1/8)e⁴ / (1-e²)^5
    one_minus_e2 = 1.0 - eccentricity**2
    f_ecc = (1.0 + 1.5 * eccentricity**2 + 0.125 * eccentricity**4) / one_minus_e2**5

    tidal_planet = 2.0 * love_number_k2_planet * (rp_m / a_m)**5 * (ms_kg / mp_kg) * f_ecc
    tidal_star = 2.0 * love_number_k2_star * (rs_m / a_m)**5 * (mp_kg / ms_kg) * f_ecc
    tidal_deg_orbit = math.degrees(tidal_planet + tidal_star) * (180.0 / math.pi)
    # correct: rates are already in radians, convert once
    tidal_deg_orbit = math.degrees(tidal_planet + tidal_star)

    total_deg_orbit = gr_deg_orbit + tidal_deg_orbit

    prec_period_yr = (360.0 / total_deg_orbit * period_days / 365.25
                      if total_deg_orbit > 0 else float("inf"))

    return ApsidialPrecessionResult(
        gr_rate_deg_per_orbit=gr_deg_orbit,
        tidal_rate_deg_per_orbit=tidal_deg_orbit,
        total_rate_deg_per_orbit=total_deg_orbit,
        precession_period_yr=prec_period_yr,
        flag="OK",
    )


def format_apsidal_precession_result(r: ApsidialPrecessionResult) -> str:
    if r.flag != "OK":
        return f"ApsidialPrecession | flag={r.flag}"
    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| GR rate | {r.gr_rate_deg_per_orbit:.4e} °/orbit |\n"
        f"| Tidal rate | {r.tidal_rate_deg_per_orbit:.4e} °/orbit |\n"
        f"| Total rate | {r.total_rate_deg_per_orbit:.4e} °/orbit |\n"
        f"| Precession period | {r.precession_period_yr:.2e} yr |\n"
        f"| Flag | {r.flag} |"
    )


def _cli() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Apsidal precession rate calculator")
    p.add_argument("period_days", type=float)
    p.add_argument("--ecc", type=float, default=0.1)
    p.add_argument("--mp", type=float, default=1.0)
    args = p.parse_args()
    r = compute_apsidal_precession(args.period_days, eccentricity=args.ecc,
                                    planet_mass_mjup=args.mp)
    print(format_apsidal_precession_result(r))


if __name__ == "__main__":
    _cli()
