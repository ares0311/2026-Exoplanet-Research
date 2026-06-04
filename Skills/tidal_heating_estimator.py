"""Estimate tidal dissipation heating rate for a planet in an eccentric orbit."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

_G = 6.674e-11
_M_SUN_KG = 1.989e30
_M_JUP_KG = 1.898e27
_R_JUP_M = 7.149e7
_AU_M = 1.496e11
_IO_HEATING_W = 1e14       # Io tidal heat flux proxy ~10^14 W


@dataclass(frozen=True)
class TidalHeatingResult:
    period_days: float
    eccentricity: float
    heating_rate_w: float          # total tidal heating power (W)
    heating_flux_wm2: float        # surface heat flux (W/m²)
    io_multiples: float            # ratio to Io's tidal heating
    dominant_source: str           # ECCENTRICITY / OBLIQUITY_TIDE
    flag: str


def compute_tidal_heating(
    period_days: float,
    stellar_mass_msun: float = 1.0,
    planet_mass_mjup: float = 1.0,
    planet_radius_rjup: float = 1.0,
    eccentricity: float = 0.0,
    q_planet: float = 1e5,
    love_number_k2: float = 0.3,
) -> TidalHeatingResult:
    """
    Tidal dissipation heating rate (Peale & Cassen 1978; Jackson et al. 2008).

    dE/dt = (21/2) * k2 / Q * G * Ms² * Rp⁵ * n⁵ / a⁶ * e²
    where n = 2π/P (mean motion) and a is derived from Kepler's third law.

    Parameters
    ----------
    period_days:         Orbital period in days.
    stellar_mass_msun:   Host star mass in solar masses.
    planet_mass_mjup:    Planet mass in Jupiter masses.
    planet_radius_rjup:  Planet radius in Jupiter radii.
    eccentricity:        Orbital eccentricity.
    q_planet:            Tidal quality factor (default 10⁵ for gas giants).
    love_number_k2:      Second-order Love number (default 0.3).
    """
    if not math.isfinite(period_days) or period_days <= 0:
        return TidalHeatingResult(period_days, eccentricity, float("nan"),
                                  float("nan"), float("nan"), "UNKNOWN", "INVALID_PERIOD")
    if not math.isfinite(eccentricity) or eccentricity < 0 or eccentricity >= 1:
        return TidalHeatingResult(period_days, eccentricity, float("nan"),
                                  float("nan"), float("nan"), "UNKNOWN", "INVALID_ECCENTRICITY")

    ms_kg = stellar_mass_msun * _M_SUN_KG
    rp_m = planet_radius_rjup * _R_JUP_M
    mp_kg = planet_mass_mjup * _M_JUP_KG

    period_s = period_days * 86400.0
    n = 2.0 * math.pi / period_s

    a_m = ((_G * (ms_kg + mp_kg)) / (n ** 2)) ** (1.0 / 3.0)

    heating_w = (
        (21.0 / 2.0)
        * (love_number_k2 / q_planet)
        * _G
        * ms_kg ** 2
        * rp_m ** 5
        * n ** 5
        / a_m ** 6
        * eccentricity ** 2
    )

    surface_area_m2 = 4.0 * math.pi * rp_m ** 2
    heat_flux = heating_w / surface_area_m2
    io_mult = heating_w / _IO_HEATING_W

    return TidalHeatingResult(
        period_days=period_days,
        eccentricity=eccentricity,
        heating_rate_w=round(heating_w, 4) if heating_w > 0 else 0.0,
        heating_flux_wm2=round(heat_flux, 6),
        io_multiples=round(io_mult, 4),
        dominant_source="ECCENTRICITY",
        flag="OK",
    )


def format_tidal_heating_result(r: TidalHeatingResult) -> str:
    def _f(v: float, fmt: str = ".4f") -> str:
        return format(v, fmt) if math.isfinite(v) else "N/A"

    return (
        f"| Parameter | Value |\n|---|---|\n"
        f"| Period (days) | {_f(r.period_days)} |\n"
        f"| Eccentricity | {_f(r.eccentricity)} |\n"
        f"| Heating rate (W) | {_f(r.heating_rate_w, '.4e')} |\n"
        f"| Heat flux (W/m²) | {_f(r.heating_flux_wm2, '.4e')} |\n"
        f"| Io multiples | {_f(r.io_multiples, '.4e')} |\n"
        f"| Dominant source | {r.dominant_source} |\n"
        f"| Flag | {r.flag} |\n"
    )


def _cli() -> int:
    p = argparse.ArgumentParser(description="Estimate tidal heating for a planet.")
    p.add_argument("period_days", type=float)
    p.add_argument("--stellar-mass-msun", type=float, default=1.0)
    p.add_argument("--planet-mass-mjup", type=float, default=1.0)
    p.add_argument("--planet-radius-rjup", type=float, default=1.0)
    p.add_argument("--eccentricity", type=float, default=0.0)
    p.add_argument("--q-planet", type=float, default=1e5)
    p.add_argument("--love-number-k2", type=float, default=0.3)
    args = p.parse_args()
    r = compute_tidal_heating(
        args.period_days, args.stellar_mass_msun, args.planet_mass_mjup,
        args.planet_radius_rjup, args.eccentricity, args.q_planet, args.love_number_k2,
    )
    print(format_tidal_heating_result(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
