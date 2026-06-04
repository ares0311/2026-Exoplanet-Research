"""Estimate expected TTV and TDV amplitudes from an exomoon (Kipping 2009)."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

_M_EARTH_KG = 5.972e24
_R_EARTH_M = 6.371e6
_G = 6.674e-11


@dataclass(frozen=True)
class ExomoonTtvResult:
    moon_mass_mearth: float
    moon_orbital_radius_km: float
    planet_mass_mearth: float
    planet_period_days: float
    ttv_amplitude_minutes: float    # transit timing variation RMS amplitude
    tdv_amplitude_minutes: float    # transit duration variation RMS amplitude
    detectable_tess: bool           # TESS TTV sensitivity ~2–5 min per transit
    detectable_cheops: bool         # CHEOPS ~1 min TTV
    flag: str


def compute_exomoon_ttv(
    moon_mass_mearth: float,
    moon_orbital_radius_km: float,
    planet_mass_mearth: float,
    planet_period_days: float,
    planet_orbital_radius_au: float = 0.1,
    stellar_mass_msun: float = 1.0,
) -> ExomoonTtvResult:
    """
    Estimate exomoon-induced TTV and TDV amplitudes.

    From Kipping (2009):
      TTV_rms ≈ (M_m / M_p) * (a_m / a_p) * P_p / (2π)
      TDV_rms ≈ TTV_rms * (v_m / v_p)

    where v_p is the planet's orbital velocity at the time of transit.

    Parameters
    ----------
    moon_mass_mearth:         Moon mass in Earth masses.
    moon_orbital_radius_km:   Moon semi-major axis in km.
    planet_mass_mearth:       Planet mass in Earth masses.
    planet_period_days:       Planet orbital period in days.
    planet_orbital_radius_au: Planet semi-major axis in AU.
    stellar_mass_msun:        Stellar mass in solar masses (for TDV scaling).
    """
    if not math.isfinite(planet_period_days) or planet_period_days <= 0:
        return ExomoonTtvResult(moon_mass_mearth, moon_orbital_radius_km, planet_mass_mearth,
                                planet_period_days, float("nan"), float("nan"),
                                False, False, "INVALID_PERIOD")
    if not math.isfinite(planet_mass_mearth) or planet_mass_mearth <= 0:
        return ExomoonTtvResult(moon_mass_mearth, moon_orbital_radius_km, planet_mass_mearth,
                                planet_period_days, float("nan"), float("nan"),
                                False, False, "INVALID_MASS")
    if not math.isfinite(moon_mass_mearth) or moon_mass_mearth <= 0:
        return ExomoonTtvResult(moon_mass_mearth, moon_orbital_radius_km, planet_mass_mearth,
                                planet_period_days, float("nan"), float("nan"),
                                False, False, "INVALID_MOON_MASS")

    _AU_M = 1.496e11
    ap_m = planet_orbital_radius_au * _AU_M
    am_m = moon_orbital_radius_km * 1000.0
    mp_kg = planet_mass_mearth * _M_EARTH_KG
    mm_kg = moon_mass_mearth * _M_EARTH_KG
    period_s = planet_period_days * 86400.0

    ttv_s = (mm_kg / mp_kg) * (am_m / ap_m) * period_s / (2.0 * math.pi)
    ttv_min = ttv_s / 60.0

    _M_SUN_KG = 1.989e30
    ms_kg = stellar_mass_msun * _M_SUN_KG
    vp_ms = math.sqrt(_G * ms_kg / ap_m) if ap_m > 0 else 1.0
    vm_ms = math.sqrt(_G * mp_kg / am_m) if am_m > 0 else 0.0

    tdv_s = ttv_s * (vm_ms / vp_ms) if vp_ms > 0 else 0.0
    tdv_min = tdv_s / 60.0

    return ExomoonTtvResult(
        moon_mass_mearth=moon_mass_mearth,
        moon_orbital_radius_km=moon_orbital_radius_km,
        planet_mass_mearth=planet_mass_mearth,
        planet_period_days=planet_period_days,
        ttv_amplitude_minutes=round(ttv_min, 4),
        tdv_amplitude_minutes=round(tdv_min, 4),
        detectable_tess=ttv_min >= 2.0,
        detectable_cheops=ttv_min >= 1.0,
        flag="OK",
    )


def format_exomoon_ttv_result(r: ExomoonTtvResult) -> str:
    def _f(v: float, fmt: str = ".4f") -> str:
        return format(v, fmt) if math.isfinite(v) else "N/A"

    return (
        f"| Parameter | Value |\n|---|---|\n"
        f"| Moon mass (M_earth) | {_f(r.moon_mass_mearth)} |\n"
        f"| Moon orbital radius (km) | {_f(r.moon_orbital_radius_km, '.1f')} |\n"
        f"| TTV amplitude (min) | {_f(r.ttv_amplitude_minutes)} |\n"
        f"| TDV amplitude (min) | {_f(r.tdv_amplitude_minutes)} |\n"
        f"| Detectable (TESS ~2 min) | {r.detectable_tess} |\n"
        f"| Detectable (CHEOPS ~1 min) | {r.detectable_cheops} |\n"
        f"| Flag | {r.flag} |\n"
    )


def _cli() -> int:
    p = argparse.ArgumentParser(description="Estimate exomoon TTV/TDV amplitudes.")
    p.add_argument("moon_mass_mearth", type=float)
    p.add_argument("moon_orbital_radius_km", type=float)
    p.add_argument("planet_mass_mearth", type=float)
    p.add_argument("planet_period_days", type=float)
    p.add_argument("--planet-orbital-radius-au", type=float, default=0.1)
    p.add_argument("--stellar-mass-msun", type=float, default=1.0)
    args = p.parse_args()
    r = compute_exomoon_ttv(
        args.moon_mass_mearth, args.moon_orbital_radius_km,
        args.planet_mass_mearth, args.planet_period_days,
        planet_orbital_radius_au=args.planet_orbital_radius_au,
        stellar_mass_msun=args.stellar_mass_msun,
    )
    print(format_exomoon_ttv_result(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
