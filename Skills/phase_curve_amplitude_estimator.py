"""Estimate thermal and reflected-light phase curve amplitudes for a transiting planet."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

_STEFAN_BOLTZMANN = 5.6704e-8   # W m⁻² K⁻⁴


@dataclass(frozen=True)
class PhaseCurveAmplitudeResult:
    planet_radius_rearth: float
    stellar_radius_rsun: float
    orbital_distance_au: float
    equilibrium_temp_k: float
    geometric_albedo: float
    reflected_amplitude_ppm: float   # reflected light variation amplitude
    thermal_amplitude_ppm: float     # thermal emission variation (day-night)
    secondary_eclipse_depth_ppm: float
    total_amplitude_ppm: float       # quadrature sum
    flag: str


_R_SUN_M = 6.957e8
_R_EARTH_M = 6.371e6
_AU_M = 1.496e11


def compute_phase_curve_amplitude(
    planet_radius_rearth: float,
    stellar_radius_rsun: float,
    orbital_distance_au: float,
    equilibrium_temp_k: float,
    stellar_teff_k: float = 5778.0,
    geometric_albedo: float = 0.1,
    dayside_factor: float = 1.26,   # Tday = f * Teq
    nightside_factor: float = 0.5,  # Tnight = g * Teq
) -> PhaseCurveAmplitudeResult:
    """
    Estimate phase curve amplitudes for reflected and thermal components.

    Reflected amplitude = Ag * (Rp/a)²  [ppm at quadrature]
    Thermal amplitude = (Rp/Rs)² * (Tday⁴ − Tnight⁴) / T*⁴  [ppm]
    Secondary eclipse depth = reflected + thermal eclipse depth.

    Parameters
    ----------
    planet_radius_rearth: Planet radius in Earth radii.
    stellar_radius_rsun:  Stellar radius in solar radii.
    orbital_distance_au:  Orbital distance in AU.
    equilibrium_temp_k:   Planet equilibrium temperature in K.
    stellar_teff_k:       Stellar effective temperature in K.
    geometric_albedo:     Geometric albedo Ag (default 0.1).
    dayside_factor:       Tday = dayside_factor * Teq.
    nightside_factor:     Tnight = nightside_factor * Teq.
    """
    if not math.isfinite(planet_radius_rearth) or planet_radius_rearth <= 0:
        return PhaseCurveAmplitudeResult(
            planet_radius_rearth, stellar_radius_rsun, orbital_distance_au,
            equilibrium_temp_k, geometric_albedo, float("nan"), float("nan"),
            float("nan"), float("nan"), "INVALID_RADIUS",
        )
    if not math.isfinite(orbital_distance_au) or orbital_distance_au <= 0:
        return PhaseCurveAmplitudeResult(
            planet_radius_rearth, stellar_radius_rsun, orbital_distance_au,
            equilibrium_temp_k, geometric_albedo, float("nan"), float("nan"),
            float("nan"), float("nan"), "INVALID_DISTANCE",
        )
    if not math.isfinite(equilibrium_temp_k) or equilibrium_temp_k <= 0:
        return PhaseCurveAmplitudeResult(
            planet_radius_rearth, stellar_radius_rsun, orbital_distance_au,
            equilibrium_temp_k, geometric_albedo, float("nan"), float("nan"),
            float("nan"), float("nan"), "INVALID_TEQ",
        )

    rp_m = planet_radius_rearth * _R_EARTH_M
    rs_m = stellar_radius_rsun * _R_SUN_M
    a_m = orbital_distance_au * _AU_M

    rp_over_rs_sq = (rp_m / rs_m) ** 2
    rp_over_a_sq = (rp_m / a_m) ** 2

    reflected_ppm = 1e6 * geometric_albedo * rp_over_a_sq

    t_day = dayside_factor * equilibrium_temp_k
    t_night = nightside_factor * equilibrium_temp_k
    thermal_ppm = 1e6 * rp_over_rs_sq * (t_day ** 4 - t_night ** 4) / stellar_teff_k ** 4

    thermal_eclipse = 1e6 * rp_over_rs_sq * (t_day ** 4) / stellar_teff_k ** 4
    secondary_depth = reflected_ppm + thermal_eclipse

    total_ppm = math.sqrt(reflected_ppm ** 2 + thermal_ppm ** 2)

    return PhaseCurveAmplitudeResult(
        planet_radius_rearth=planet_radius_rearth,
        stellar_radius_rsun=stellar_radius_rsun,
        orbital_distance_au=orbital_distance_au,
        equilibrium_temp_k=equilibrium_temp_k,
        geometric_albedo=geometric_albedo,
        reflected_amplitude_ppm=round(reflected_ppm, 4),
        thermal_amplitude_ppm=round(thermal_ppm, 4),
        secondary_eclipse_depth_ppm=round(secondary_depth, 4),
        total_amplitude_ppm=round(total_ppm, 4),
        flag="OK",
    )


def format_phase_curve_amplitude_result(r: PhaseCurveAmplitudeResult) -> str:
    def _f(v: float, fmt: str = ".4f") -> str:
        return format(v, fmt) if math.isfinite(v) else "N/A"

    return (
        f"| Parameter | Value |\n|---|---|\n"
        f"| Reflected amplitude (ppm) | {_f(r.reflected_amplitude_ppm)} |\n"
        f"| Thermal amplitude (ppm) | {_f(r.thermal_amplitude_ppm)} |\n"
        f"| Secondary eclipse depth (ppm) | {_f(r.secondary_eclipse_depth_ppm)} |\n"
        f"| Total amplitude (ppm) | {_f(r.total_amplitude_ppm)} |\n"
        f"| Geometric albedo | {_f(r.geometric_albedo)} |\n"
        f"| T_eq (K) | {_f(r.equilibrium_temp_k, '.1f')} |\n"
        f"| Flag | {r.flag} |\n"
    )


def _cli() -> int:
    p = argparse.ArgumentParser(description="Estimate phase curve amplitudes.")
    p.add_argument("planet_radius_rearth", type=float)
    p.add_argument("stellar_radius_rsun", type=float)
    p.add_argument("orbital_distance_au", type=float)
    p.add_argument("equilibrium_temp_k", type=float)
    p.add_argument("--stellar-teff-k", type=float, default=5778.0)
    p.add_argument("--geometric-albedo", type=float, default=0.1)
    args = p.parse_args()
    r = compute_phase_curve_amplitude(
        args.planet_radius_rearth, args.stellar_radius_rsun,
        args.orbital_distance_au, args.equilibrium_temp_k,
        stellar_teff_k=args.stellar_teff_k,
        geometric_albedo=args.geometric_albedo,
    )
    print(format_phase_curve_amplitude_result(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
