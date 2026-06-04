"""Estimate photometric distance from stellar apparent magnitude and physical parameters."""
from __future__ import annotations

import math
from dataclasses import dataclass

_SIGMA = 5.670374419e-8  # W m^-2 K^-4
_RSUN_M = 6.957e8
_L_SUN = 3.828e26        # W
_PC_M = 3.085677581e16   # metres per parsec


@dataclass(frozen=True)
class PhotometricDistanceResult:
    luminosity_lsun: float
    absolute_magnitude: float
    distance_pc: float
    distance_ly: float
    distance_modulus: float
    flag: str


def compute_photometric_distance(
    apparent_magnitude: float,
    stellar_teff_k: float,
    stellar_radius_rsun: float,
    extinction_mag: float = 0.0,
) -> PhotometricDistanceResult:
    """Estimate distance from apparent magnitude + Stefan-Boltzmann luminosity.

    L = 4π R² σ Teff⁴
    M_bol = M_bol_sun - 2.5 log10(L/L_sun)
    μ = m - M_bol - A  → d = 10^((μ+5)/5) pc

    Args:
        apparent_magnitude: V-band apparent magnitude
        stellar_teff_k: stellar effective temperature (K)
        stellar_radius_rsun: stellar radius (solar radii)
        extinction_mag: line-of-sight extinction in magnitudes (default 0)
    """
    _M_BOL_SUN = 4.74

    if stellar_teff_k <= 0.0:
        return PhotometricDistanceResult(float("nan"), float("nan"), float("nan"),
                                         float("nan"), float("nan"), "INVALID_TEFF")
    if stellar_radius_rsun <= 0.0:
        return PhotometricDistanceResult(float("nan"), float("nan"), float("nan"),
                                         float("nan"), float("nan"), "INVALID_RADIUS")
    if extinction_mag < 0.0:
        return PhotometricDistanceResult(float("nan"), float("nan"), float("nan"),
                                         float("nan"), float("nan"), "INVALID_EXTINCTION")

    r_m = stellar_radius_rsun * _RSUN_M
    l_w = 4.0 * math.pi * r_m**2 * _SIGMA * stellar_teff_k**4
    l_lsun = l_w / _L_SUN

    m_bol = _M_BOL_SUN - 2.5 * math.log10(l_lsun)
    dist_modulus = apparent_magnitude - m_bol - extinction_mag
    dist_pc = 10.0 ** ((dist_modulus + 5.0) / 5.0)
    dist_ly = dist_pc * 3.26156

    return PhotometricDistanceResult(
        luminosity_lsun=l_lsun,
        absolute_magnitude=m_bol,
        distance_pc=dist_pc,
        distance_ly=dist_ly,
        distance_modulus=dist_modulus,
        flag="OK",
    )


def format_photometric_distance_result(r: PhotometricDistanceResult) -> str:
    if r.flag != "OK":
        return f"PhotometricDistance | flag={r.flag}"
    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| Luminosity | {r.luminosity_lsun:.3f} L☉ |\n"
        f"| Absolute bolometric mag | {r.absolute_magnitude:.2f} |\n"
        f"| Distance modulus | {r.distance_modulus:.2f} mag |\n"
        f"| Distance | {r.distance_pc:.1f} pc |\n"
        f"| Distance | {r.distance_ly:.0f} ly |\n"
        f"| Flag | {r.flag} |"
    )


def _cli() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Photometric distance estimator")
    p.add_argument("apparent_mag", type=float, help="Apparent magnitude")
    p.add_argument("teff", type=float, help="Stellar Teff (K)")
    p.add_argument("radius_rsun", type=float, help="Stellar radius (Rsun)")
    p.add_argument("--extinction", type=float, default=0.0, help="Extinction (mag)")
    args = p.parse_args()
    r = compute_photometric_distance(args.apparent_mag, args.teff, args.radius_rsun,
                                     extinction_mag=args.extinction)
    print(format_photometric_distance_result(r))


if __name__ == "__main__":
    _cli()
