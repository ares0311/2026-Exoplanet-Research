"""Predict thermal and reflected secondary eclipse depth at a given wavelength."""
from __future__ import annotations

import math
from dataclasses import dataclass

_H = 6.626e-34      # Planck constant
_C = 2.998e8        # speed of light m/s
_K_B = 1.381e-23    # Boltzmann constant


def _planck_wm3(wavelength_m: float, temp_k: float) -> float:
    """Spectral radiance B_λ in W m^-3 sr^-1."""
    x = _H * _C / (_K_B * temp_k * wavelength_m)
    if x > 700.0:
        return 0.0
    return (2.0 * _H * _C**2 / wavelength_m**5) / (math.exp(x) - 1.0)


@dataclass(frozen=True)
class SecondaryEclipseDepthResult:
    wavelength_um: float
    thermal_depth_ppm: float
    reflected_depth_ppm: float
    total_depth_ppm: float
    thermal_fraction: float
    day_side_temp_k: float
    flag: str


def predict_secondary_eclipse_depth(
    planet_radius_rjup: float,
    stellar_radius_rsun: float,
    stellar_teff_k: float,
    equilibrium_temp_k: float,
    wavelength_um: float = 4.5,
    geometric_albedo: float = 0.1,
    day_night_ratio: float = 1.3,
) -> SecondaryEclipseDepthResult:
    """Predict secondary eclipse depth from Planck functions + reflected light.

    Thermal: depth = (Rp/Rs)² × B_λ(T_day) / B_λ(T_star)
    Reflected: depth = Ag × (Rp/a)²  [not wavelength-dependent]
    Total = thermal + reflected

    Args:
        planet_radius_rjup: planet radius (Jupiter radii)
        stellar_radius_rsun: stellar radius (solar radii)
        stellar_teff_k: stellar effective temperature (K)
        equilibrium_temp_k: planet equilibrium temperature (K)
        wavelength_um: observation wavelength (microns)
        geometric_albedo: planet geometric albedo
        day_night_ratio: ratio of day-side to equilibrium temperature (typ. 1.1–1.5)
    """
    _RJUP_M = 7.1492e7
    _RSUN_M = 6.957e8

    if planet_radius_rjup <= 0.0:
        return SecondaryEclipseDepthResult(wavelength_um, float("nan"), float("nan"),
                                            float("nan"), float("nan"), float("nan"),
                                            "INVALID_PLANET_RADIUS")
    if stellar_radius_rsun <= 0.0:
        return SecondaryEclipseDepthResult(wavelength_um, float("nan"), float("nan"),
                                            float("nan"), float("nan"), float("nan"),
                                            "INVALID_STELLAR_RADIUS")
    if stellar_teff_k <= 0.0:
        return SecondaryEclipseDepthResult(wavelength_um, float("nan"), float("nan"),
                                            float("nan"), float("nan"), float("nan"),
                                            "INVALID_STELLAR_TEFF")
    if equilibrium_temp_k <= 0.0:
        return SecondaryEclipseDepthResult(wavelength_um, float("nan"), float("nan"),
                                            float("nan"), float("nan"), float("nan"),
                                            "INVALID_TEQ")
    if wavelength_um <= 0.0:
        return SecondaryEclipseDepthResult(wavelength_um, float("nan"), float("nan"),
                                            float("nan"), float("nan"), float("nan"),
                                            "INVALID_WAVELENGTH")

    rp_m = planet_radius_rjup * _RJUP_M
    rs_m = stellar_radius_rsun * _RSUN_M
    wl_m = wavelength_um * 1e-6
    t_day = equilibrium_temp_k * day_night_ratio

    b_planet = _planck_wm3(wl_m, t_day)
    b_star = _planck_wm3(wl_m, stellar_teff_k)

    radius_ratio_sq = (rp_m / rs_m) ** 2
    thermal_depth_ppm = (1e6 * radius_ratio_sq * b_planet / b_star) if b_star > 0.0 else 0.0

    # Reflected: Ag × (Rp/Rs)²  (upper bound without a/Rs info)
    reflected_depth_ppm = 1e6 * geometric_albedo * radius_ratio_sq

    total_depth_ppm = thermal_depth_ppm + reflected_depth_ppm
    thermal_fraction = thermal_depth_ppm / total_depth_ppm if total_depth_ppm > 0 else 0.0

    return SecondaryEclipseDepthResult(
        wavelength_um=wavelength_um,
        thermal_depth_ppm=thermal_depth_ppm,
        reflected_depth_ppm=reflected_depth_ppm,
        total_depth_ppm=total_depth_ppm,
        thermal_fraction=thermal_fraction,
        day_side_temp_k=t_day,
        flag="OK",
    )


def format_secondary_eclipse_result(r: SecondaryEclipseDepthResult) -> str:
    if r.flag != "OK":
        return f"SecondaryEclipseDepth | flag={r.flag}"
    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| Wavelength | {r.wavelength_um:.2f} μm |\n"
        f"| Day-side temperature | {r.day_side_temp_k:.0f} K |\n"
        f"| Thermal eclipse depth | {r.thermal_depth_ppm:.1f} ppm |\n"
        f"| Reflected eclipse depth | {r.reflected_depth_ppm:.1f} ppm |\n"
        f"| Total eclipse depth | {r.total_depth_ppm:.1f} ppm |\n"
        f"| Thermal fraction | {r.thermal_fraction:.3f} |\n"
        f"| Flag | {r.flag} |"
    )


def _cli() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Secondary eclipse depth predictor")
    p.add_argument("rp_rjup", type=float, help="Planet radius (Rjup)")
    p.add_argument("rs_rsun", type=float, help="Stellar radius (Rsun)")
    p.add_argument("teff_k", type=float, help="Stellar Teff (K)")
    p.add_argument("teq_k", type=float, help="Equilibrium temperature (K)")
    p.add_argument("--wavelength", type=float, default=4.5, help="Wavelength (microns)")
    p.add_argument("--albedo", type=float, default=0.1, help="Geometric albedo")
    args = p.parse_args()
    r = predict_secondary_eclipse_depth(
        args.rp_rjup, args.rs_rsun, args.teff_k, args.teq_k,
        wavelength_um=args.wavelength, geometric_albedo=args.albedo,
    )
    print(format_secondary_eclipse_result(r))


if __name__ == "__main__":
    _cli()
