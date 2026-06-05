"""Compute wavelength-dependent transit depth from Rayleigh scattering."""
from __future__ import annotations

import math
from dataclasses import dataclass

_K_B = 1.381e-23
_G_EARTH = 9.807
_REARTH_M = 6.371e6
_MEARTH_KG = 5.972e24
_G_GRAV = 6.674e-11
_AMU_KG = 1.6605e-27


@dataclass(frozen=True)
class WavelengthTransitDepthResult:
    reference_depth_ppm: float
    scale_height_km: float
    rayleigh_slope_ppm_per_ln_wavelength: float
    wavelengths_um: tuple[float, ...]
    depths_ppm: tuple[float, ...]
    flag: str


def compute_wavelength_dependent_depth(
    reference_depth_ppm: float,
    reference_wavelength_um: float,
    wavelengths_um: list[float],
    planet_radius_rearth: float,
    stellar_radius_rsun: float,
    atmosphere_temp_k: float = 1000.0,
    mean_molecular_weight_amu: float = 2.3,
    planet_mass_mearth: float | None = None,
) -> WavelengthTransitDepthResult:
    """Compute Rayleigh-scattering wavelength-dependent transit depth.

    Lecavelier des Etangs et al. (2008):
      δ(λ) = δ₀ + (4 H Rp / Rs²) × ln(λ₀/λ)  [ppm]

    where H = k_B T / (μ g) is the scale height.

    Args:
        reference_depth_ppm: transit depth at reference wavelength (ppm)
        reference_wavelength_um: reference wavelength (microns)
        wavelengths_um: list of wavelengths to evaluate (microns)
        planet_radius_rearth: planet radius (Earth radii)
        stellar_radius_rsun: stellar radius (solar radii)
        atmosphere_temp_k: upper atmosphere temperature (K)
        mean_molecular_weight_amu: mean molecular weight (amu); 2.3 = solar/H2-dominated
        planet_mass_mearth: planet mass (Earth masses); if None uses surface gravity estimate
    """
    _RSUN_M = 6.957e8

    if reference_depth_ppm <= 0.0:
        return WavelengthTransitDepthResult(reference_depth_ppm, float("nan"),
                                             float("nan"), (), (), "INVALID_DEPTH")
    if reference_wavelength_um <= 0.0 or not wavelengths_um:
        return WavelengthTransitDepthResult(reference_depth_ppm, float("nan"),
                                             float("nan"), (), (), "INVALID_WAVELENGTH")
    if planet_radius_rearth <= 0.0 or stellar_radius_rsun <= 0.0:
        return WavelengthTransitDepthResult(reference_depth_ppm, float("nan"),
                                             float("nan"), (), (), "INVALID_RADII")

    rp_m = planet_radius_rearth * _REARTH_M
    rs_m = stellar_radius_rsun * _RSUN_M
    mu_kg = mean_molecular_weight_amu * _AMU_KG

    if planet_mass_mearth is not None and planet_mass_mearth > 0:
        mp_kg = planet_mass_mearth * _MEARTH_KG
        g = _G_GRAV * mp_kg / rp_m**2
    else:
        g = _G_EARTH

    h_m = _K_B * atmosphere_temp_k / (mu_kg * g)
    h_km = h_m / 1000.0

    slope = 4.0 * h_m * rp_m / rs_m**2 * 1e6

    depths = []
    for wl in wavelengths_um:
        if wl <= 0.0:
            depths.append(float("nan"))
        else:
            d = reference_depth_ppm + slope * math.log(reference_wavelength_um / wl)
            depths.append(max(d, 0.0))

    return WavelengthTransitDepthResult(
        reference_depth_ppm=reference_depth_ppm,
        scale_height_km=h_km,
        rayleigh_slope_ppm_per_ln_wavelength=slope,
        wavelengths_um=tuple(wavelengths_um),
        depths_ppm=tuple(depths),
        flag="OK",
    )


def format_wavelength_depth_result(r: WavelengthTransitDepthResult) -> str:
    if r.flag != "OK":
        return f"WavelengthTransitDepth | flag={r.flag}"
    rows = "\n".join(
        f"| {wl:.3f} | {d:.1f} |"
        for wl, d in zip(r.wavelengths_um, r.depths_ppm, strict=False)
    )
    return (
        f"Scale height: {r.scale_height_km:.1f} km | "
        f"Rayleigh slope: {r.rayleigh_slope_ppm_per_ln_wavelength:.2f} ppm/ln(λ) | "
        f"flag={r.flag}\n\n"
        f"| λ (μm) | Depth (ppm) |\n"
        f"|---|---|\n"
        + rows
    )


def _cli() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Rayleigh scattering wavelength-dependent depth")
    p.add_argument("depth_ppm", type=float)
    p.add_argument("ref_wavelength_um", type=float)
    p.add_argument("--wavelengths", type=float, nargs="+",
                   default=[0.5, 0.7, 1.0, 1.5, 2.0, 4.5])
    p.add_argument("--rp", type=float, default=1.0)
    p.add_argument("--rs", type=float, default=1.0)
    p.add_argument("--temp", type=float, default=1000.0)
    args = p.parse_args()
    r = compute_wavelength_dependent_depth(
        args.depth_ppm, args.ref_wavelength_um, args.wavelengths,
        args.rp, args.rs, atmosphere_temp_k=args.temp,
    )
    print(format_wavelength_depth_result(r))


if __name__ == "__main__":
    _cli()
