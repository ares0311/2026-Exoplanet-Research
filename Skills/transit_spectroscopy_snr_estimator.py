"""Estimate transmission spectroscopy SNR per atmospheric scale height for JWST / Ariel."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

_R_EARTH_M = 6.371e6
_R_SUN_M = 6.957e8
_K_B = 1.381e-23       # Boltzmann constant
_M_H_KG = 1.673e-27    # hydrogen atom mass
_G_EARTH = 9.807       # m/s² — used as default surface gravity

# Instrument noise floors (ppm per transit, approximate)
_JWST_NOISE_FLOOR_PPM = 15.0
_ARIEL_NOISE_FLOOR_PPM = 50.0


@dataclass(frozen=True)
class TransmissionSpectroscopyResult:
    scale_height_km: float
    signal_per_scale_height_ppm: float
    n_scale_heights: int             # number of scale heights used in signal calculation
    n_transits_jwst_5sigma: int      # transits for 5σ JWST detection
    n_transits_ariel_5sigma: int     # transits for 5σ Ariel detection
    detectable_jwst_single: bool
    detectable_ariel_single: bool
    flag: str


def compute_transmission_spectroscopy_snr(
    planet_radius_rearth: float,
    stellar_radius_rsun: float,
    equilibrium_temp_k: float,
    planet_mass_mearth: float | None = None,
    mean_molecular_weight: float = 2.3,
    n_scale_heights: int = 5,
    surface_gravity_ms2: float | None = None,
) -> TransmissionSpectroscopyResult:
    """
    Estimate transmission spectroscopy signal and required JWST / Ariel transits.

    Scale height: H = k_B * T / (μ * g)
    Signal per scale height: δ = 2 * H * Rp / Rs²  [fractional; multiply by 1e6 for ppm]

    For n_scale_heights absorption features:
      Signal = n * δ  (ppm)

    N_transits = ceil[(target_snr * noise_floor / signal)²]

    Parameters
    ----------
    planet_radius_rearth:    Planet radius in Earth radii.
    stellar_radius_rsun:     Stellar radius in solar radii.
    equilibrium_temp_k:      Planet equilibrium temperature in K.
    planet_mass_mearth:      Planet mass in Earth masses (used for surface gravity; optional).
    mean_molecular_weight:   Mean molecular weight (default 2.3 = H₂-dominated).
    n_scale_heights:         Feature depth in scale heights to detect (default 5).
    surface_gravity_ms2:     Surface gravity in m/s² (overrides mass-based estimate).
    """
    if not math.isfinite(planet_radius_rearth) or planet_radius_rearth <= 0:
        return TransmissionSpectroscopyResult(float("nan"), float("nan"), n_scale_heights,
                                              0, 0, False, False, "INVALID_RADIUS")
    if not math.isfinite(stellar_radius_rsun) or stellar_radius_rsun <= 0:
        return TransmissionSpectroscopyResult(float("nan"), float("nan"), n_scale_heights,
                                              0, 0, False, False, "INVALID_STELLAR_RADIUS")
    if not math.isfinite(equilibrium_temp_k) or equilibrium_temp_k <= 0:
        return TransmissionSpectroscopyResult(float("nan"), float("nan"), n_scale_heights,
                                              0, 0, False, False, "INVALID_TEQ")

    rp_m = planet_radius_rearth * _R_EARTH_M
    rs_m = stellar_radius_rsun * _R_SUN_M

    if (surface_gravity_ms2 is not None
            and math.isfinite(surface_gravity_ms2) and surface_gravity_ms2 > 0):
        g = surface_gravity_ms2
    elif (planet_mass_mearth is not None
          and math.isfinite(planet_mass_mearth) and planet_mass_mearth > 0):
        _G = 6.674e-11
        _M_EARTH_KG = 5.972e24
        g = _G * planet_mass_mearth * _M_EARTH_KG / rp_m ** 2
    else:
        g = _G_EARTH  # use Earth gravity as default when mass is unknown

    mu_kg = mean_molecular_weight * _M_H_KG
    scale_height_m = _K_B * equilibrium_temp_k / (mu_kg * g)
    scale_height_km = scale_height_m / 1000.0

    signal_per_h_ppm = 1e6 * 2.0 * scale_height_m * rp_m / rs_m ** 2
    total_signal_ppm = n_scale_heights * signal_per_h_ppm

    def _n_transits(noise_floor: float, target_snr: float = 5.0) -> int:
        if total_signal_ppm <= 0:
            return 999999
        ratio = (target_snr * noise_floor / total_signal_ppm)
        return max(1, math.ceil(ratio ** 2))

    n_jwst = _n_transits(_JWST_NOISE_FLOOR_PPM)
    n_ariel = _n_transits(_ARIEL_NOISE_FLOOR_PPM)

    return TransmissionSpectroscopyResult(
        scale_height_km=round(scale_height_km, 4),
        signal_per_scale_height_ppm=round(signal_per_h_ppm, 4),
        n_scale_heights=n_scale_heights,
        n_transits_jwst_5sigma=n_jwst,
        n_transits_ariel_5sigma=n_ariel,
        detectable_jwst_single=n_jwst <= 1,
        detectable_ariel_single=n_ariel <= 1,
        flag="OK",
    )


def format_transmission_spectroscopy_result(r: TransmissionSpectroscopyResult) -> str:
    def _f(v: float, fmt: str = ".4f") -> str:
        return format(v, fmt) if math.isfinite(v) else "N/A"

    return (
        f"| Parameter | Value |\n|---|---|\n"
        f"| Scale height (km) | {_f(r.scale_height_km)} |\n"
        f"| Signal per H (ppm) | {_f(r.signal_per_scale_height_ppm)} |\n"
        f"| N scale heights | {r.n_scale_heights} |\n"
        f"| N transits JWST 5σ | {r.n_transits_jwst_5sigma} |\n"
        f"| N transits Ariel 5σ | {r.n_transits_ariel_5sigma} |\n"
        f"| Detectable (JWST single) | {r.detectable_jwst_single} |\n"
        f"| Detectable (Ariel single) | {r.detectable_ariel_single} |\n"
        f"| Flag | {r.flag} |\n"
    )


def _cli() -> int:
    p = argparse.ArgumentParser(description="Estimate transmission spectroscopy SNR.")
    p.add_argument("planet_radius_rearth", type=float)
    p.add_argument("stellar_radius_rsun", type=float)
    p.add_argument("equilibrium_temp_k", type=float)
    p.add_argument("--planet-mass-mearth", type=float, default=None)
    p.add_argument("--mean-molecular-weight", type=float, default=2.3)
    p.add_argument("--n-scale-heights", type=int, default=5)
    args = p.parse_args()
    r = compute_transmission_spectroscopy_snr(
        args.planet_radius_rearth, args.stellar_radius_rsun, args.equilibrium_temp_k,
        planet_mass_mearth=args.planet_mass_mearth,
        mean_molecular_weight=args.mean_molecular_weight,
        n_scale_heights=args.n_scale_heights,
    )
    print(format_transmission_spectroscopy_result(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
