"""Score atmospheric detectability against JWST/Ariel noise floors."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

# JWST NIRSpec/NIRISS single-transit noise floor ~10-20 ppm for bright stars
# Ariel Tier 2 noise floor ~50 ppm per transit
# Transmission amplitude ~ 2 * Rp * H / R*^2  (scale height units in ppm)
_K_BOLTZMANN = 1.38065e-23   # J/K
_AMU_KG = 1.6605e-27          # kg
_G_MS2 = 9.80665              # m/s^2 (Earth surface gravity)

# Instrument noise floors (ppm per transit)
_JWST_NOISE_PPM = 15.0
_ARIEL_NOISE_PPM = 50.0


@dataclass(frozen=True)
class AtmosphericDetectabilityResult:
    tsm: float                    # Transmission Spectroscopy Metric
    scale_height_km: float        # Atmospheric scale height
    transmission_amplitude_ppm: float   # 2 Rp H / R*^2
    n_transits_jwst: float        # transits needed for 5-sigma with JWST
    n_transits_ariel: float       # transits needed for 5-sigma with Ariel
    jwst_detectable_single: bool  # detectable in one transit with JWST
    ariel_detectable_single: bool
    flag: str


def score_atmospheric_detectability(
    planet_radius_rearth: float,
    stellar_radius_rsun: float,
    stellar_tmag: float,
    equilibrium_temp_k: float,
    planet_mass_mearth: float | None = None,
    mean_molecular_weight: float = 2.3,
    snr_threshold: float = 5.0,
) -> AtmosphericDetectabilityResult:
    """
    Score atmospheric detectability from key planet/star parameters.

    Scale height: H = k T_eq / (mu m_H g)
    Transmission amplitude: delta ~ 2 Rp H / R*^2 in ppm
    Instrument noise scaled by stellar brightness (Tmag).

    TSM ~ (Rp^3 * T_eq) / (Mp * R*^2) * 10^(-Tmag/5)
    (Kempton et al. 2018 simplified form)
    """
    if not math.isfinite(planet_radius_rearth) or planet_radius_rearth <= 0:
        return AtmosphericDetectabilityResult(
            tsm=float("nan"), scale_height_km=float("nan"),
            transmission_amplitude_ppm=float("nan"),
            n_transits_jwst=float("nan"), n_transits_ariel=float("nan"),
            jwst_detectable_single=False, ariel_detectable_single=False,
            flag="INVALID_RADIUS",
        )
    if not math.isfinite(equilibrium_temp_k) or equilibrium_temp_k <= 0:
        return AtmosphericDetectabilityResult(
            tsm=float("nan"), scale_height_km=float("nan"),
            transmission_amplitude_ppm=float("nan"),
            n_transits_jwst=float("nan"), n_transits_ariel=float("nan"),
            jwst_detectable_single=False, ariel_detectable_single=False,
            flag="INVALID_TEQILIBRIUM",
        )

    # Physical constants
    r_earth_m = 6.371e6
    r_sun_m = 6.957e8

    rp_m = planet_radius_rearth * r_earth_m
    rs_m = stellar_radius_rsun * r_sun_m

    # Estimate mass if not provided (Chen & Kipping 2017 simplified)
    if planet_mass_mearth is None:
        if planet_radius_rearth < 1.23:
            mp_mearth = planet_radius_rearth ** 3.0
        elif planet_radius_rearth < 14.26:
            mp_mearth = planet_radius_rearth ** 1.7
        else:
            mp_mearth = 0.2398 * planet_radius_rearth ** 0.55 * 317.83
        mp_mearth = max(0.1, mp_mearth)
    else:
        mp_mearth = planet_mass_mearth

    g_surface = _G_MS2 * (mp_mearth / planet_radius_rearth ** 2)

    # Scale height (km)
    mu_kg = mean_molecular_weight * _AMU_KG
    h_m = _K_BOLTZMANN * equilibrium_temp_k / (mu_kg * g_surface)
    h_km = h_m / 1000.0

    # Transmission amplitude (ppm) — 2 scale heights worth
    amp_ppm = 2.0 * rp_m * h_m / rs_m ** 2 * 1e6

    # Stellar brightness correction: noise ∝ 10^(Tmag/5) relative to Tmag=0
    brightness_factor = 10.0 ** (stellar_tmag / 5.0)
    jwst_noise = _JWST_NOISE_PPM * brightness_factor
    ariel_noise = _ARIEL_NOISE_PPM * brightness_factor

    # SNR per transit = amplitude / noise; n_transits = (threshold/SNR_1)^2
    snr_jwst_1 = amp_ppm / jwst_noise if jwst_noise > 0 else 0.0
    snr_ariel_1 = amp_ppm / ariel_noise if ariel_noise > 0 else 0.0

    n_jwst = (snr_threshold / snr_jwst_1) ** 2 if snr_jwst_1 > 0 else float("inf")
    n_ariel = (snr_threshold / snr_ariel_1) ** 2 if snr_ariel_1 > 0 else float("inf")

    # TSM (Kempton 2018 simplified; scale factor omitted)
    tsm = (planet_radius_rearth ** 3 * equilibrium_temp_k / (
        mp_mearth * stellar_radius_rsun ** 2
    )) * 10 ** (-stellar_tmag / 5.0)

    return AtmosphericDetectabilityResult(
        tsm=round(tsm, 4),
        scale_height_km=round(h_km, 2),
        transmission_amplitude_ppm=round(amp_ppm, 2),
        n_transits_jwst=round(n_jwst, 2) if math.isfinite(n_jwst) else float("inf"),
        n_transits_ariel=round(n_ariel, 2) if math.isfinite(n_ariel) else float("inf"),
        jwst_detectable_single=snr_jwst_1 >= snr_threshold,
        ariel_detectable_single=snr_ariel_1 >= snr_threshold,
        flag="OK",
    )


def format_detectability_result(r: AtmosphericDetectabilityResult) -> str:
    def _f(v: float, fmt: str = ".4f") -> str:
        return format(v, fmt) if math.isfinite(v) else "inf"

    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| TSM | {_f(r.tsm)} |\n"
        f"| Scale height (km) | {_f(r.scale_height_km, '.2f')} |\n"
        f"| Transmission amplitude (ppm) | {_f(r.transmission_amplitude_ppm, '.2f')} |\n"
        f"| N transits (JWST 5-sigma) | {_f(r.n_transits_jwst, '.1f')} |\n"
        f"| N transits (Ariel 5-sigma) | {_f(r.n_transits_ariel, '.1f')} |\n"
        f"| JWST single-transit detectable | {r.jwst_detectable_single} |\n"
        f"| Ariel single-transit detectable | {r.ariel_detectable_single} |\n"
        f"| Flag | {r.flag} |\n"
    )


def _cli() -> int:
    p = argparse.ArgumentParser(description="Score atmospheric detectability.")
    p.add_argument("planet_radius_rearth", type=float)
    p.add_argument("stellar_radius_rsun", type=float)
    p.add_argument("stellar_tmag", type=float)
    p.add_argument("equilibrium_temp_k", type=float)
    p.add_argument("--planet-mass-mearth", type=float, default=None)
    args = p.parse_args()
    r = score_atmospheric_detectability(
        args.planet_radius_rearth, args.stellar_radius_rsun,
        args.stellar_tmag, args.equilibrium_temp_k, args.planet_mass_mearth,
    )
    print(format_detectability_result(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
