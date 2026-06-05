"""Compute atmospheric transmission feature amplitude (ppm) from scale height."""
from __future__ import annotations

from dataclasses import dataclass

_K_B = 1.380649e-23   # Boltzmann constant (J/K)
_G_EARTH = 9.807       # m/s²
_MEARTH_KG = 5.972e24
_REARTH_M = 6.371e6
_AMU_KG = 1.6605e-27


@dataclass(frozen=True)
class TransmissionResult:
    scale_height_km: float          # atmospheric scale height H (km)
    signal_per_scale_height_ppm: float   # δ per H = 2 * (Rp/Rs²) * H/Rs * 1e6
    n_scale_heights: float          # assumed scale heights in feature
    feature_amplitude_ppm: float    # total amplitude
    detectability: str              # DETECTABLE / MARGINAL / UNDETECTABLE
    flag: str


def compute_transmission_amplitude(
    planet_radius_rearth: float,
    planet_mass_mearth: float,
    stellar_radius_rsun: float,
    equilibrium_temperature_k: float,
    mean_molecular_weight_amu: float = 2.3,
    n_scale_heights: float = 5.0,
    photometric_precision_ppm: float = 50.0,
) -> TransmissionResult:
    """Compute atmospheric transmission feature amplitude.

    Scale height: H = k_B * T_eq / (μ * g)
    Transmission amplitude per scale height: δ = 2 * Rp * H / Rs²
    Feature amplitude = n_H * δ  (n_H typically 5–10 for broad features)

    Args:
        planet_radius_rearth: planet radius (Earth radii)
        planet_mass_mearth: planet mass (Earth masses)
        stellar_radius_rsun: stellar radius (solar radii)
        equilibrium_temperature_k: equilibrium temperature (K)
        mean_molecular_weight_amu: atmospheric mean molecular weight (amu)
        n_scale_heights: number of scale heights across feature (default 5)
        photometric_precision_ppm: instrument precision per transit (ppm)
    """
    _RSUN_M = 6.957e8

    if planet_radius_rearth <= 0.0:
        return TransmissionResult(float("nan"), float("nan"), float("nan"),
                                   float("nan"), "UNKNOWN", "INVALID_RADIUS")
    if planet_mass_mearth <= 0.0:
        return TransmissionResult(float("nan"), float("nan"), float("nan"),
                                   float("nan"), "UNKNOWN", "INVALID_MASS")
    if stellar_radius_rsun <= 0.0:
        return TransmissionResult(float("nan"), float("nan"), float("nan"),
                                   float("nan"), "UNKNOWN", "INVALID_STELLAR_RADIUS")
    if equilibrium_temperature_k <= 0.0:
        return TransmissionResult(float("nan"), float("nan"), float("nan"),
                                   float("nan"), "UNKNOWN", "INVALID_TEMPERATURE")

    rp_m = planet_radius_rearth * _REARTH_M
    rs_m = stellar_radius_rsun * _RSUN_M

    # Surface gravity
    g_surface = _G_EARTH * (planet_mass_mearth) / (planet_radius_rearth**2)
    g_surface = max(g_surface, 0.1)  # lower bound for very low mass planets

    # Scale height
    mu_kg = mean_molecular_weight_amu * _AMU_KG
    h_m = _K_B * equilibrium_temperature_k / (mu_kg * g_surface)
    h_km = h_m / 1e3

    # Signal per scale height (Seager & Sasselov 2000)
    delta_per_h = 2.0 * rp_m * h_m / rs_m**2 * 1e6

    feature_ppm = n_scale_heights * delta_per_h

    if feature_ppm >= 3.0 * photometric_precision_ppm:
        detect = "DETECTABLE"
    elif feature_ppm >= photometric_precision_ppm:
        detect = "MARGINAL"
    else:
        detect = "UNDETECTABLE"

    return TransmissionResult(
        scale_height_km=h_km,
        signal_per_scale_height_ppm=delta_per_h,
        n_scale_heights=n_scale_heights,
        feature_amplitude_ppm=feature_ppm,
        detectability=detect,
        flag="OK",
    )


def format_transmission_result(r: TransmissionResult) -> str:
    if r.flag != "OK":
        return f"Transmission | flag={r.flag}"
    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| Scale height | {r.scale_height_km:.1f} km |\n"
        f"| Signal per H | {r.signal_per_scale_height_ppm:.2f} ppm |\n"
        f"| Scale heights | {r.n_scale_heights:.1f} |\n"
        f"| Feature amplitude | {r.feature_amplitude_ppm:.2f} ppm |\n"
        f"| Detectability | {r.detectability} |\n"
        f"| Flag | {r.flag} |"
    )


def _cli() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Atmospheric transmission amplitude calculator")
    p.add_argument("planet_radius_rearth", type=float)
    p.add_argument("planet_mass_mearth", type=float)
    p.add_argument("stellar_radius_rsun", type=float)
    p.add_argument("teq_k", type=float)
    p.add_argument("--mu", type=float, default=2.3)
    p.add_argument("--n-h", type=float, default=5.0)
    args = p.parse_args()
    r = compute_transmission_amplitude(args.planet_radius_rearth, args.planet_mass_mearth,
                                        args.stellar_radius_rsun, args.teq_k,
                                        mean_molecular_weight_amu=args.mu,
                                        n_scale_heights=args.n_h)
    print(format_transmission_result(r))


if __name__ == "__main__":
    _cli()
