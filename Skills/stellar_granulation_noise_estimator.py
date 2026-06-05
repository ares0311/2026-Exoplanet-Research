"""Estimate stellar granulation and oscillation noise for photometry."""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class GranulationNoiseResult:
    nu_max_uhz: float             # asteroseismic frequency of maximum power (µHz)
    tau_gran_min: float           # granulation timescale (minutes)
    sigma_gran_ppm: float         # granulation noise per cadence (ppm)
    sigma_osc_ppm: float          # oscillation noise per cadence (ppm)
    sigma_total_ppm: float        # total stellar noise (ppm)
    flag: str


def estimate_granulation_noise(
    stellar_teff_k: float,
    stellar_logg: float,
    stellar_radius_rsun: float = 1.0,
    cadence_min: float = 2.0,
) -> GranulationNoiseResult:
    """Estimate stellar granulation + p-mode oscillation noise.

    Asteroseismic scaling relations (Kjeldsen & Bedding 1995; Chaplin+2011):
      ν_max = ν_max_sun × (g/g_sun) × (Teff/Teff_sun)^(-0.5)
      τ_gran ≈ 3.0 × (g/g_sun)^(-0.5) × (Teff/Teff_sun)^(-0.25) minutes  (Kallinger+2014)

    Granulation amplitude (Harvey model approximation):
      σ_gran ≈ a_gran × sqrt(τ_gran / cadence)  where a_gran ∝ g^(-0.5)

    Args:
        stellar_teff_k: effective temperature (K)
        stellar_logg: surface gravity (log10(g/cm/s²))
        stellar_radius_rsun: stellar radius (solar radii)
        cadence_min: photometric cadence (minutes)
    """
    if stellar_teff_k <= 0.0:
        return GranulationNoiseResult(float("nan"), float("nan"), float("nan"),
                                       float("nan"), float("nan"), "INVALID_TEFF")
    if stellar_logg < 0.0 or stellar_logg > 6.0:
        return GranulationNoiseResult(float("nan"), float("nan"), float("nan"),
                                       float("nan"), float("nan"), "INVALID_LOGG")
    if cadence_min <= 0.0:
        return GranulationNoiseResult(float("nan"), float("nan"), float("nan"),
                                       float("nan"), float("nan"), "INVALID_CADENCE")

    _TEFF_SUN = 5778.0
    _LOGG_SUN = 4.438
    _NU_MAX_SUN = 3090.0  # µHz

    g_ratio = 10.0 ** (stellar_logg - _LOGG_SUN)
    teff_ratio = stellar_teff_k / _TEFF_SUN

    nu_max = _NU_MAX_SUN * g_ratio * teff_ratio ** (-0.5)

    # Granulation timescale (Kallinger et al. 2014)
    tau_gran_min = 3.0 * g_ratio ** (-0.5) * teff_ratio ** (-0.25)

    # Granulation amplitude (Bastien+2013 for solar-like stars)
    # a_gran ~ 0.1 * (L/Lsun)^0.838 * (M/Msun)^(-0.41) ppm  (rough scaling)
    # Using simpler g-scaling: a_gran ~ a_sun / sqrt(g/g_sun)
    a_gran_sun_ppm = 15.0  # solar granulation amplitude ~15 ppm in TESS 2-min
    a_gran_ppm = a_gran_sun_ppm / math.sqrt(g_ratio)

    # Noise contribution at given cadence from Harvey profile
    # Power density at ν_cadence = 1/(2*cadence): P_gran = 2*a²*τ / (1 + (ν*τ)²)
    nu_cadence_uhz = 1.0 / (cadence_min * 60.0) * 1e6
    tau_gran_s = tau_gran_min * 60.0
    tau_gran_uhz = tau_gran_s / 1e6  # sec → 1/µHz conversion
    power_gran = (2.0 * a_gran_ppm**2 * tau_gran_s /
                  (1.0 + (nu_cadence_uhz * tau_gran_uhz)**2))
    sigma_gran_ppm = math.sqrt(max(power_gran * nu_cadence_uhz / 1e6, 0.0))
    # Simplified: direct noise estimate
    sigma_gran_ppm = a_gran_ppm * math.sqrt(
        min(tau_gran_min / cadence_min, 1.0)
    )

    # Oscillation noise: amplitude ~ 3 ppm for solar, scales with L/Lsun
    a_osc_sun_ppm = 3.0
    a_osc_ppm = a_osc_sun_ppm / g_ratio ** 0.5 * teff_ratio ** 2.0
    # Only contributes if cadence resolves oscillations
    sigma_osc_ppm = a_osc_ppm if cadence_min < 60.0 / nu_max * 1e6 / 60.0 else 0.0

    sigma_total = math.sqrt(sigma_gran_ppm**2 + sigma_osc_ppm**2)

    return GranulationNoiseResult(
        nu_max_uhz=nu_max,
        tau_gran_min=tau_gran_min,
        sigma_gran_ppm=sigma_gran_ppm,
        sigma_osc_ppm=sigma_osc_ppm,
        sigma_total_ppm=sigma_total,
        flag="OK",
    )


def format_granulation_noise_result(r: GranulationNoiseResult) -> str:
    if r.flag != "OK":
        return f"GranulationNoise | flag={r.flag}"
    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| ν_max | {r.nu_max_uhz:.1f} µHz |\n"
        f"| τ_gran | {r.tau_gran_min:.1f} min |\n"
        f"| σ_gran | {r.sigma_gran_ppm:.2f} ppm |\n"
        f"| σ_osc | {r.sigma_osc_ppm:.2f} ppm |\n"
        f"| σ_total | {r.sigma_total_ppm:.2f} ppm |\n"
        f"| Flag | {r.flag} |"
    )


def _cli() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Stellar granulation noise estimator")
    p.add_argument("teff_k", type=float)
    p.add_argument("logg", type=float)
    p.add_argument("--cadence-min", type=float, default=2.0)
    args = p.parse_args()
    r = estimate_granulation_noise(args.teff_k, args.logg, cadence_min=args.cadence_min)
    print(format_granulation_noise_result(r))


if __name__ == "__main__":
    _cli()
