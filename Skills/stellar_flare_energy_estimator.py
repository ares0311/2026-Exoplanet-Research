"""Estimate stellar flare energy from amplitude, duration, and stellar luminosity."""
from __future__ import annotations

from dataclasses import dataclass

_L_SUN = 3.828e26  # W


@dataclass(frozen=True)
class FlareEnergyResult:
    equivalent_duration_s: float   # t_eq = ∫(ΔF/F) dt
    flare_energy_j: float          # E = L_star × t_eq
    flare_energy_log10: float      # log10(E / J)
    flare_class: str               # A / B / C / M / X (Hawley+2014 scale)
    flag: str


def estimate_flare_energy(
    flare_amplitude_ppm: float,
    flare_duration_hours: float,
    stellar_luminosity_lsun: float = 1.0,
    stellar_teff_k: float = 5778.0,
    shape: str = "exponential",
) -> FlareEnergyResult:
    """Estimate bolometric flare energy.

    Equivalent duration: t_eq ≈ amplitude_fraction × duration × shape_factor
    Shape factors: 'exponential' → 0.5 (area under 1-e^(-t/τ)); 'box' → 1.0

    Flare class boundaries (log10 E/J, Hawley+2014):
      A < 29, B < 30, C < 31, M < 32, X ≥ 32

    Args:
        flare_amplitude_ppm: peak flux excess above quiescent level (ppm)
        flare_duration_hours: flare duration (hours)
        stellar_luminosity_lsun: stellar bolometric luminosity (solar luminosities)
        stellar_teff_k: stellar effective temperature (K) — reserved for SED correction
        shape: 'exponential' or 'box' light curve shape
    """
    if flare_amplitude_ppm <= 0.0:
        return FlareEnergyResult(float("nan"), float("nan"),
                                  float("nan"), "UNKNOWN", "INVALID_AMPLITUDE")
    if flare_duration_hours <= 0.0:
        return FlareEnergyResult(float("nan"), float("nan"),
                                  float("nan"), "UNKNOWN", "INVALID_DURATION")
    if stellar_luminosity_lsun <= 0.0:
        return FlareEnergyResult(float("nan"), float("nan"),
                                  float("nan"), "UNKNOWN", "INVALID_LUMINOSITY")

    shape_factor = 0.5 if shape == "exponential" else 1.0
    amplitude_frac = flare_amplitude_ppm * 1e-6

    t_eq_s = amplitude_frac * flare_duration_hours * 3600.0 * shape_factor

    l_star_w = stellar_luminosity_lsun * _L_SUN
    energy_j = l_star_w * t_eq_s

    import math
    log_e = math.log10(energy_j) if energy_j > 0 else float("-inf")

    if log_e < 29.0:
        fclass = "A"
    elif log_e < 30.0:
        fclass = "B"
    elif log_e < 31.0:
        fclass = "C"
    elif log_e < 32.0:
        fclass = "M"
    else:
        fclass = "X"

    return FlareEnergyResult(
        equivalent_duration_s=t_eq_s,
        flare_energy_j=energy_j,
        flare_energy_log10=log_e,
        flare_class=fclass,
        flag="OK",
    )


def format_flare_energy_result(r: FlareEnergyResult) -> str:
    if r.flag != "OK":
        return f"FlareEnergy | flag={r.flag}"
    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| Equivalent duration | {r.equivalent_duration_s:.1f} s |\n"
        f"| Flare energy | {r.flare_energy_j:.3e} J |\n"
        f"| log₁₀(E/J) | {r.flare_energy_log10:.2f} |\n"
        f"| Flare class | {r.flare_class} |\n"
        f"| Flag | {r.flag} |"
    )


def _cli() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Stellar flare energy estimator")
    p.add_argument("amplitude_ppm", type=float)
    p.add_argument("duration_hours", type=float)
    p.add_argument("--luminosity", type=float, default=1.0)
    args = p.parse_args()
    r = estimate_flare_energy(args.amplitude_ppm, args.duration_hours,
                               stellar_luminosity_lsun=args.luminosity)
    print(format_flare_energy_result(r))


if __name__ == "__main__":
    _cli()
