"""Estimate planet-planet TTV amplitude near mean-motion resonance."""
from __future__ import annotations

import math
from dataclasses import dataclass

_G = 6.674e-11
_MSUN_KG = 1.989e30
_MEARTH_KG = 5.972e24


@dataclass(frozen=True)
class TTVAmplitudeResult:
    ttv_amplitude_minutes: float    # TTV semi-amplitude
    libration_period_days: float    # TTV libration period
    proximity_to_resonance: float   # |Δ| = |j*P2/P1 - (j-1)| distance from MMR
    resonance_order: int            # order j of nearest resonance
    ttv_class: str                  # STRONG / MODERATE / WEAK
    flag: str


def compute_ttv_amplitude(
    inner_period_days: float,
    outer_period_days: float,
    perturber_mass_mearth: float,
    stellar_mass_msun: float = 1.0,
    inner_eccentricity: float = 0.0,
    outer_eccentricity: float = 0.0,
) -> TTVAmplitudeResult:
    """Estimate first-order near-resonance TTV amplitude (Lithwick, Xie & Wu 2012).

    For j:(j-1) near-resonance, the TTV amplitude of the inner planet:
      δt ≈ (P_inner / π) × (m_pert / M★) / |Δ|
    where Δ = j × P_outer / P_inner - (j-1) - 1 (fractional distance from exact resonance).

    The libration period:
      τ_lib ≈ P_inner / |Δ| × (M★ / m_pert)^(1/2) × (...)

    Args:
        inner_period_days: inner planet orbital period (days)
        outer_period_days: outer planet orbital period (days)
        perturber_mass_mearth: perturbing (outer) planet mass (Earth masses)
        stellar_mass_msun: stellar mass (solar masses)
        inner_eccentricity: inner planet eccentricity
        outer_eccentricity: outer planet eccentricity
    """
    if inner_period_days <= 0.0 or outer_period_days <= 0.0:
        return TTVAmplitudeResult(float("nan"), float("nan"), float("nan"),
                                   0, "UNKNOWN", "INVALID_PERIOD")
    if perturber_mass_mearth <= 0.0:
        return TTVAmplitudeResult(float("nan"), float("nan"), float("nan"),
                                   0, "UNKNOWN", "INVALID_MASS")
    if outer_period_days <= inner_period_days:
        return TTVAmplitudeResult(float("nan"), float("nan"), float("nan"),
                                   0, "UNKNOWN", "INVALID_ORDERING")

    ms_kg = stellar_mass_msun * _MSUN_KG
    mp_kg = perturber_mass_mearth * _MEARTH_KG
    mass_ratio = mp_kg / ms_kg

    period_ratio = outer_period_days / inner_period_days

    # Find nearest first-order resonance j:(j-1)
    best_j = 2
    best_delta = float("inf")
    for j in range(2, 20):
        # j:j-1 resonance ratio = j/(j-1)
        resonant_ratio = j / (j - 1)
        delta = abs(period_ratio - resonant_ratio) / resonant_ratio
        if delta < best_delta:
            best_delta = delta
            best_j = j

    j = best_j
    resonant_ratio = j / (j - 1)
    # Linear distance from resonance (Lithwick+2012 Δ definition)
    delta = period_ratio / resonant_ratio - 1.0

    # Avoid division by zero if exactly on resonance
    abs_delta = max(abs(delta), 1e-6)

    # TTV amplitude (Lithwick+2012 eq. 1, simplified)
    ttv_days = inner_period_days / math.pi * mass_ratio / abs_delta
    ttv_min = ttv_days * 24.0 * 60.0

    # Libration period
    tau_lib_days = inner_period_days / abs_delta * (ms_kg / mp_kg) ** 0.5 * mass_ratio

    if ttv_min >= 30.0:
        ttv_class = "STRONG"
    elif ttv_min >= 5.0:
        ttv_class = "MODERATE"
    else:
        ttv_class = "WEAK"

    return TTVAmplitudeResult(
        ttv_amplitude_minutes=ttv_min,
        libration_period_days=tau_lib_days,
        proximity_to_resonance=abs_delta,
        resonance_order=j,
        ttv_class=ttv_class,
        flag="OK",
    )


def format_ttv_amplitude_result(r: TTVAmplitudeResult) -> str:
    if r.flag != "OK":
        return f"TTVAmplitude | flag={r.flag}"
    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| TTV amplitude | {r.ttv_amplitude_minutes:.2f} min |\n"
        f"| Libration period | {r.libration_period_days:.1f} d |\n"
        f"| Nearest MMR | {r.resonance_order}:{r.resonance_order-1} |\n"
        f"| |Δ| from resonance | {r.proximity_to_resonance:.4f} |\n"
        f"| TTV class | {r.ttv_class} |\n"
        f"| Flag | {r.flag} |"
    )


def _cli() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Planet-planet TTV amplitude estimator")
    p.add_argument("inner_period_days", type=float)
    p.add_argument("outer_period_days", type=float)
    p.add_argument("perturber_mass_mearth", type=float)
    p.add_argument("--ms", type=float, default=1.0)
    args = p.parse_args()
    r = compute_ttv_amplitude(args.inner_period_days, args.outer_period_days,
                               args.perturber_mass_mearth, stellar_mass_msun=args.ms)
    print(format_ttv_amplitude_result(r))


if __name__ == "__main__":
    _cli()
