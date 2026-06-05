"""Estimate projected spin-orbit obliquity λ from Rossiter-McLaughlin amplitude."""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ObliquityRMResult:
    rm_amplitude_ms: float          # expected RM amplitude (m/s)
    lambda_deg: float               # projected obliquity (degrees)
    obliquity_class: str            # ALIGNED / MISALIGNED / RETROGRADE
    impact_contribution: float      # correction factor from impact parameter
    flag: str


def compute_obliquity_from_rm(
    vsini_kms: float,
    depth_ppm: float,
    impact_parameter: float = 0.0,
    observed_rm_amplitude_ms: float | None = None,
    period_days: float | None = None,
    stellar_radius_rsun: float = 1.0,
) -> ObliquityRMResult:
    """Compute projected obliquity λ from RM amplitude.

    RM amplitude (Queloz et al. 2000 approximation):
      ΔRV_RM ≈ v·sin(i) × (Rp/Rs)² × √(1 - b²) × cos(λ)

    If observed_rm_amplitude_ms provided, invert to find λ.
    Otherwise returns expected amplitude for λ=0 (aligned case).

    Args:
        vsini_kms: stellar projected rotation velocity (km/s)
        depth_ppm: transit depth (ppm) = (Rp/Rs)²
        impact_parameter: transit impact parameter b
        observed_rm_amplitude_ms: measured RM amplitude (m/s); if given, solve for λ
        period_days: orbital period (days); optional, for velocity check
        stellar_radius_rsun: stellar radius (solar radii)
    """
    if vsini_kms <= 0.0:
        return ObliquityRMResult(float("nan"), float("nan"), "UNKNOWN", float("nan"),
                                  "INVALID_VSINI")
    if depth_ppm <= 0.0:
        return ObliquityRMResult(float("nan"), float("nan"), "UNKNOWN", float("nan"),
                                  "INVALID_DEPTH")
    if not (0.0 <= impact_parameter < 1.5):
        return ObliquityRMResult(float("nan"), float("nan"), "UNKNOWN", float("nan"),
                                  "INVALID_IMPACT")

    k_sq = depth_ppm * 1e-6   # (Rp/Rs)²
    vsini_ms = vsini_kms * 1e3

    # Impact parameter correction: chord velocity ∝ √(1-b²)
    chord_factor = math.sqrt(max(1.0 - impact_parameter**2, 0.0))

    # Expected amplitude (aligned, λ=0)
    rm_amp_ms = vsini_ms * k_sq * chord_factor

    if observed_rm_amplitude_ms is not None:
        obs = observed_rm_amplitude_ms
        if rm_amp_ms > 0:
            cos_lambda = obs / rm_amp_ms
            cos_lambda = max(min(cos_lambda, 1.0), -1.0)
            lambda_deg = math.degrees(math.acos(abs(cos_lambda)))
            # Sign: negative amplitude → retrograde
            if obs < 0:
                lambda_deg = 180.0 - lambda_deg
        else:
            lambda_deg = 0.0
    else:
        lambda_deg = 0.0  # assumed aligned

    if lambda_deg < 30.0:
        obl_class = "ALIGNED"
    elif lambda_deg < 120.0:
        obl_class = "MISALIGNED"
    else:
        obl_class = "RETROGRADE"

    return ObliquityRMResult(
        rm_amplitude_ms=rm_amp_ms,
        lambda_deg=lambda_deg,
        obliquity_class=obl_class,
        impact_contribution=chord_factor,
        flag="OK",
    )


def format_obliquity_rm_result(r: ObliquityRMResult) -> str:
    if r.flag != "OK":
        return f"ObliquityRM | flag={r.flag}"
    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| RM amplitude (aligned) | {r.rm_amplitude_ms:.2f} m/s |\n"
        f"| Projected obliquity λ | {r.lambda_deg:.1f} ° |\n"
        f"| Obliquity class | {r.obliquity_class} |\n"
        f"| Chord factor √(1-b²) | {r.impact_contribution:.3f} |\n"
        f"| Flag | {r.flag} |"
    )


def _cli() -> None:
    import argparse

    p = argparse.ArgumentParser(description="RM obliquity calculator")
    p.add_argument("vsini_kms", type=float)
    p.add_argument("depth_ppm", type=float)
    p.add_argument("--b", type=float, default=0.0)
    p.add_argument("--obs-rm-ms", type=float, default=None)
    args = p.parse_args()
    r = compute_obliquity_from_rm(args.vsini_kms, args.depth_ppm,
                                   impact_parameter=args.b,
                                   observed_rm_amplitude_ms=args.obs_rm_ms)
    print(format_obliquity_rm_result(r))


if __name__ == "__main__":
    _cli()
