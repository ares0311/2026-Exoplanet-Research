"""Check Hill stability for adjacent planet pairs in a multi-planet system."""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class HillStabilityResult:
    mutual_hill_radius_au: float
    separation_hill_radii: float   # (a2 - a1) / R_H_mutual
    stability_flag: str            # STABLE / MARGINAL / UNSTABLE
    flag: str


def check_hill_stability(
    stellar_mass_msun: float,
    inner_mass_mearth: float,
    inner_sma_au: float,
    outer_mass_mearth: float,
    outer_sma_au: float,
    inner_eccentricity: float = 0.0,
    outer_eccentricity: float = 0.0,
) -> HillStabilityResult:
    """Check Hill stability for two adjacent planets.

    Mutual Hill radius: R_H = ((m1+m2)/(3*Ms))^(1/3) * (a1+a2)/2
    Stability criterion (Gladman 1993): Δ = (a2-a1)/R_H > 2√3 ≈ 3.46

    Args:
        stellar_mass_msun: stellar mass (solar masses)
        inner_mass_mearth: inner planet mass (Earth masses)
        inner_sma_au: inner planet semi-major axis (AU)
        outer_mass_mearth: outer planet mass (Earth masses)
        outer_sma_au: outer planet semi-major axis (AU)
        inner_eccentricity: inner planet eccentricity
        outer_eccentricity: outer planet eccentricity
    """
    _MEARTH_MSUN = 3.003e-6

    if stellar_mass_msun <= 0.0:
        return HillStabilityResult(float("nan"), float("nan"), "UNKNOWN",
                                    "INVALID_STELLAR_MASS")
    if inner_mass_mearth <= 0.0 or outer_mass_mearth <= 0.0:
        return HillStabilityResult(float("nan"), float("nan"), "UNKNOWN",
                                    "INVALID_PLANET_MASS")
    if inner_sma_au <= 0.0 or outer_sma_au <= 0.0:
        return HillStabilityResult(float("nan"), float("nan"), "UNKNOWN",
                                    "INVALID_SMA")
    if outer_sma_au <= inner_sma_au:
        return HillStabilityResult(float("nan"), float("nan"), "UNKNOWN",
                                    "INVALID_ORDERING")

    m1 = inner_mass_mearth * _MEARTH_MSUN
    m2 = outer_mass_mearth * _MEARTH_MSUN
    ms = stellar_mass_msun

    a_mean = (inner_sma_au + outer_sma_au) / 2.0
    r_h = a_mean * ((m1 + m2) / (3.0 * ms)) ** (1.0 / 3.0)

    delta = (outer_sma_au - inner_sma_au) / r_h

    # Eccentric correction: effective separation reduced by e terms
    ecc_correction = 1.0 - (inner_eccentricity + outer_eccentricity)
    delta_eff = delta * max(ecc_correction, 0.01)

    critical = 2.0 * math.sqrt(3.0)  # ≈ 3.464
    if delta_eff >= critical * 1.5:
        stability = "STABLE"
    elif delta_eff >= critical:
        stability = "MARGINAL"
    else:
        stability = "UNSTABLE"

    return HillStabilityResult(
        mutual_hill_radius_au=r_h,
        separation_hill_radii=delta,
        stability_flag=stability,
        flag="OK",
    )


def format_hill_stability_result(r: HillStabilityResult) -> str:
    if r.flag != "OK":
        return f"HillStability | flag={r.flag}"
    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| Mutual Hill radius | {r.mutual_hill_radius_au:.5f} AU |\n"
        f"| Separation (Hill radii) | {r.separation_hill_radii:.2f} |\n"
        f"| Stability | {r.stability_flag} |\n"
        f"| Flag | {r.flag} |"
    )


def _cli() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Hill stability checker for adjacent planets")
    p.add_argument("stellar_mass_msun", type=float)
    p.add_argument("inner_mass_mearth", type=float)
    p.add_argument("inner_sma_au", type=float)
    p.add_argument("outer_mass_mearth", type=float)
    p.add_argument("outer_sma_au", type=float)
    args = p.parse_args()
    r = check_hill_stability(args.stellar_mass_msun, args.inner_mass_mearth,
                              args.inner_sma_au, args.outer_mass_mearth, args.outer_sma_au)
    print(format_hill_stability_result(r))


if __name__ == "__main__":
    _cli()
