"""Estimate RV orbital solution parameters from semi-amplitude and priors."""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RvOrbitalSolutionResult:
    k1_ms: float
    period_days: float
    eccentricity: float
    omega_deg: float
    gamma_ms: float
    planet_mass_sini_mearth: float
    flag: str


_G = 6.674e-11
_MSUN_KG = 1.989e30
_MJUP_KG = 1.898e27
_MEARTH_KG = 5.972e24


def compute_rv_orbital_solution(
    k1_ms: float,
    period_days: float,
    stellar_mass_msun: float = 1.0,
    eccentricity: float = 0.0,
    omega_deg: float = 90.0,
    gamma_ms: float = 0.0,
    inclination_deg: float = 90.0,
) -> RvOrbitalSolutionResult:
    """Compute planet minimum mass from RV orbital parameters.

    Uses the standard RV relation:
      K = (2πG/P)^(1/3) * Mp*sin(i) / (Ms + Mp)^(2/3) / sqrt(1 - e²)

    Solved for Mp*sin(i) iteratively (Mp << Ms approximation for small planets).

    Args:
        k1_ms: RV semi-amplitude (m/s)
        period_days: orbital period (days)
        stellar_mass_msun: stellar mass (solar masses)
        eccentricity: orbital eccentricity
        omega_deg: argument of periastron (degrees)
        gamma_ms: systemic velocity (m/s)
        inclination_deg: orbital inclination (degrees); default 90 = edge-on
    """
    if k1_ms <= 0.0:
        return RvOrbitalSolutionResult(k1_ms, period_days, eccentricity,
                                        omega_deg, gamma_ms, float("nan"), "INVALID_K1")
    if period_days <= 0.0:
        return RvOrbitalSolutionResult(k1_ms, period_days, eccentricity,
                                        omega_deg, gamma_ms, float("nan"), "INVALID_PERIOD")
    if stellar_mass_msun <= 0.0:
        return RvOrbitalSolutionResult(k1_ms, period_days, eccentricity,
                                        omega_deg, gamma_ms, float("nan"), "INVALID_STELLAR_MASS")
    if not (0.0 <= eccentricity < 1.0):
        return RvOrbitalSolutionResult(k1_ms, period_days, eccentricity,
                                        omega_deg, gamma_ms, float("nan"), "INVALID_ECCENTRICITY")

    p_s = period_days * 86400.0
    ms_kg = stellar_mass_msun * _MSUN_KG
    sin_i = math.sin(math.radians(inclination_deg))
    sin_i = max(sin_i, 1e-6)

    ecc_factor = math.sqrt(1.0 - eccentricity**2)
    # Mp*sin(i) ≈ K * (P / 2πG)^(1/3) * Ms^(2/3) * sqrt(1-e²) [small Mp approximation]
    mp_sin_i_kg = (
        k1_ms * (p_s / (2.0 * math.pi * _G)) ** (1.0 / 3.0)
        * ms_kg ** (2.0 / 3.0) * ecc_factor
    )
    mp_mearth = mp_sin_i_kg / (_MEARTH_KG * sin_i)

    return RvOrbitalSolutionResult(
        k1_ms=k1_ms,
        period_days=period_days,
        eccentricity=eccentricity,
        omega_deg=omega_deg,
        gamma_ms=gamma_ms,
        planet_mass_sini_mearth=mp_mearth,
        flag="OK",
    )


def format_rv_orbital_solution_result(r: RvOrbitalSolutionResult) -> str:
    if r.flag != "OK":
        return f"RvOrbitalSolution | flag={r.flag}"
    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| K₁ | {r.k1_ms:.2f} m/s |\n"
        f"| Period | {r.period_days:.4f} days |\n"
        f"| Eccentricity | {r.eccentricity:.4f} |\n"
        f"| ω | {r.omega_deg:.1f}° |\n"
        f"| γ | {r.gamma_ms:.2f} m/s |\n"
        f"| Mp·sin(i) | {r.planet_mass_sini_mearth:.2f} M⊕ |\n"
        f"| Flag | {r.flag} |"
    )


def _cli() -> None:
    import argparse

    p = argparse.ArgumentParser(description="RV orbital solution solver")
    p.add_argument("k1_ms", type=float, help="RV semi-amplitude (m/s)")
    p.add_argument("period_days", type=float, help="Orbital period (days)")
    p.add_argument("--mstar", type=float, default=1.0, help="Stellar mass (Msun)")
    p.add_argument("--ecc", type=float, default=0.0, help="Eccentricity")
    p.add_argument("--omega", type=float, default=90.0, help="Arg. periastron (deg)")
    args = p.parse_args()
    r = compute_rv_orbital_solution(args.k1_ms, args.period_days,
                                     stellar_mass_msun=args.mstar,
                                     eccentricity=args.ecc, omega_deg=args.omega)
    print(format_rv_orbital_solution_result(r))


if __name__ == "__main__":
    _cli()
