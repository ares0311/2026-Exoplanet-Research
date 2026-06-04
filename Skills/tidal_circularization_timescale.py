"""Estimate tidal circularization timescale for hot Jupiter systems."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

# Zahn (1977) / Jackson et al. (2008) prescription
# τ_circ = (4/63) * (M★/Mp) * (a/Rp)^5 * Qp / (n)   [simplified]
# where n = 2π/P, Qp = planetary tidal quality factor
# More useful: τ_circ ≈ (Q★' / n) * (M★/Mp) * (a/R★)^5   (stellar tide dominance)

_MSUN_KG = 1.989e30
_MJUP_KG = 1.898e27
_RSUN_M = 6.957e8
_RJUP_M = 7.149e7
_AU_M = 1.495978707e11
_YR_S = 3.156e7


@dataclass(frozen=True)
class CircularizationResult:
    period_days: float
    eccentricity: float
    timescale_gyr: float        # tidal circularization timescale
    currently_circular: bool    # e < 0.05
    expected_circular: bool     # timescale < stellar age
    dominant_tide: str          # STELLAR / PLANETARY
    flag: str


def compute_circularization_timescale(
    period_days: float,
    mass_star_msun: float = 1.0,
    mass_planet_mjup: float = 1.0,
    radius_star_rsun: float = 1.0,
    radius_planet_rjup: float = 1.0,
    eccentricity: float = 0.0,
    q_star: float = 1e6,
    q_planet: float = 1e5,
    stellar_age_gyr: float = 5.0,
) -> CircularizationResult:
    """
    Estimate tidal circularization timescale.

    Uses Jackson et al. (2008) formulation:
    1/τ = (63/4) * sqrt(G M★³/a^13) * (Rp^5/Q'p) / (Mp * M★^0.5)
        + (63/4) * sqrt(G M★³/a^13) * (R★^5/Q'★) / (M★ * Mp^0.5)   [stellar tide]

    Returns timescale in Gyr for the dominant tidal channel.
    """
    if not math.isfinite(period_days) or period_days <= 0:
        return CircularizationResult(
            period_days=period_days, eccentricity=eccentricity,
            timescale_gyr=float("nan"), currently_circular=False,
            expected_circular=False, dominant_tide="UNKNOWN", flag="INVALID_PERIOD",
        )
    if not math.isfinite(mass_star_msun) or mass_star_msun <= 0:
        return CircularizationResult(
            period_days=period_days, eccentricity=eccentricity,
            timescale_gyr=float("nan"), currently_circular=False,
            expected_circular=False, dominant_tide="UNKNOWN", flag="INVALID_MASS",
        )

    # Semi-major axis from Kepler's 3rd law (in AU, 1 Msun assumed)
    p_yr = period_days / 365.25
    a_au = (mass_star_msun * p_yr**2) ** (1.0 / 3.0)
    a_m = a_au * _AU_M

    ms = mass_star_msun * _MSUN_KG
    mp = mass_planet_mjup * _MJUP_KG
    rs = radius_star_rsun * _RSUN_M
    rp = radius_planet_rjup * _RJUP_M
    n = 2.0 * math.pi / (period_days * 86400.0)

    # Planetary tide contribution: τ_p ~ (4/63) * Qp * (a/Rp)^5 / n * (Mp/M★)
    tau_p_s = (4.0 / 63.0) * q_planet * (a_m / rp) ** 5 / n * (mp / ms)

    # Stellar tide contribution: τ_★ ~ (4/63) * Q★ * (a/R★)^5 / n * (M★/Mp)
    tau_s_s = (4.0 / 63.0) * q_star * (a_m / rs) ** 5 / n * (ms / mp)

    # Dominant channel (shorter timescale)
    if tau_p_s <= tau_s_s:
        tau_s = tau_p_s
        dominant = "PLANETARY"
    else:
        tau_s = tau_s_s
        dominant = "STELLAR"

    tau_gyr = tau_s / (_YR_S * 1e9)

    return CircularizationResult(
        period_days=period_days,
        eccentricity=eccentricity,
        timescale_gyr=round(tau_gyr, 4),
        currently_circular=eccentricity < 0.05,
        expected_circular=tau_gyr < stellar_age_gyr,
        dominant_tide=dominant,
        flag="OK",
    )


def format_circularization_result(r: CircularizationResult) -> str:
    tau_str = f"{r.timescale_gyr:.4f}" if math.isfinite(r.timescale_gyr) else "N/A"
    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| Period (days) | {r.period_days:.4f} |\n"
        f"| Eccentricity | {r.eccentricity:.4f} |\n"
        f"| Circularization timescale (Gyr) | {tau_str} |\n"
        f"| Currently circular (e < 0.05) | {r.currently_circular} |\n"
        f"| Expected circular (<stellar age) | {r.expected_circular} |\n"
        f"| Dominant tide | {r.dominant_tide} |\n"
        f"| Flag | {r.flag} |\n"
    )


def _cli() -> int:
    p = argparse.ArgumentParser(description="Estimate tidal circularization timescale.")
    p.add_argument("period_days", type=float)
    p.add_argument("--mass-star-msun", type=float, default=1.0)
    p.add_argument("--mass-planet-mjup", type=float, default=1.0)
    p.add_argument("--radius-star-rsun", type=float, default=1.0)
    p.add_argument("--radius-planet-rjup", type=float, default=1.0)
    p.add_argument("--eccentricity", type=float, default=0.0)
    p.add_argument("--q-star", type=float, default=1e6)
    p.add_argument("--q-planet", type=float, default=1e5)
    args = p.parse_args()
    r = compute_circularization_timescale(
        args.period_days, args.mass_star_msun, args.mass_planet_mjup,
        args.radius_star_rsun, args.radius_planet_rjup, args.eccentricity,
        args.q_star, args.q_planet,
    )
    print(format_circularization_result(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
