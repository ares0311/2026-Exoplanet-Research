"""Compute RV semi-amplitude and planet mass upper limit from orbital parameters."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

# Physical constants (SI)
_G = 6.674e-11          # m^3 kg^-1 s^-2
_MSUN_KG = 1.989e30     # kg
_MEARTH_KG = 5.972e24   # kg
_MEARTH_MSUN = 3.003e-6
_AU_M = 1.496e11        # m
_DAY_S = 86400.0


@dataclass(frozen=True)
class RvMassLimitResult:
    period_days: float
    stellar_mass_msun: float
    rv_precision_ms: float
    k_amplitude_ms: float
    mass_sini_mearth: float
    mass_upper_limit_mearth: float
    flag: str


def compute_rv_mass_limit(
    period_days: float,
    stellar_mass_msun: float,
    rv_precision_ms: float,
    n_obs: int = 10,
    eccentricity: float = 0.0,
    inclination_deg: float = 90.0,
) -> RvMassLimitResult:
    """
    Compute the detectable RV semi-amplitude K and implied planet mass upper limit.

    K = (2πG/P)^(1/3) * Mp·sin(i) / ((Ms + Mp)^(2/3) * sqrt(1-e²))

    Linearized (Mp << Ms):
        K ≈ (2πG/P)^(1/3) * Mp·sin(i) / Ms^(2/3) / sqrt(1-e²)

    mass_upper_limit: smallest mass detectable at SNR=3 with n_obs observations.
    k_amplitude: K for mass_upper_limit.
    """
    for name, val in [
        ("period_days", period_days),
        ("stellar_mass_msun", stellar_mass_msun),
        ("rv_precision_ms", rv_precision_ms),
    ]:
        if not math.isfinite(val) or val <= 0.0:
            return RvMassLimitResult(
                period_days=period_days,
                stellar_mass_msun=stellar_mass_msun,
                rv_precision_ms=rv_precision_ms,
                k_amplitude_ms=float("nan"),
                mass_sini_mearth=float("nan"),
                mass_upper_limit_mearth=float("nan"),
                flag=f"INVALID_{name.upper()}",
            )
    if not math.isfinite(eccentricity) or not (0.0 <= eccentricity < 1.0):
        return RvMassLimitResult(
            period_days=period_days,
            stellar_mass_msun=stellar_mass_msun,
            rv_precision_ms=rv_precision_ms,
            k_amplitude_ms=float("nan"),
            mass_sini_mearth=float("nan"),
            mass_upper_limit_mearth=float("nan"),
            flag="INVALID_ECCENTRICITY",
        )

    period_s = period_days * _DAY_S
    ms_kg = stellar_mass_msun * _MSUN_KG
    sin_i = math.sin(math.radians(inclination_deg))

    # Effective K per unit Mp*sin(i) [m/s per kg]
    k_per_mp_sini = (
        (2.0 * math.pi * _G / period_s) ** (1.0 / 3.0)
        / ms_kg ** (2.0 / 3.0)
        / math.sqrt(1.0 - eccentricity**2)
    )

    # Detection threshold: K > 3 * rv_precision / sqrt(n_obs)
    k_threshold = 3.0 * rv_precision_ms / math.sqrt(max(n_obs, 1))

    # Mass upper limit: smallest Mp detectable
    mp_limit_kg = k_threshold / (k_per_mp_sini * sin_i) if sin_i > 0 else float("inf")
    mp_limit_mearth = mp_limit_kg / _MEARTH_KG

    # K for the limit mass
    k_amplitude = k_threshold

    # Mp*sin(i) for the limit (same as limit since we solved for it)
    mass_sini_mearth = mp_limit_mearth * sin_i

    return RvMassLimitResult(
        period_days=period_days,
        stellar_mass_msun=stellar_mass_msun,
        rv_precision_ms=rv_precision_ms,
        k_amplitude_ms=round(k_amplitude, 4),
        mass_sini_mearth=round(mass_sini_mearth, 4),
        mass_upper_limit_mearth=round(mp_limit_mearth, 4),
        flag="OK",
    )


def format_rv_mass_limit(r: RvMassLimitResult) -> str:
    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| Period (days) | {r.period_days:.4f} |\n"
        f"| Stellar mass (M☉) | {r.stellar_mass_msun:.3f} |\n"
        f"| RV precision (m/s) | {r.rv_precision_ms:.2f} |\n"
        f"| Detectable K (m/s) | {r.k_amplitude_ms:.4f} |\n"
        f"| Mp·sin(i) upper limit (M⊕) | {r.mass_sini_mearth:.4f} |\n"
        f"| Mp upper limit (M⊕) | {r.mass_upper_limit_mearth:.4f} |\n"
        f"| Flag | {r.flag} |\n"
    )


def _cli() -> int:
    p = argparse.ArgumentParser(description="Compute RV mass upper limit.")
    p.add_argument("period_days", type=float)
    p.add_argument("stellar_mass_msun", type=float)
    p.add_argument("rv_precision_ms", type=float)
    p.add_argument("--n-obs", type=int, default=10)
    p.add_argument("--eccentricity", type=float, default=0.0)
    args = p.parse_args()
    r = compute_rv_mass_limit(
        args.period_days, args.stellar_mass_msun, args.rv_precision_ms,
        n_obs=args.n_obs, eccentricity=args.eccentricity,
    )
    print(format_rv_mass_limit(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
