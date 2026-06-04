"""Compute the required RV precision and number of observations to detect a planet."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

_G = 6.674e-11
_M_SUN_KG = 1.989e30
_M_EARTH_KG = 5.972e24
_M_JUP_KG = 1.898e27


@dataclass(frozen=True)
class RvPrecisionResult:
    period_days: float
    planet_mass_mearth: float
    stellar_mass_msun: float
    eccentricity: float
    k_amplitude_ms: float           # RV semi-amplitude in m/s
    required_precision_ms: float    # single-measurement precision for 5σ detection
    n_observations_for_5sigma: int  # assuming sqrt(N) improvement
    detectable_harps: bool          # HARPS ~1 m/s
    detectable_espresso: bool       # ESPRESSO ~0.1 m/s
    detectable_pfs: bool            # PFS/NEID ~0.5 m/s
    flag: str


def compute_rv_precision_requirement(
    period_days: float,
    planet_mass_mearth: float,
    stellar_mass_msun: float = 1.0,
    eccentricity: float = 0.0,
    sin_i: float = 1.0,
    target_snr: float = 5.0,
    n_observations: int = 20,
) -> RvPrecisionResult:
    """
    Compute RV semi-amplitude K and required per-measurement precision.

    K = (2πG)^(1/3) * Mp*sin(i) / ((Mp+Ms)^(2/3) * P^(1/3)) / sqrt(1 − e²)

    Required precision for target_snr sigma detection with n_observations:
      σ_req = K / (target_snr / sqrt(n_observations))

    Parameters
    ----------
    period_days:         Orbital period in days.
    planet_mass_mearth:  Planet mass in Earth masses.
    stellar_mass_msun:   Stellar mass in solar masses.
    eccentricity:        Orbital eccentricity.
    sin_i:               sin(inclination) (default 1.0 = edge-on).
    target_snr:          Required detection S/N (default 5).
    n_observations:      Number of observations (default 20).
    """
    if not math.isfinite(period_days) or period_days <= 0:
        return RvPrecisionResult(period_days, planet_mass_mearth, stellar_mass_msun,
                                 eccentricity, float("nan"), float("nan"), 0,
                                 False, False, False, "INVALID_PERIOD")
    if not math.isfinite(planet_mass_mearth) or planet_mass_mearth <= 0:
        return RvPrecisionResult(period_days, planet_mass_mearth, stellar_mass_msun,
                                 eccentricity, float("nan"), float("nan"), 0,
                                 False, False, False, "INVALID_MASS")
    if not math.isfinite(eccentricity) or eccentricity < 0 or eccentricity >= 1:
        return RvPrecisionResult(period_days, planet_mass_mearth, stellar_mass_msun,
                                 eccentricity, float("nan"), float("nan"), 0,
                                 False, False, False, "INVALID_ECCENTRICITY")

    mp_kg = planet_mass_mearth * _M_EARTH_KG
    ms_kg = stellar_mass_msun * _M_SUN_KG
    period_s = period_days * 86400.0

    prefactor = (2.0 * math.pi * _G) ** (1.0 / 3.0)
    k_ms = (
        prefactor
        * mp_kg * sin_i
        / ((mp_kg + ms_kg) ** (2.0 / 3.0))
        / period_s ** (1.0 / 3.0)
        / math.sqrt(1.0 - eccentricity ** 2)
    )

    required_precision = k_ms * math.sqrt(n_observations) / target_snr

    n_for_5sigma = max(1, math.ceil((target_snr * required_precision / k_ms) ** 2))

    return RvPrecisionResult(
        period_days=period_days,
        planet_mass_mearth=planet_mass_mearth,
        stellar_mass_msun=stellar_mass_msun,
        eccentricity=eccentricity,
        k_amplitude_ms=round(k_ms, 6),
        required_precision_ms=round(required_precision, 6),
        n_observations_for_5sigma=n_for_5sigma,
        detectable_harps=k_ms >= 1.0,
        detectable_espresso=k_ms >= 0.1,
        detectable_pfs=k_ms >= 0.5,
        flag="OK",
    )


def format_rv_precision_result(r: RvPrecisionResult) -> str:
    def _f(v: float, fmt: str = ".6f") -> str:
        return format(v, fmt) if math.isfinite(v) else "N/A"

    return (
        f"| Parameter | Value |\n|---|---|\n"
        f"| K amplitude (m/s) | {_f(r.k_amplitude_ms, '.4f')} |\n"
        f"| Required precision (m/s) | {_f(r.required_precision_ms, '.4f')} |\n"
        f"| N obs for 5σ | {r.n_observations_for_5sigma} |\n"
        f"| Detectable (HARPS ~1 m/s) | {r.detectable_harps} |\n"
        f"| Detectable (ESPRESSO ~0.1 m/s) | {r.detectable_espresso} |\n"
        f"| Detectable (PFS/NEID ~0.5 m/s) | {r.detectable_pfs} |\n"
        f"| Flag | {r.flag} |\n"
    )


def _cli() -> int:
    p = argparse.ArgumentParser(description="Compute required RV precision for planet detection.")
    p.add_argument("period_days", type=float)
    p.add_argument("planet_mass_mearth", type=float)
    p.add_argument("--stellar-mass-msun", type=float, default=1.0)
    p.add_argument("--eccentricity", type=float, default=0.0)
    p.add_argument("--sin-i", type=float, default=1.0)
    p.add_argument("--target-snr", type=float, default=5.0)
    p.add_argument("--n-observations", type=int, default=20)
    args = p.parse_args()
    r = compute_rv_precision_requirement(
        args.period_days, args.planet_mass_mearth,
        stellar_mass_msun=args.stellar_mass_msun,
        eccentricity=args.eccentricity,
        sin_i=args.sin_i,
        target_snr=args.target_snr,
        n_observations=args.n_observations,
    )
    print(format_rv_precision_result(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
