"""Check whether radial-velocity confirmation of a planet candidate is feasible.

Estimates the RV semi-amplitude K from planet mass, stellar mass, and period,
then computes the expected SNR given an instrument's precision and number of
observations.  Distinct from ``rv_semiamplitude_estimator`` (computes K only)
— this adds the detectability decision.

Public API
----------
RVDetectabilityResult(planet_mass_mearth, stellar_mass_msun, period_days,
                      k_ms, rv_precision_ms, snr_rv, n_obs_required,
                      is_detectable, flag)
check_rv_detectability(planet_mass_mearth, stellar_mass_msun, period_days, *,
                       rv_precision_ms, n_obs, detection_snr) -> RVDetectabilityResult
format_rv_detectability(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# Physical constants (SI)
_G_SI = 6.674e-11     # m^3 kg^-1 s^-2
_MSUN_KG = 1.989e30   # kg
_MEARTH_KG = 5.972e24 # kg
_DAYS_TO_SEC = 86400.0


@dataclass(frozen=True)
class RVDetectabilityResult:
    planet_mass_mearth: float
    stellar_mass_msun: float
    period_days: float
    k_ms: float | None              # RV semi-amplitude (m/s)
    rv_precision_ms: float
    snr_rv: float | None            # K × sqrt(N_obs) / σ_RV
    n_obs_required: int | None      # to reach detection_snr
    is_detectable: bool
    flag: str  # "OK" | "INVALID"


def check_rv_detectability(
    planet_mass_mearth: float,
    stellar_mass_msun: float,
    period_days: float,
    *,
    rv_precision_ms: float = 1.0,
    n_obs: int = 10,
    detection_snr: float = 5.0,
) -> RVDetectabilityResult:
    """Estimate RV semi-amplitude and detectability for a planet candidate.

    Assumes circular orbit, sin(i) = 1 (edge-on), Mp << M_star.
    K = (2πG/P)^(1/3) × Mp / M_star^(2/3)

    Args:
        planet_mass_mearth: Planet mass (Earth masses).
        stellar_mass_msun: Stellar mass (solar masses).
        period_days: Orbital period (days).
        rv_precision_ms: RV instrument precision (m/s, 1-σ per observation).
        n_obs: Number of RV observations.
        detection_snr: Required SNR for detection (default 5).

    Returns:
        :class:`RVDetectabilityResult`.
    """
    for val in (planet_mass_mearth, stellar_mass_msun, period_days, rv_precision_ms):
        if not math.isfinite(val) or val <= 0:
            return RVDetectabilityResult(
                planet_mass_mearth, stellar_mass_msun, period_days,
                None, rv_precision_ms, None, None, False, "INVALID"
            )
    if n_obs < 1 or detection_snr <= 0:
        return RVDetectabilityResult(
            planet_mass_mearth, stellar_mass_msun, period_days,
            None, rv_precision_ms, None, None, False, "INVALID"
        )

    mp_kg = planet_mass_mearth * _MEARTH_KG
    ms_kg = stellar_mass_msun * _MSUN_KG
    p_sec = period_days * _DAYS_TO_SEC

    k = ((2 * math.pi * _G_SI / p_sec) ** (1.0 / 3.0)
         * mp_kg / ms_kg ** (2.0 / 3.0))

    snr = k * math.sqrt(n_obs) / rv_precision_ms
    n_req = math.ceil((detection_snr * rv_precision_ms / k) ** 2) if k > 0 else None
    is_det = snr >= detection_snr

    return RVDetectabilityResult(
        planet_mass_mearth=planet_mass_mearth,
        stellar_mass_msun=stellar_mass_msun,
        period_days=period_days,
        k_ms=round(k, 4),
        rv_precision_ms=rv_precision_ms,
        snr_rv=round(snr, 3),
        n_obs_required=n_req,
        is_detectable=is_det,
        flag="OK",
    )


def format_rv_detectability(result: RVDetectabilityResult) -> str:
    """Format RV detectability result as Markdown."""
    status = "**DETECTABLE**" if result.is_detectable else "Not detectable"
    lines = [
        "## RV Detectability Checker",
        "",
        f"- Planet mass: {result.planet_mass_mearth} M_earth",
        f"- Stellar mass: {result.stellar_mass_msun} M_sun",
        f"- Period: {result.period_days} days",
        f"- **RV semi-amplitude K: {result.k_ms} m/s**",
        f"- RV precision: {result.rv_precision_ms} m/s",
        f"- SNR (N={result.snr_rv}): {result.snr_rv}",
        f"- Observations required: {result.n_obs_required}",
        f"- **Status: {status}**",
        f"- **Flag: {result.flag}**",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="rv_detectability_checker",
        description="Check RV detectability of a planet candidate.",
    )
    parser.add_argument("planet_mass_mearth", type=float)
    parser.add_argument("stellar_mass_msun", type=float)
    parser.add_argument("period_days", type=float)
    parser.add_argument("--rv-precision", type=float, default=1.0)
    parser.add_argument("--n-obs", type=int, default=10)
    args = parser.parse_args(argv)

    result = check_rv_detectability(
        args.planet_mass_mearth, args.stellar_mass_msun, args.period_days,
        rv_precision_ms=args.rv_precision, n_obs=args.n_obs,
    )
    print(format_rv_detectability(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
