"""Estimate radial-velocity semi-amplitude K from transit parameters.

Uses the mass-function formulation with a circular-orbit assumption.
Optionally propagates uncertainties via quadrature.

Public API
----------
RVResult(k_ms, k_err_ms, mass_companion_mjup, mass_companion_mearth,
         period_days, stellar_mass_msun, inclination_deg, flag)
estimate_rv_semiamplitude(period_days, stellar_mass_msun,
                          companion_mass_mjup, *, inclination_deg,
                          period_err_days, stellar_mass_err_msun,
                          companion_mass_err_mjup) -> RVResult
format_rv_result(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# Physical constants (SI)
_G = 6.674e-11           # m³ kg⁻¹ s⁻²
_MSUN_KG = 1.989e30      # kg
_MJUP_KG = 1.898e27      # kg
_MEARTH_KG = 5.972e24    # kg
_DAY_S = 86400.0         # s
_MJUP_TO_MEARTH = _MJUP_KG / _MEARTH_KG


@dataclass(frozen=True)
class RVResult:
    k_ms: float                     # RV semi-amplitude in m/s
    k_err_ms: float | None
    mass_companion_mjup: float
    mass_companion_mearth: float
    period_days: float
    stellar_mass_msun: float
    inclination_deg: float
    flag: str                       # "PLANET", "BROWN_DWARF", "STELLAR"


def _k_amplitude(
    period_days: float,
    m_star_kg: float,
    m_comp_kg: float,
    inc_deg: float,
) -> float:
    """Circular-orbit RV semi-amplitude in m/s."""
    P = period_days * _DAY_S
    inc_rad = math.radians(inc_deg)
    # K = (2πG/P)^(1/3) * m_c * sin(i) / (m_star + m_c)^(2/3)
    factor = (2.0 * math.pi * _G / P) ** (1.0 / 3.0)
    k = factor * m_comp_kg * math.sin(inc_rad) / (m_star_kg + m_comp_kg) ** (2.0 / 3.0)
    return k


def estimate_rv_semiamplitude(
    period_days: float,
    stellar_mass_msun: float,
    companion_mass_mjup: float,
    *,
    inclination_deg: float = 90.0,
    period_err_days: float | None = None,
    stellar_mass_err_msun: float | None = None,
    companion_mass_err_mjup: float | None = None,
) -> RVResult:
    """Estimate RV semi-amplitude for a circular orbit.

    Args:
        period_days: Orbital period in days.
        stellar_mass_msun: Stellar mass in solar masses.
        companion_mass_mjup: Companion mass in Jupiter masses.
        inclination_deg: Orbital inclination (default 90°, edge-on).
        period_err_days: 1-sigma period uncertainty.
        stellar_mass_err_msun: 1-sigma stellar mass uncertainty.
        companion_mass_err_mjup: 1-sigma companion mass uncertainty.

    Returns:
        :class:`RVResult`.
    """
    P = max(float(period_days), 1e-9)
    m_star = max(float(stellar_mass_msun), 1e-9) * _MSUN_KG
    m_comp = max(float(companion_mass_mjup), 0.0) * _MJUP_KG
    inc = float(inclination_deg)

    k = _k_amplitude(P, m_star, m_comp, inc)

    k_err: float | None = None
    errs_provided = [period_err_days, stellar_mass_err_msun, companion_mass_err_mjup]
    if any(e is not None for e in errs_provided):
        # Numerical partial derivatives
        dp = P * 1e-4
        dm_s = m_star * 1e-4
        dm_c = max(m_comp * 1e-4, 1e3)

        terms: list[float] = []
        if period_err_days is not None:
            dk_dP = (_k_amplitude(P + dp, m_star, m_comp, inc) - k) / dp
            terms.append((dk_dP * float(period_err_days) * _DAY_S) ** 2)
        if stellar_mass_err_msun is not None:
            dk_dMs = (_k_amplitude(P, m_star + dm_s, m_comp, inc) - k) / dm_s
            terms.append((dk_dMs * float(stellar_mass_err_msun) * _MSUN_KG) ** 2)
        if companion_mass_err_mjup is not None:
            dk_dMc = (_k_amplitude(P, m_star, m_comp + dm_c, inc) - k) / dm_c
            terms.append((dk_dMc * float(companion_mass_err_mjup) * _MJUP_KG) ** 2)
        if terms:
            k_err = math.sqrt(sum(terms))

    m_comp_mjup = companion_mass_mjup
    m_comp_mearth = m_comp_mjup * _MJUP_TO_MEARTH

    if m_comp_mjup < 13.0:
        flag = "PLANET"
    elif m_comp_mjup < 80.0:
        flag = "BROWN_DWARF"
    else:
        flag = "STELLAR"

    return RVResult(
        k_ms=round(k, 4),
        k_err_ms=round(k_err, 4) if k_err is not None else None,
        mass_companion_mjup=companion_mass_mjup,
        mass_companion_mearth=round(m_comp_mearth, 2),
        period_days=period_days,
        stellar_mass_msun=stellar_mass_msun,
        inclination_deg=inclination_deg,
        flag=flag,
    )


def format_rv_result(result: RVResult) -> str:
    """Format RV semi-amplitude result as Markdown."""
    err_str = f" ± {result.k_err_ms:.2f}" if result.k_err_ms is not None else ""
    lines = [
        "## RV Semi-Amplitude Estimate",
        "",
        f"- Period: {result.period_days:.4f} d",
        f"- Stellar mass: {result.stellar_mass_msun:.3f} M☉",
        f"- Companion mass: {result.mass_companion_mjup:.3f} M♃ "
        f"({result.mass_companion_mearth:.1f} M⊕)",
        f"- Inclination: {result.inclination_deg:.1f}°",
        f"- **K = {result.k_ms:.2f}{err_str} m/s**",
        f"- Classification: **{result.flag}**",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="rv_semiamplitude_estimator",
        description="Estimate RV semi-amplitude from transit parameters.",
    )
    parser.add_argument("period_days", type=float)
    parser.add_argument("stellar_mass_msun", type=float)
    parser.add_argument("companion_mass_mjup", type=float)
    parser.add_argument("--inclination", type=float, default=90.0)
    args = parser.parse_args(argv)

    result = estimate_rv_semiamplitude(
        args.period_days, args.stellar_mass_msun, args.companion_mass_mjup,
        inclination_deg=args.inclination,
    )
    print(format_rv_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
