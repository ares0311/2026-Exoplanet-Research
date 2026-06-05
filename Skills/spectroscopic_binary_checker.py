"""Check for spectroscopic binary indicators from RV amplitude and stellar type."""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SpectroscopicBinaryResult:
    mass_function_msun: float       # f(m) = Mp³·sin³(i) / (Mp+Ms)²
    min_companion_mass_msun: float  # minimum companion mass (sin i = 1)
    rv_semiamplitude_ms: float      # K (m/s) used or implied
    sb_class: str                   # SB1 / SB2_POSSIBLE / NOT_SB / AMBIGUOUS
    stellar_mass_ratio_q: float     # q = m2/m1 (lower bound)
    flag: str


def check_spectroscopic_binary(
    rv_semiamplitude_ms: float,
    period_days: float,
    stellar_mass_msun: float = 1.0,
    stellar_teff_k: float = 5778.0,
    eccentricity: float = 0.0,
    secondary_lines_detected: bool = False,
) -> SpectroscopicBinaryResult:
    """Check for spectroscopic binary indicators.

    Mass function (binary orbit):
      f(m) = (m2 sin i)³ / (m1 + m2)² = P K³ (1-e²)^(3/2) / (2π G)

    SB1 indicator: K > stellar jitter threshold and f(m) consistent with stellar companion.
    SB2 indicator: secondary spectral lines detected, or q > 0.3.

    Args:
        rv_semiamplitude_ms: measured RV semi-amplitude K (m/s)
        period_days: orbital period (days)
        stellar_mass_msun: primary stellar mass (solar masses)
        stellar_teff_k: primary effective temperature (K)
        eccentricity: orbital eccentricity
        secondary_lines_detected: True if secondary spectral lines seen
    """
    _G = 6.674e-11
    _MSUN_KG = 1.989e30
    _MSUN = 1.0

    if rv_semiamplitude_ms <= 0.0:
        return SpectroscopicBinaryResult(float("nan"), float("nan"), rv_semiamplitude_ms,
                                          "UNKNOWN", float("nan"), "INVALID_RV")
    if period_days <= 0.0:
        return SpectroscopicBinaryResult(float("nan"), float("nan"), rv_semiamplitude_ms,
                                          "UNKNOWN", float("nan"), "INVALID_PERIOD")

    p_s = period_days * 86400.0
    k_ms = rv_semiamplitude_ms
    k_si = k_ms  # m/s

    # Mass function f(m) = P K^3 (1-e^2)^(3/2) / (2π G)  [kg]
    f_m_kg = p_s * k_si**3 * (1.0 - eccentricity**2) ** 1.5 / (2.0 * math.pi * _G)
    f_m_msun = f_m_kg / _MSUN_KG

    # Minimum companion mass (sin i = 1): solve m2³/(m1+m2)² = f(m)
    # Approximate: for m2 << m1: m2 ≈ (f(m) × m1²)^(1/3)
    # Iterative refinement
    m1 = stellar_mass_msun
    m2 = (f_m_msun * m1**2) ** (1.0 / 3.0)
    for _ in range(20):
        m2_new = (f_m_msun * (m1 + m2)**2) ** (1.0 / 3.0)
        if abs(m2_new - m2) < 1e-8:
            break
        m2 = m2_new
    m2 = max(m2, 0.0)

    # Mass ratio lower bound
    q = m2 / m1 if m1 > 0 else 0.0

    # Classification
    if secondary_lines_detected or q >= 0.3:
        sb_class = "SB2_POSSIBLE"
    elif m2 >= 0.08:  # stellar mass companion
        sb_class = "SB1"
    elif k_ms > 500.0:  # very large K but low mass function → low inclination possible
        sb_class = "AMBIGUOUS"
    else:
        sb_class = "NOT_SB"

    return SpectroscopicBinaryResult(
        mass_function_msun=f_m_msun,
        min_companion_mass_msun=m2,
        rv_semiamplitude_ms=k_ms,
        sb_class=sb_class,
        stellar_mass_ratio_q=q,
        flag="OK",
    )


def format_spectroscopic_binary_result(r: SpectroscopicBinaryResult) -> str:
    if r.flag != "OK":
        return f"SpectroscopicBinary | flag={r.flag}"
    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| Mass function f(m) | {r.mass_function_msun:.4e} M☉ |\n"
        f"| Min companion mass | {r.min_companion_mass_msun:.4f} M☉ |\n"
        f"| K amplitude | {r.rv_semiamplitude_ms:.1f} m/s |\n"
        f"| Mass ratio q (lower) | {r.stellar_mass_ratio_q:.4f} |\n"
        f"| SB class | {r.sb_class} |\n"
        f"| Flag | {r.flag} |"
    )


def _cli() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Spectroscopic binary indicator checker")
    p.add_argument("rv_semiamplitude_ms", type=float)
    p.add_argument("period_days", type=float)
    p.add_argument("--ms", type=float, default=1.0)
    p.add_argument("--ecc", type=float, default=0.0)
    p.add_argument("--sb2", action="store_true")
    args = p.parse_args()
    r = check_spectroscopic_binary(args.rv_semiamplitude_ms, args.period_days,
                                    stellar_mass_msun=args.ms, eccentricity=args.ecc,
                                    secondary_lines_detected=args.sb2)
    print(format_spectroscopic_binary_result(r))


if __name__ == "__main__":
    _cli()
