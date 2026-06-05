"""Check for co-orbital (Trojan/horseshoe) companions via expected TTV amplitude."""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CoOrbitalResult:
    expected_ttv_amplitude_min: float   # for Trojan at L4/L5
    trojan_mass_limit_mearth: float     # companion mass detectable at given TTV
    horseshoe_period_yr: float          # horseshoe libration period
    co_orbital_class: str               # TROJAN_POSSIBLE / HORSESHOE_POSSIBLE / NOT_CONSTRAINED
    flag: str


def check_co_orbital(
    period_days: float,
    planet_mass_mearth: float,
    stellar_mass_msun: float = 1.0,
    observed_ttv_amplitude_min: float | None = None,
    trojan_mass_mearth: float = 1.0,
) -> CoOrbitalResult:
    """Estimate TTV signature from co-orbital (Trojan) companion.

    For a Trojan at L4/L5, the TTV amplitude of the primary planet is:
      Δt ≈ (m_Trojan / (Mp + m_Trojan)) × P / (2π) × sqrt(3) / 2
    (Ford & Holman 2007; Laughlin & Chambers 2002)

    Horseshoe libration period (approx):
      τ_horse ≈ (Ms / (3*Mp))^(1/3) × P_orb / (2π × eps)
    where eps ≈ (Mp/3Ms)^(1/3) is the libration half-amplitude.

    Args:
        period_days: planet orbital period (days)
        planet_mass_mearth: planet mass (Earth masses)
        stellar_mass_msun: stellar mass (solar masses)
        observed_ttv_amplitude_min: observed TTV amplitude (minutes); if given,
            derives upper limit on co-orbital mass
        trojan_mass_mearth: hypothetical Trojan mass (Earth masses)
    """
    _MEARTH_MSUN = 3.003e-6

    if period_days <= 0.0:
        return CoOrbitalResult(float("nan"), float("nan"),
                                float("nan"), "UNKNOWN", "INVALID_PERIOD")
    if planet_mass_mearth <= 0.0:
        return CoOrbitalResult(float("nan"), float("nan"),
                                float("nan"), "UNKNOWN", "INVALID_MASS")

    mp_msun = planet_mass_mearth * _MEARTH_MSUN
    mt_msun = trojan_mass_mearth * _MEARTH_MSUN
    ms = stellar_mass_msun
    p_days = period_days

    # Trojan TTV (Ford & Holman 2007 eq. 2 for L4/L5 Trojan)
    ttv_days = (mt_msun / (mp_msun + mt_msun)) * p_days / (2.0 * math.pi) * math.sqrt(3.0)
    ttv_min = ttv_days * 24.0 * 60.0

    # Mass limit from observed TTV
    if observed_ttv_amplitude_min is not None and observed_ttv_amplitude_min > 0.0:
        obs_ttv_days = observed_ttv_amplitude_min / (24.0 * 60.0)
        mass_limit_msun = obs_ttv_days * (mp_msun) / (p_days / (2.0 * math.pi) * math.sqrt(3.0)
                                                        - obs_ttv_days)
        mass_limit_mearth = max(mass_limit_msun / _MEARTH_MSUN, 0.0)
    else:
        mass_limit_mearth = float("nan")

    # Horseshoe libration period
    eps = (mp_msun / (3.0 * ms)) ** (1.0 / 3.0)
    tau_horse_days = p_days * (ms / (3.0 * mp_msun)) ** (1.0 / 3.0) / (2.0 * math.pi * eps)
    tau_horse_yr = tau_horse_days / 365.25

    if ttv_min >= 1.0:
        co_class = "TROJAN_POSSIBLE"
    elif tau_horse_yr < 1000.0:
        co_class = "HORSESHOE_POSSIBLE"
    else:
        co_class = "NOT_CONSTRAINED"

    return CoOrbitalResult(
        expected_ttv_amplitude_min=ttv_min,
        trojan_mass_limit_mearth=mass_limit_mearth,
        horseshoe_period_yr=tau_horse_yr,
        co_orbital_class=co_class,
        flag="OK",
    )


def format_co_orbital_result(r: CoOrbitalResult) -> str:
    if r.flag != "OK":
        return f"CoOrbital | flag={r.flag}"
    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| Expected Trojan TTV | {r.expected_ttv_amplitude_min:.3f} min |\n"
        f"| Trojan mass limit | {r.trojan_mass_limit_mearth:.2f} M_Earth |\n"
        f"| Horseshoe period | {r.horseshoe_period_yr:.2e} yr |\n"
        f"| Co-orbital class | {r.co_orbital_class} |\n"
        f"| Flag | {r.flag} |"
    )


def _cli() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Co-orbital companion checker")
    p.add_argument("period_days", type=float)
    p.add_argument("planet_mass_mearth", type=float)
    p.add_argument("--trojan-mass", type=float, default=1.0)
    p.add_argument("--observed-ttv-min", type=float, default=None)
    args = p.parse_args()
    r = check_co_orbital(args.period_days, args.planet_mass_mearth,
                          trojan_mass_mearth=args.trojan_mass,
                          observed_ttv_amplitude_min=args.observed_ttv_min)
    print(format_co_orbital_result(r))


if __name__ == "__main__":
    _cli()
