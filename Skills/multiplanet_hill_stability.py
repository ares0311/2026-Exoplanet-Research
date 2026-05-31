"""Check Hill stability for pairs of planets in a multi-planet system."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class HillStabilityResult:
    period_inner_days: float
    period_outer_days: float
    mass_inner_mearth: float
    mass_outer_mearth: float
    stellar_mass_msun: float
    delta_hill: float
    delta_crit: float
    stable: bool
    flag: str


_MEARTH_MSUN = 3.003e-6


def check_hill_stability(
    period_inner_days: float,
    period_outer_days: float,
    mass_inner_mearth: float,
    mass_outer_mearth: float,
    stellar_mass_msun: float = 1.0,
) -> HillStabilityResult:
    """
    Check mutual Hill stability for two planets.

    Uses the Gladman (1993) criterion:
        Δ = (a_out - a_in) / R_H,mutual  >  2√3  ≈ 3.46

    where R_H,mutual = ((m1+m2)/(3*M*))^(1/3) * (a1+a2)/2.
    Periods converted to semi-major axes via Kepler's third law (a in AU).
    """
    for name, val in [
        ("period_inner_days", period_inner_days),
        ("period_outer_days", period_outer_days),
        ("mass_inner_mearth", mass_inner_mearth),
        ("mass_outer_mearth", mass_outer_mearth),
        ("stellar_mass_msun", stellar_mass_msun),
    ]:
        if not math.isfinite(val) or val <= 0.0:
            return HillStabilityResult(
                period_inner_days=period_inner_days,
                period_outer_days=period_outer_days,
                mass_inner_mearth=mass_inner_mearth,
                mass_outer_mearth=mass_outer_mearth,
                stellar_mass_msun=stellar_mass_msun,
                delta_hill=float("nan"),
                delta_crit=2.0 * math.sqrt(3.0),
                stable=False,
                flag=f"INVALID_{name.upper()}",
            )

    if period_inner_days >= period_outer_days:
        return HillStabilityResult(
            period_inner_days=period_inner_days,
            period_outer_days=period_outer_days,
            mass_inner_mearth=mass_inner_mearth,
            mass_outer_mearth=mass_outer_mearth,
            stellar_mass_msun=stellar_mass_msun,
            delta_hill=float("nan"),
            delta_crit=2.0 * math.sqrt(3.0),
            stable=False,
            flag="PERIOD_ORDER_ERROR",
        )

    # Kepler 3rd law: a^3 = M* * (P/yr)^2  =>  a in AU, M* in Msun
    p_in_yr = period_inner_days / 365.25
    p_out_yr = period_outer_days / 365.25
    a_in = (stellar_mass_msun * p_in_yr**2) ** (1.0 / 3.0)
    a_out = (stellar_mass_msun * p_out_yr**2) ** (1.0 / 3.0)

    m_in_msun = mass_inner_mearth * _MEARTH_MSUN
    m_out_msun = mass_outer_mearth * _MEARTH_MSUN

    rh_mutual = ((m_in_msun + m_out_msun) / (3.0 * stellar_mass_msun)) ** (1.0 / 3.0)
    rh_mutual *= (a_in + a_out) / 2.0

    delta_hill = (a_out - a_in) / rh_mutual
    delta_crit = 2.0 * math.sqrt(3.0)
    stable = delta_hill > delta_crit

    return HillStabilityResult(
        period_inner_days=period_inner_days,
        period_outer_days=period_outer_days,
        mass_inner_mearth=mass_inner_mearth,
        mass_outer_mearth=mass_outer_mearth,
        stellar_mass_msun=stellar_mass_msun,
        delta_hill=round(delta_hill, 4),
        delta_crit=round(delta_crit, 4),
        stable=stable,
        flag="OK" if stable else "HILL_UNSTABLE",
    )


def format_stability_result(r: HillStabilityResult) -> str:
    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| Period inner (days) | {r.period_inner_days:.4f} |\n"
        f"| Period outer (days) | {r.period_outer_days:.4f} |\n"
        f"| Δ_Hill | {r.delta_hill:.4f} |\n"
        f"| Δ_crit (2√3) | {r.delta_crit:.4f} |\n"
        f"| Stable | {r.stable} |\n"
        f"| Flag | {r.flag} |\n"
    )


def _cli() -> int:
    p = argparse.ArgumentParser(description="Check mutual Hill stability of two planets.")
    p.add_argument("period_inner_days", type=float)
    p.add_argument("period_outer_days", type=float)
    p.add_argument("mass_inner_mearth", type=float)
    p.add_argument("mass_outer_mearth", type=float)
    p.add_argument("--stellar-mass", type=float, default=1.0)
    args = p.parse_args()
    r = check_hill_stability(
        args.period_inner_days, args.period_outer_days,
        args.mass_inner_mearth, args.mass_outer_mearth,
        args.stellar_mass,
    )
    print(format_stability_result(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
