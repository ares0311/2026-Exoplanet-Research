"""Estimate Kozai-Lidov cycle parameters for a hierarchical triple system."""
from __future__ import annotations

import math
from dataclasses import dataclass

_SEC_PER_DAY = 86400.0
_G = 6.674e-11
_MSUN_KG = 1.989e30


@dataclass(frozen=True)
class KozaiLidovResult:
    mutual_inclination_deg: float
    max_eccentricity: float
    kozai_cycle_period_yr: float
    is_above_critical_angle: bool
    critical_inclination_deg: float
    flag: str


def compute_kozai_lidov(
    inner_period_days: float,
    outer_period_days: float,
    mutual_inclination_deg: float,
    inner_mass_msun: float = 1.0,
    outer_mass_msun: float = 0.5,
    outer_eccentricity: float = 0.0,
    inner_eccentricity: float = 0.0,
) -> KozaiLidovResult:
    """Compute Kozai-Lidov oscillation parameters for a hierarchical triple.

    KL cycle period (Kiseleva et al. 1998 / Lithwick & Naoz 2011):
      P_KL ≈ (P_out² / P_in) × (M_in + M_out) / M_out × (1 - e_out²)^(3/2)

    Maximum eccentricity (quadrupole, test particle):
      e_max = sqrt(1 - (5/3) × cos²(i_mutual))  [valid for i > i_crit]

    Critical inclination: i_crit = arccos(sqrt(3/5)) ≈ 39.23°

    Args:
        inner_period_days: period of inner binary/planet (days)
        outer_period_days: period of outer companion (days)
        mutual_inclination_deg: mutual inclination between inner and outer orbits (degrees)
        inner_mass_msun: total mass of inner system (solar masses)
        outer_mass_msun: mass of outer companion (solar masses)
        outer_eccentricity: eccentricity of outer orbit
        inner_eccentricity: initial eccentricity of inner orbit
    """
    if inner_period_days <= 0.0:
        return KozaiLidovResult(mutual_inclination_deg, float("nan"),
                                 float("nan"), False, 39.23, "INVALID_INNER_PERIOD")
    if outer_period_days <= inner_period_days:
        return KozaiLidovResult(mutual_inclination_deg, float("nan"),
                                 float("nan"), False, 39.23, "OUTER_NOT_LONGER")
    if not (0.0 <= mutual_inclination_deg <= 180.0):
        return KozaiLidovResult(mutual_inclination_deg, float("nan"),
                                 float("nan"), False, 39.23, "INVALID_INCLINATION")
    if outer_mass_msun <= 0.0:
        return KozaiLidovResult(mutual_inclination_deg, float("nan"),
                                 float("nan"), False, 39.23, "INVALID_MASS")

    i_rad = math.radians(mutual_inclination_deg)
    cos_i = math.cos(i_rad)
    i_crit_deg = math.degrees(math.acos(math.sqrt(3.0 / 5.0)))
    is_above = abs(mutual_inclination_deg - 90.0) < (90.0 - i_crit_deg)

    # Maximum eccentricity (quadrupole test-particle approximation)
    cos2_i = cos_i**2
    arg = 1.0 - (5.0 / 3.0) * cos2_i
    e_max = inner_eccentricity if arg <= 0.0 else min(math.sqrt(arg), 0.9999)

    p_in_yr = inner_period_days / 365.25
    p_out_yr = outer_period_days / 365.25
    total_mass = inner_mass_msun + outer_mass_msun

    p_kl = (p_out_yr**2 / p_in_yr) * (total_mass / outer_mass_msun) * (
        1.0 - outer_eccentricity**2
    ) ** 1.5

    return KozaiLidovResult(
        mutual_inclination_deg=mutual_inclination_deg,
        max_eccentricity=e_max,
        kozai_cycle_period_yr=p_kl,
        is_above_critical_angle=is_above,
        critical_inclination_deg=i_crit_deg,
        flag="OK",
    )


def format_kozai_lidov_result(r: KozaiLidovResult) -> str:
    if r.flag != "OK":
        return f"KozaiLidov | flag={r.flag}"
    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| Mutual inclination | {r.mutual_inclination_deg:.1f}° |\n"
        f"| Critical inclination | {r.critical_inclination_deg:.2f}° |\n"
        f"| Above critical angle | {'YES' if r.is_above_critical_angle else 'no'} |\n"
        f"| Max eccentricity (quad.) | {r.max_eccentricity:.4f} |\n"
        f"| KL cycle period | {r.kozai_cycle_period_yr:.1f} yr |\n"
        f"| Flag | {r.flag} |"
    )


def _cli() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Kozai-Lidov oscillation estimator")
    p.add_argument("inner_period_days", type=float)
    p.add_argument("outer_period_days", type=float)
    p.add_argument("inclination_deg", type=float)
    args = p.parse_args()
    r = compute_kozai_lidov(args.inner_period_days, args.outer_period_days,
                             args.inclination_deg)
    print(format_kozai_lidov_result(r))


if __name__ == "__main__":
    _cli()
