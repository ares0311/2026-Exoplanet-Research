"""Estimate protoplanetary disk formation zones from stellar luminosity."""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class FormationZonesResult:
    snow_line_au: float             # H2O ice condensation radius
    silicate_line_au: float         # silicate dust condensation (rock line)
    co2_line_au: float              # CO2 ice line
    co_line_au: float               # CO ice line
    inner_hole_au: float            # dust sublimation radius
    rocky_zone_outer_au: float      # outer edge of rocky planet zone
    habitable_zone_inner_au: float  # inner HZ boundary (recent Venus)
    habitable_zone_outer_au: float  # outer HZ boundary (early Mars)
    flag: str


def estimate_formation_zones(
    stellar_luminosity_lsun: float,
    stellar_teff_k: float = 5778.0,
    disk_alpha: float = 0.006,
) -> FormationZonesResult:
    """Estimate protoplanetary disk condensation zones.

    Ice/condensation lines scale with stellar flux (Hayashi 1981):
      r_ice ≈ 2.7 × (L★/L☉)^0.5 AU   (H2O snow line)
      r_rock ≈ 0.5 × (L★/L☉)^0.5 AU  (silicate sublimation)

    Habitable zone (Kopparapu+2013 empirical):
      r_HZ_inner ≈ 0.75 × (L★/L☉)^0.5 AU
      r_HZ_outer ≈ 1.77 × (L★/L☉)^0.5 AU

    Args:
        stellar_luminosity_lsun: stellar luminosity (solar luminosities)
        stellar_teff_k: effective temperature (K) for inner dust sublimation
        disk_alpha: disk viscosity parameter (Shakura-Sunyaev α)
    """
    if stellar_luminosity_lsun <= 0.0:
        return FormationZonesResult(float("nan"), float("nan"), float("nan"),
                                     float("nan"), float("nan"), float("nan"),
                                     float("nan"), float("nan"), "INVALID_LUMINOSITY")

    lum = stellar_luminosity_lsun
    sqrt_l = math.sqrt(lum)

    # Condensation lines (Hayashi 1981 + Kennedy & Kenyon 2008)
    snow_line = 2.7 * sqrt_l           # H2O
    silicate_line = 0.5 * sqrt_l       # rock/silicate sublimation
    co2_line = 5.5 * sqrt_l            # CO2 ice
    co_line = 9.0 * sqrt_l             # CO ice

    # Inner dust sublimation radius: T_sub ~ 1500 K (silicates)
    # R_sub = R★ × (T★/T_sub)²  ≈ 0.034 × (L/L☉)^0.5  AU
    inner_hole = 0.034 * sqrt_l

    # Rocky planet zone outer edge ≈ snow line
    rocky_zone_outer = snow_line

    # Habitable zone (Kopparapu+2013)
    t_eff = stellar_teff_k
    t_s = t_eff - 5780.0
    hz_inner = (0.7533 + 1.4022e-4 * t_s + 2.2968e-9 * t_s**2 -
                3.5832e-12 * t_s**3 - 5.1097e-15 * t_s**4) * sqrt_l
    hz_outer = (1.7753 + 1.4316e-4 * t_s + 2.9875e-9 * t_s**2 -
                7.5702e-12 * t_s**3 - 1.1635e-15 * t_s**4) * sqrt_l

    return FormationZonesResult(
        snow_line_au=snow_line,
        silicate_line_au=silicate_line,
        co2_line_au=co2_line,
        co_line_au=co_line,
        inner_hole_au=inner_hole,
        rocky_zone_outer_au=rocky_zone_outer,
        habitable_zone_inner_au=max(hz_inner, 0.01),
        habitable_zone_outer_au=max(hz_outer, 0.02),
        flag="OK",
    )


def format_formation_zones_result(r: FormationZonesResult) -> str:
    if r.flag != "OK":
        return f"FormationZones | flag={r.flag}"
    return (
        f"| Zone | Distance (AU) |\n"
        f"|---|---|\n"
        f"| Dust sublimation (inner hole) | {r.inner_hole_au:.3f} |\n"
        f"| Silicate line | {r.silicate_line_au:.3f} |\n"
        f"| HZ inner | {r.habitable_zone_inner_au:.3f} |\n"
        f"| HZ outer | {r.habitable_zone_outer_au:.3f} |\n"
        f"| H₂O snow line | {r.snow_line_au:.3f} |\n"
        f"| CO₂ ice line | {r.co2_line_au:.3f} |\n"
        f"| CO ice line | {r.co_line_au:.3f} |\n"
        f"| Flag | {r.flag} |"
    )


def _cli() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Planet formation zone estimator")
    p.add_argument("luminosity_lsun", type=float)
    p.add_argument("--teff", type=float, default=5778.0)
    args = p.parse_args()
    r = estimate_formation_zones(args.luminosity_lsun, stellar_teff_k=args.teff)
    print(format_formation_zones_result(r))


if __name__ == "__main__":
    _cli()
