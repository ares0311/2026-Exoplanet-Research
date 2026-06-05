"""Estimate planetary oblateness from rotation period and bulk properties."""
from __future__ import annotations

import math
from dataclasses import dataclass

_G = 6.674e-11
_MJUP_KG = 1.898e27
_RJUP_M = 7.1492e7
_SEC_PER_DAY = 86400.0


@dataclass(frozen=True)
class PlanetOblatenessResult:
    rotation_period_hours: float
    oblateness: float               # f = (R_eq - R_pol) / R_eq
    equatorial_radius_factor: float # R_eq / R_mean
    polar_radius_factor: float      # R_pol / R_mean
    transit_depth_difference_ppm: float  # transit depth change due to oblateness
    flag: str


def compute_planet_oblateness(
    planet_mass_mjup: float,
    planet_radius_rjup: float,
    rotation_period_hours: float,
    love_number_k2: float = 0.5,
) -> PlanetOblatenessResult:
    """Estimate oblateness from rotation and gravity.

    Equilibrium oblateness (Chandrasekhar 1969 / Seager & Hui 2002):
      f ≈ (5 k₂ / 4) × (ω² R³ / G M)

    Transit depth difference: δ_oblate - δ_sphere ≈ 2f × (R_p/R_s)² ppm × (R_s/R_p)²
    Here we return the fractional depth change = 2f (to be multiplied by (Rp/Rs)²).

    Args:
        planet_mass_mjup: planet mass (Jupiter masses)
        planet_radius_rjup: planet mean radius (Jupiter radii)
        rotation_period_hours: planetary rotation period (hours)
        love_number_k2: fluid Love number k₂ (0.5 = uniform density, ~0.3 for rocky)
    """
    if planet_mass_mjup <= 0.0:
        return PlanetOblatenessResult(rotation_period_hours, float("nan"),
                                       float("nan"), float("nan"), float("nan"),
                                       "INVALID_MASS")
    if planet_radius_rjup <= 0.0:
        return PlanetOblatenessResult(rotation_period_hours, float("nan"),
                                       float("nan"), float("nan"), float("nan"),
                                       "INVALID_RADIUS")
    if rotation_period_hours <= 0.0:
        return PlanetOblatenessResult(rotation_period_hours, float("nan"),
                                       float("nan"), float("nan"), float("nan"),
                                       "INVALID_ROTATION_PERIOD")

    mp_kg = planet_mass_mjup * _MJUP_KG
    rp_m = planet_radius_rjup * _RJUP_M
    omega = 2.0 * math.pi / (rotation_period_hours * 3600.0)

    f = (5.0 * love_number_k2 / 4.0) * (omega**2 * rp_m**3) / (_G * mp_kg)
    f = min(f, 0.5)  # physical upper limit

    r_eq_factor = 1.0 + f / 3.0
    r_pol_factor = 1.0 - 2.0 * f / 3.0

    # Transit depth difference ≈ 2f × depth; return 2f in ppm-per-unit-depth
    depth_diff_ppm = 2.0 * f * 1e6

    return PlanetOblatenessResult(
        rotation_period_hours=rotation_period_hours,
        oblateness=f,
        equatorial_radius_factor=r_eq_factor,
        polar_radius_factor=r_pol_factor,
        transit_depth_difference_ppm=depth_diff_ppm,
        flag="OK",
    )


def format_planet_oblateness_result(r: PlanetOblatenessResult) -> str:
    if r.flag != "OK":
        return f"PlanetOblateness | flag={r.flag}"
    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| Rotation period | {r.rotation_period_hours:.2f} hours |\n"
        f"| Oblateness f | {r.oblateness:.5f} |\n"
        f"| R_eq / R_mean | {r.equatorial_radius_factor:.5f} |\n"
        f"| R_pol / R_mean | {r.polar_radius_factor:.5f} |\n"
        f"| Transit depth diff. | {r.transit_depth_difference_ppm:.1f} ppm per unit depth |\n"
        f"| Flag | {r.flag} |"
    )


def _cli() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Planet oblateness estimator")
    p.add_argument("mass_mjup", type=float)
    p.add_argument("radius_rjup", type=float)
    p.add_argument("rotation_hours", type=float)
    args = p.parse_args()
    r = compute_planet_oblateness(args.mass_mjup, args.radius_rjup, args.rotation_hours)
    print(format_planet_oblateness_result(r))


if __name__ == "__main__":
    _cli()
