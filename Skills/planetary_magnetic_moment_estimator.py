"""Estimate planetary magnetic moment from dynamo scaling laws."""
from __future__ import annotations

import math
from dataclasses import dataclass

_MEARTH_KG = 5.972e24
_REARTH_M = 6.371e6
_MJUP_KG = 1.898e27
_RJUP_M = 7.1492e7
_G = 6.674e-11


@dataclass(frozen=True)
class MagneticMomentResult:
    magnetic_moment_earth_units: float  # M / M_Earth (Earth = 8e22 A·m²)
    surface_field_gauss: float          # equatorial surface field (Gauss)
    magnetospheric_radius_rp: float     # magnetopause standoff in planet radii
    dynamo_class: str                   # STRONG / MODERATE / WEAK / ABSENT
    flag: str


_M_EARTH_AM2 = 8.0e22   # A·m²  (Earth's magnetic dipole moment)


def estimate_magnetic_moment(
    planet_mass_mearth: float,
    planet_radius_rearth: float,
    rotation_period_days: float = 1.0,
    core_fraction: float = 0.3,
    conductivity_factor: float = 1.0,
) -> MagneticMomentResult:
    """Estimate planetary magnetic moment from Christensen+2009 dynamo scaling.

    Christensen et al. (2009) energy flux scaling:
      M ∝ (ρ_core)^(1/6) × (q_ohm)^(1/3) × R_core^(7/3)
    Simplified power-law (Zuluaga & Cuartas 2012):
      M / M_Earth ≈ (M_p / M_Earth)^0.75 × f_core^(4/3) × (P_rot/P_Earth)^(-0.5)

    Args:
        planet_mass_mearth: planet mass (Earth masses)
        planet_radius_rearth: planet radius (Earth radii)
        rotation_period_days: rotation period (days); shorter → stronger field
        core_fraction: fractional core radius (default 0.3 for Earth-like)
        conductivity_factor: relative core conductivity (1 = Earth-like)
    """
    if planet_mass_mearth <= 0.0:
        return MagneticMomentResult(float("nan"), float("nan"), float("nan"),
                                     "UNKNOWN", "INVALID_MASS")
    if planet_radius_rearth <= 0.0:
        return MagneticMomentResult(float("nan"), float("nan"), float("nan"),
                                     "UNKNOWN", "INVALID_RADIUS")
    if rotation_period_days <= 0.0:
        return MagneticMomentResult(float("nan"), float("nan"), float("nan"),
                                     "UNKNOWN", "INVALID_PERIOD")

    p_earth_days = 1.0  # Earth's rotation period

    # Zuluaga & Cuartas (2012) simplified scaling
    m_ratio = (planet_mass_mearth**0.75 *
               (core_fraction / 0.3) ** (4.0 / 3.0) *
               (rotation_period_days / p_earth_days) ** (-0.5) *
               conductivity_factor)

    m_am2 = m_ratio * _M_EARTH_AM2

    # Surface equatorial field: B = μ0 * M / (4π * Rp^3)
    rp_m = planet_radius_rearth * _REARTH_M
    mu0 = 4.0 * math.pi * 1e-7
    b_surface_t = mu0 * m_am2 / (4.0 * math.pi * rp_m**3)
    b_surface_gauss = b_surface_t * 1e4   # Tesla → Gauss

    # Magnetopause standoff: R_mp = Rp * (B²/(2μ0 * ρ_sw * v_sw²))^(1/6)
    # Approximate: R_mp / Rp ≈ (M/M_Earth)^(1/3) × k
    r_mp_rp = 10.0 * m_ratio ** (1.0 / 3.0)  # 10 Rp for Earth

    if m_ratio >= 5.0:
        dclass = "STRONG"
    elif m_ratio >= 0.5:
        dclass = "MODERATE"
    elif m_ratio >= 0.05:
        dclass = "WEAK"
    else:
        dclass = "ABSENT"

    return MagneticMomentResult(
        magnetic_moment_earth_units=m_ratio,
        surface_field_gauss=b_surface_gauss,
        magnetospheric_radius_rp=r_mp_rp,
        dynamo_class=dclass,
        flag="OK",
    )


def format_magnetic_moment_result(r: MagneticMomentResult) -> str:
    if r.flag != "OK":
        return f"MagneticMoment | flag={r.flag}"
    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| Magnetic moment | {r.magnetic_moment_earth_units:.3f} M_Earth |\n"
        f"| Surface field | {r.surface_field_gauss:.3f} G |\n"
        f"| Magnetopause | {r.magnetospheric_radius_rp:.1f} Rp |\n"
        f"| Dynamo class | {r.dynamo_class} |\n"
        f"| Flag | {r.flag} |"
    )


def _cli() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Planetary magnetic moment estimator")
    p.add_argument("mass_mearth", type=float)
    p.add_argument("radius_rearth", type=float)
    p.add_argument("--prot", type=float, default=1.0, help="Rotation period (days)")
    args = p.parse_args()
    r = estimate_magnetic_moment(args.mass_mearth, args.radius_rearth,
                                  rotation_period_days=args.prot)
    print(format_magnetic_moment_result(r))


if __name__ == "__main__":
    _cli()
