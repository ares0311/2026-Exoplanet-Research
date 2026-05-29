"""Compute escape velocity for a planet.

v_esc = sqrt(2 * G * M_p / R_p) in km/s

Constants:
    G = 6.674e-11 m^3 kg^-1 s^-2
    M_earth = 5.972e24 kg
    R_earth = 6.371e6 m

Atmospheric retention thresholds (rough heuristics):
    H2  retained if v_esc > 10 km/s
    H2O retained if v_esc > 5  km/s

Public API
----------
EscapeVelocityResult(v_esc_kms, can_retain_h2, can_retain_h2o, flag)
compute_escape_velocity(mass_mearth, radius_rearth) -> EscapeVelocityResult
format_escape_velocity(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass

_G = 6.674e-11      # m^3 kg^-1 s^-2
_M_EARTH = 5.972e24  # kg
_R_EARTH = 6.371e6   # m
_H2_THRESHOLD_KMS = 10.0
_H2O_THRESHOLD_KMS = 5.0


@dataclass(frozen=True)
class EscapeVelocityResult:
    v_esc_kms: float
    can_retain_h2: bool    # v_esc > 10 km/s (rough threshold)
    can_retain_h2o: bool   # v_esc > 5 km/s
    flag: str  # "OK" always


def compute_escape_velocity(
    mass_mearth: float,
    radius_rearth: float,
) -> EscapeVelocityResult:
    """Compute escape velocity for a planet.

    Args:
        mass_mearth: Planet mass in Earth masses.
        radius_rearth: Planet radius in Earth radii.

    Returns:
        :class:`EscapeVelocityResult`.
    """
    if mass_mearth <= 0 or radius_rearth <= 0:
        return EscapeVelocityResult(
            v_esc_kms=0.0,
            can_retain_h2=False,
            can_retain_h2o=False,
            flag="ERROR",
        )

    m_kg = float(mass_mearth) * _M_EARTH
    r_m = float(radius_rearth) * _R_EARTH
    v_esc_ms = math.sqrt(2.0 * _G * m_kg / r_m)
    v_esc_kms = v_esc_ms / 1000.0

    return EscapeVelocityResult(
        v_esc_kms=round(v_esc_kms, 4),
        can_retain_h2=v_esc_kms > _H2_THRESHOLD_KMS,
        can_retain_h2o=v_esc_kms > _H2O_THRESHOLD_KMS,
        flag="OK",
    )


def format_escape_velocity(result: EscapeVelocityResult) -> str:
    """Format escape velocity result as Markdown."""
    lines = [
        "## Planet Escape Velocity",
        "",
        f"- Escape velocity: **{result.v_esc_kms:.3f} km/s**",
        f"- Can retain H₂ (>10 km/s): {'yes' if result.can_retain_h2 else 'no'}",
        f"- Can retain H₂O (>5 km/s): {'yes' if result.can_retain_h2o else 'no'}",
        f"- Flag: {result.flag}",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        prog="planet_escape_velocity",
        description=__doc__,
    )
    p.add_argument("mass_mearth", type=float, help="Planet mass in Earth masses")
    p.add_argument("radius_rearth", type=float, help="Planet radius in Earth radii")
    args = p.parse_args(argv)
    r = compute_escape_velocity(args.mass_mearth, args.radius_rearth)
    print(format_escape_velocity(r))
    return 0 if r.flag == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
