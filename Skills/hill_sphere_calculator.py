"""Compute the Hill sphere radius for a planet.

r_H = a * (M_p / (3 * M_star))^(1/3)

Result includes r_H in AU and in planet radii.
stability_flag is "stable" if r_H > 0.01 AU else "unstable".

Public API
----------
HillSphereResult(r_hill_au, r_hill_rp, stability_flag, flag)
compute_hill_sphere(a_au, m_planet_mearth, m_star_msun, r_planet_rearth) -> HillSphereResult
format_hill_sphere_result(result) -> str
"""
from __future__ import annotations

from dataclasses import dataclass

_M_EARTH_TO_MSUN = 3.003e-6  # 1 Earth mass in solar masses
_AU_TO_REARTH = 23455.0  # 1 AU in Earth radii


@dataclass(frozen=True)
class HillSphereResult:
    r_hill_au: float
    r_hill_rp: float
    stability_flag: str  # "stable" / "unstable"
    flag: str = "OK"


def compute_hill_sphere(
    a_au: float,
    m_planet_mearth: float,
    m_star_msun: float = 1.0,
    r_planet_rearth: float = 1.0,
) -> HillSphereResult:
    """Compute Hill sphere radius for a planet.

    Args:
        a_au: Semi-major axis in AU.
        m_planet_mearth: Planet mass in Earth masses.
        m_star_msun: Host star mass in solar masses.
        r_planet_rearth: Planet radius in Earth radii (for r_hill_rp).

    Returns:
        :class:`HillSphereResult`.
    """
    if any(v <= 0 for v in (a_au, m_planet_mearth, m_star_msun, r_planet_rearth)):
        return HillSphereResult(r_hill_au=0.0, r_hill_rp=0.0, stability_flag="unstable",
                                flag="ERROR")

    a = float(a_au)
    mp_msun = float(m_planet_mearth) * _M_EARTH_TO_MSUN
    ms = float(m_star_msun)

    r_hill_au = a * (mp_msun / (3.0 * ms)) ** (1.0 / 3.0)
    r_hill_rearth = r_hill_au * _AU_TO_REARTH
    r_hill_rp = r_hill_rearth / float(r_planet_rearth)

    stability = "stable" if r_hill_au > 0.01 else "unstable"

    return HillSphereResult(
        r_hill_au=round(r_hill_au, 8),
        r_hill_rp=round(r_hill_rp, 3),
        stability_flag=stability,
        flag="OK",
    )


def format_hill_sphere_result(result: HillSphereResult) -> str:
    """Format Hill sphere result as Markdown."""
    lines = [
        "## Hill Sphere",
        "",
        f"- Hill radius: **{result.r_hill_au:.6f} AU**",
        f"- Hill radius: **{result.r_hill_rp:.2f} R_p**",
        f"- Stability: **{result.stability_flag}**",
        f"- Flag: {result.flag}",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="hill_sphere_calculator",
        description="Compute Hill sphere radius for a planet.",
    )
    parser.add_argument("a_au", type=float, help="Semi-major axis in AU")
    parser.add_argument("m_planet_mearth", type=float, help="Planet mass in Earth masses")
    parser.add_argument("--m-star", type=float, default=1.0, help="Star mass in M_sun")
    parser.add_argument("--r-planet", type=float, default=1.0, help="Planet radius in R_earth")
    args = parser.parse_args(argv)

    result = compute_hill_sphere(
        args.a_au, args.m_planet_mearth,
        m_star_msun=args.m_star,
        r_planet_rearth=args.r_planet,
    )
    print(format_hill_sphere_result(result))
    return 0 if result.flag == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
