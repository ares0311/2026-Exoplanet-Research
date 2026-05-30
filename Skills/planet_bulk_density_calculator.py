"""Compute bulk density of a planet from mass and radius.

Density rho = M / (4/3 * pi * R^3) in g/cm^3
using M_earth_g = 5.972e27 g and R_earth_cm = 6.371e8 cm.

Composition hints:
    iron      : rho > 8 g/cm^3
    rocky     : 3 <= rho <= 8 g/cm^3
    water_world: 1 <= rho < 3 g/cm^3
    gas_dwarf  : rho < 1 g/cm^3

Public API
----------
BulkDensityResult(density_gcc, composition_hint, flag)
compute_bulk_density(m_planet_mearth, r_planet_rearth) -> BulkDensityResult
format_bulk_density_result(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass

_M_EARTH_G = 5.972e27  # grams
_R_EARTH_CM = 6.371e8  # cm


@dataclass(frozen=True)
class BulkDensityResult:
    density_gcc: float
    composition_hint: str  # "iron" / "rocky" / "water_world" / "gas_dwarf"
    flag: str = "OK"


def _composition_hint(rho: float) -> str:
    if rho > 8.0:
        return "iron"
    if rho >= 3.0:
        return "rocky"
    if rho >= 1.0:
        return "water_world"
    return "gas_dwarf"


def compute_bulk_density(
    m_planet_mearth: float,
    r_planet_rearth: float,
) -> BulkDensityResult:
    """Compute bulk density from planet mass and radius.

    Args:
        m_planet_mearth: Planet mass in Earth masses.
        r_planet_rearth: Planet radius in Earth radii.

    Returns:
        :class:`BulkDensityResult`.
    """
    if m_planet_mearth <= 0 or r_planet_rearth <= 0:
        return BulkDensityResult(density_gcc=0.0, composition_hint="gas_dwarf", flag="ERROR")

    m_g = float(m_planet_mearth) * _M_EARTH_G
    r_cm = float(r_planet_rearth) * _R_EARTH_CM
    volume_cm3 = (4.0 / 3.0) * math.pi * r_cm**3
    rho = m_g / volume_cm3

    return BulkDensityResult(
        density_gcc=round(rho, 4),
        composition_hint=_composition_hint(rho),
        flag="OK",
    )


def format_bulk_density_result(result: BulkDensityResult) -> str:
    """Format bulk density result as Markdown."""
    lines = [
        "## Planet Bulk Density",
        "",
        f"- Density: **{result.density_gcc:.3f} g/cm³**",
        f"- Composition hint: **{result.composition_hint}**",
        f"- Flag: {result.flag}",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="planet_bulk_density_calculator",
        description="Compute planet bulk density from mass and radius.",
    )
    parser.add_argument("m_planet_mearth", type=float, help="Planet mass in Earth masses")
    parser.add_argument("r_planet_rearth", type=float, help="Planet radius in Earth radii")
    args = parser.parse_args(argv)

    result = compute_bulk_density(args.m_planet_mearth, args.r_planet_rearth)
    print(format_bulk_density_result(result))
    return 0 if result.flag == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
