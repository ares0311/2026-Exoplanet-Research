"""Classify planetary interior structure from mass-radius using Zeng+2019 boundaries."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InteriorStructureResult:
    composition: str          # IRON_RICH / EARTH_LIKE / WATER_WORLD / GAS_DWARF / GAS_GIANT
    bulk_density_gcc: float   # g/cm³
    earth_density_ratio: float
    radius_category: str      # ROCKY / SUPER_EARTH / MINI_NEPTUNE / NEPTUNE / JOVIAN
    flag: str


_EARTH_DENSITY = 5.51   # g/cm³
_EARTH_MASS = 5.972e24  # kg
_EARTH_RADIUS = 6.371e6 # m


def classify_interior_structure(
    planet_mass_mearth: float,
    planet_radius_rearth: float,
) -> InteriorStructureResult:
    """Classify interior structure using Zeng et al. (2019) mass-radius diagram.

    Composition boundaries at given mass:
      - Iron-rich:   R < 0.84 * M^0.274   (50% Fe, 50% MgSiO3)
      - Earth-like:  R < 1.00 * M^0.274   (pure rocky, Zeng+2016)
      - Water world: R < 1.41 * M^0.280   (50% water ice)
      - Gas dwarf:   R < 2.00 * M^0.200   (H/He envelope ~1%)
      - Gas giant:   R ≥ 2.00 * M^0.200

    Args:
        planet_mass_mearth: planet mass (Earth masses)
        planet_radius_rearth: planet radius (Earth radii)
    """
    if planet_mass_mearth <= 0.0:
        return InteriorStructureResult("UNKNOWN", float("nan"),
                                        float("nan"), "UNKNOWN", "INVALID_MASS")
    if planet_radius_rearth <= 0.0:
        return InteriorStructureResult("UNKNOWN", float("nan"),
                                        float("nan"), "UNKNOWN", "INVALID_RADIUS")

    m = planet_mass_mearth
    r = planet_radius_rearth

    # Bulk density in g/cm³
    volume_earth = (4.0 / 3.0) * 3.14159 * _EARTH_RADIUS**3  # m³
    volume_planet = volume_earth * r**3
    mass_planet = _EARTH_MASS * m
    density_si = mass_planet / volume_planet  # kg/m³
    density_gcc = density_si * 1e-3  # g/cm³

    # Zeng+2019 boundary curves
    r_iron = 0.84 * m ** 0.274
    r_rocky = 1.00 * m ** 0.274
    r_water = 1.41 * m ** 0.280
    r_gas_dwarf = 2.00 * m ** 0.200

    if r < r_iron:
        composition = "IRON_RICH"
    elif r < r_rocky:
        composition = "EARTH_LIKE"
    elif r < r_water:
        composition = "WATER_WORLD"
    elif r < r_gas_dwarf:
        composition = "GAS_DWARF"
    else:
        composition = "GAS_GIANT"

    if r < 1.25:
        rad_cat = "ROCKY"
    elif r < 2.0:
        rad_cat = "SUPER_EARTH"
    elif r < 4.0:
        rad_cat = "MINI_NEPTUNE"
    elif r < 6.0:
        rad_cat = "NEPTUNE"
    else:
        rad_cat = "JOVIAN"

    return InteriorStructureResult(
        composition=composition,
        bulk_density_gcc=density_gcc,
        earth_density_ratio=density_gcc / _EARTH_DENSITY,
        radius_category=rad_cat,
        flag="OK",
    )


def format_interior_structure_result(r: InteriorStructureResult) -> str:
    if r.flag != "OK":
        return f"InteriorStructure | flag={r.flag}"
    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| Composition | {r.composition} |\n"
        f"| Bulk density | {r.bulk_density_gcc:.2f} g/cm³ |\n"
        f"| ρ / ρ_Earth | {r.earth_density_ratio:.3f} |\n"
        f"| Radius category | {r.radius_category} |\n"
        f"| Flag | {r.flag} |"
    )


def _cli() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Planet interior structure classifier")
    p.add_argument("mass_mearth", type=float)
    p.add_argument("radius_rearth", type=float)
    args = p.parse_args()
    r = classify_interior_structure(args.mass_mearth, args.radius_rearth)
    print(format_interior_structure_result(r))


if __name__ == "__main__":
    _cli()
