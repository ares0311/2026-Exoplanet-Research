"""Classify likely atmospheric composition from planet bulk density and mass-radius diagram."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

_M_EARTH_KG = 5.972e24
_R_EARTH_M = 6.371e6


@dataclass(frozen=True)
class AtmosphericCompositionResult:
    planet_radius_rearth: float
    planet_mass_mearth: float
    bulk_density_gcm3: float
    envelope_fraction: float        # approximate gas envelope fraction (0–1)
    composition_class: str          # ROCKY / WATER_WORLD / GAS_DWARF / NEPTUNIAN / GAS_GIANT
    likely_composition: str         # human-readable description
    flag: str


# Density thresholds (g/cm³) from Zeng et al. (2019) and Rogers (2015)
_EARTH_DENSITY = 5.51   # g/cm³


def compute_atmospheric_composition(
    planet_radius_rearth: float,
    planet_mass_mearth: float,
) -> AtmosphericCompositionResult:
    """
    Classify likely atmospheric composition from bulk density.

    Boundaries (approximate):
    ρ > 5.5 g/cm³    → ROCKY (predominantly rock/iron, thin or no atmosphere)
    3.0–5.5 g/cm³    → WATER_WORLD (significant water/ice fraction)
    1.0–3.0 g/cm³    → GAS_DWARF (sub-Neptune with gas envelope, <10% by mass)
    0.3–1.0 g/cm³    → NEPTUNIAN (substantial gas envelope, Neptune-like)
    ρ < 0.3 g/cm³    → GAS_GIANT (hydrogen-dominated, Jupiter-like)

    Envelope fraction approximated as (1 − ρ/ρ_rock) where ρ_rock = 5.5 g/cm³.

    Parameters
    ----------
    planet_radius_rearth: Planet radius in Earth radii.
    planet_mass_mearth:   Planet mass in Earth masses.
    """
    if not math.isfinite(planet_radius_rearth) or planet_radius_rearth <= 0:
        return AtmosphericCompositionResult(planet_radius_rearth, planet_mass_mearth,
                                            float("nan"), float("nan"),
                                            "UNKNOWN", "Unknown", "INVALID_RADIUS")
    if not math.isfinite(planet_mass_mearth) or planet_mass_mearth <= 0:
        return AtmosphericCompositionResult(planet_radius_rearth, planet_mass_mearth,
                                            float("nan"), float("nan"),
                                            "UNKNOWN", "Unknown", "INVALID_MASS")

    rp_m = planet_radius_rearth * _R_EARTH_M
    mp_kg = planet_mass_mearth * _M_EARTH_KG
    volume_m3 = (4.0 / 3.0) * math.pi * rp_m ** 3
    density_kg_m3 = mp_kg / volume_m3
    density_gcm3 = density_kg_m3 / 1000.0

    envelope_frac = max(0.0, min(1.0, 1.0 - density_gcm3 / 5.5))

    if density_gcm3 > 5.5:
        comp_class = "ROCKY"
        description = "Rocky/iron composition; minimal or no atmosphere expected"
    elif density_gcm3 > 3.0:
        comp_class = "WATER_WORLD"
        description = "Significant water ice/ocean; mixed rocky-water composition"
    elif density_gcm3 > 1.0:
        comp_class = "GAS_DWARF"
        description = "Sub-Neptune with H/He gas envelope (<~10% by mass)"
    elif density_gcm3 > 0.3:
        comp_class = "NEPTUNIAN"
        description = "Neptune-like; substantial H/He or volatile-rich envelope"
    else:
        comp_class = "GAS_GIANT"
        description = "Gas giant; predominantly H/He; Jupiter/Saturn analog"

    return AtmosphericCompositionResult(
        planet_radius_rearth=planet_radius_rearth,
        planet_mass_mearth=planet_mass_mearth,
        bulk_density_gcm3=round(density_gcm3, 4),
        envelope_fraction=round(envelope_frac, 4),
        composition_class=comp_class,
        likely_composition=description,
        flag="OK",
    )


def format_atmospheric_composition_result(r: AtmosphericCompositionResult) -> str:
    def _f(v: float, fmt: str = ".4f") -> str:
        return format(v, fmt) if math.isfinite(v) else "N/A"

    return (
        f"| Parameter | Value |\n|---|---|\n"
        f"| Radius (R_earth) | {_f(r.planet_radius_rearth)} |\n"
        f"| Mass (M_earth) | {_f(r.planet_mass_mearth)} |\n"
        f"| Bulk density (g/cm³) | {_f(r.bulk_density_gcm3)} |\n"
        f"| Gas envelope fraction | {_f(r.envelope_fraction)} |\n"
        f"| Composition class | {r.composition_class} |\n"
        f"| Likely composition | {r.likely_composition} |\n"
        f"| Flag | {r.flag} |\n"
    )


def _cli() -> int:
    p = argparse.ArgumentParser(
        description="Classify planet atmospheric composition from M-R diagram."
    )
    p.add_argument("planet_radius_rearth", type=float)
    p.add_argument("planet_mass_mearth", type=float)
    args = p.parse_args()
    r = compute_atmospheric_composition(args.planet_radius_rearth, args.planet_mass_mearth)
    print(format_atmospheric_composition_result(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
