"""Estimate planet mass from radius using the Chen & Kipping (2017) M-R relation.

Applies a broken power-law mass-radius relation with three regimes:
rocky (R < 1.23 R⊕), volatile-rich (1.23 ≤ R < 14.26 R⊕), and
Jovian (R ≥ 14.26 R⊕).  Distinct from ``rv_semiamplitude_estimator``
(which starts from mass) and ``planet_radius_estimator`` (which derives R
from observed depth + stellar radius).

Public API
----------
PlanetMassResult(radius_rearth, mass_mearth, mass_mjup,
                 composition_class, regime, flag)
estimate_planet_mass(radius_rearth) -> PlanetMassResult
format_planet_mass_result(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# Chen & Kipping (2017) power-law coefficients
# regime: (c, gamma) where M = c * R^gamma, R in R_earth, M in M_earth
_REGIMES: list[tuple[str, str, float, float, float, float]] = [
    # (name, composition, R_lo, R_hi, c, gamma)
    ("rocky",       "Rocky",          0.0,    1.23,   0.9781, 3.58),
    ("volatile",    "Volatile-rich",  1.23,   14.26,  1.436,  1.70),
    ("jovian",      "Jovian",         14.26,  1e9,    1317.0, 0.01),
]

_MEARTH_TO_MJUP = 1.0 / 317.83


@dataclass(frozen=True)
class PlanetMassResult:
    radius_rearth: float
    mass_mearth: float | None
    mass_mjup: float | None
    composition_class: str | None
    regime: str | None
    flag: str  # "OK" | "INVALID"


def estimate_planet_mass(radius_rearth: float) -> PlanetMassResult:
    """Estimate planet mass from radius via Chen & Kipping (2017).

    Args:
        radius_rearth: Planet radius in Earth radii.

    Returns:
        :class:`PlanetMassResult`.
    """
    if not math.isfinite(radius_rearth) or radius_rearth <= 0:
        return PlanetMassResult(radius_rearth, None, None, None, None, "INVALID")
    if radius_rearth > 100.0:
        return PlanetMassResult(radius_rearth, None, None, None, None, "INVALID")

    for regime, composition, r_lo, r_hi, c, gamma in _REGIMES:
        if r_lo <= radius_rearth < r_hi:
            mass_me = c * (radius_rearth ** gamma)
            mass_mj = mass_me * _MEARTH_TO_MJUP
            return PlanetMassResult(
                radius_rearth=radius_rearth,
                mass_mearth=round(mass_me, 4),
                mass_mjup=round(mass_mj, 6),
                composition_class=composition,
                regime=regime,
                flag="OK",
            )

    return PlanetMassResult(radius_rearth, None, None, None, None, "INVALID")


def format_planet_mass_result(result: PlanetMassResult) -> str:
    """Format planet mass estimate as Markdown."""
    lines = [
        "## Planet Mass Estimator (Chen & Kipping 2017)",
        "",
        f"- Radius: {result.radius_rearth} R_earth",
        f"- **Mass: {result.mass_mearth} M_earth ({result.mass_mjup} M_jup)**",
        f"- Composition class: {result.composition_class}",
        f"- Regime: {result.regime}",
        f"- **Flag: {result.flag}**",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="planet_mass_estimator",
        description="Estimate planet mass from radius via Chen & Kipping (2017).",
    )
    parser.add_argument("radius_rearth", type=float)
    args = parser.parse_args(argv)

    result = estimate_planet_mass(args.radius_rearth)
    print(format_planet_mass_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
