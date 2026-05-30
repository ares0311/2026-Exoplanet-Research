"""Check dynamical stability for multi-planet systems via Hill radius criterion.

Public API:
    HillStabilityResult  -- frozen dataclass
    check_hill_stability(periods_days, stellar_mass_msun,
                         planet_masses_mearth) -> HillStabilityResult
    format_hill_stability(result) -> str
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

_AU_PER_DAY_CBRT = (
    (6.674e-11 * 1.989e30) ** (1.0 / 3.0)
    / (2 * math.pi) ** (2.0 / 3.0)
    / 1.496e11
)
_MEARTH_MSUN = 3.003e-6
_HILL_FACTOR = 3.5


@dataclass(frozen=True)
class HillStabilityResult:
    n_planets: int
    min_hill_separation: float
    is_stable: bool
    flag: str


def _semi_major_axis_au(period_days: float, stellar_mass_msun: float) -> float:
    p_years = period_days / 365.25
    return (stellar_mass_msun * p_years ** 2) ** (1.0 / 3.0)


def check_hill_stability(
    periods_days: list[float],
    stellar_mass_msun: float,
    planet_masses_mearth: list[float],
) -> HillStabilityResult:
    if len(periods_days) < 2:
        return HillStabilityResult(
            n_planets=len(periods_days), min_hill_separation=0.0,
            is_stable=True, flag="SINGLE_PLANET",
        )
    if stellar_mass_msun <= 0:
        return HillStabilityResult(
            n_planets=len(periods_days), min_hill_separation=0.0,
            is_stable=False, flag="INVALID_STELLAR_MASS",
        )
    if len(planet_masses_mearth) != len(periods_days):
        return HillStabilityResult(
            n_planets=len(periods_days), min_hill_separation=0.0,
            is_stable=False, flag="MASS_PERIOD_MISMATCH",
        )
    pairs = sorted(zip(periods_days, planet_masses_mearth, strict=False))
    separations: list[float] = []
    for i in range(len(pairs) - 1):
        p1, m1 = pairs[i]
        p2, m2 = pairs[i + 1]
        a1 = _semi_major_axis_au(p1, stellar_mass_msun)
        a2 = _semi_major_axis_au(p2, stellar_mass_msun)
        m1_msun = m1 * _MEARTH_MSUN
        m2_msun = m2 * _MEARTH_MSUN
        rh1 = a1 * (m1_msun / (3.0 * stellar_mass_msun)) ** (1.0 / 3.0)
        rh2 = a2 * (m2_msun / (3.0 * stellar_mass_msun)) ** (1.0 / 3.0)
        rh_mutual = (rh1 + rh2) / 2.0
        separation_hill = (a2 - a1) / rh_mutual if rh_mutual > 0 else 0.0
        separations.append(separation_hill)
    min_sep = min(separations)
    is_stable = min_sep >= _HILL_FACTOR
    flag = "STABLE" if is_stable else "POTENTIALLY_UNSTABLE"
    return HillStabilityResult(
        n_planets=len(periods_days),
        min_hill_separation=min_sep,
        is_stable=is_stable,
        flag=flag,
    )


def format_hill_stability(result: HillStabilityResult) -> str:
    lines = [
        "## Hill Sphere Stability Check",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| N Planets | {result.n_planets} |",
        f"| Min Hill Separation | {result.min_hill_separation:.2f} R_H |",
        f"| Is Stable | {result.is_stable} |",
        f"| Flag | {result.flag} |",
    ]
    return "\n".join(lines)


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Check Hill sphere stability.")
    parser.add_argument("--periods", nargs="+", type=float, required=True)
    parser.add_argument("--masses-mearth", nargs="+", type=float, required=True)
    parser.add_argument("--stellar-mass", type=float, default=1.0)
    args = parser.parse_args()
    result = check_hill_stability(args.periods, args.stellar_mass, args.masses_mearth)
    print(format_hill_stability(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
