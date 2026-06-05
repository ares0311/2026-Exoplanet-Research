"""Locate a planet relative to the photoevaporation valley (radius gap)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvaporationValleyResult:
    planet_radius_rearth: float
    period_days: float
    gap_radius_rearth: float
    radius_ratio: float      # planet_radius / gap_radius
    valley_position: str     # ABOVE_GAP / IN_GAP / BELOW_GAP
    composition_hint: str    # GAS_ENVELOPE / TRANSITION / ROCKY_CORE
    flag: str


def locate_evaporation_valley(
    planet_radius_rearth: float,
    period_days: float,
    stellar_mass_msun: float = 1.0,
) -> EvaporationValleyResult:
    """Locate planet relative to the Fulton radius gap / evaporation valley.

    Fulton et al. (2017) / Van Eylen et al. (2018) gap centre:
      R_gap(P) ≈ 1.9 × (P / 10 d)^-0.11 × (Ms / Msun)^0.26  [R⊕]

    Gap width: ±0.3 R⊕ around R_gap.
    Above gap: gas envelopes retained; below gap: rocky cores / evaporated.

    Args:
        planet_radius_rearth: planet radius (Earth radii)
        period_days: orbital period (days)
        stellar_mass_msun: stellar mass (solar masses)
    """
    if planet_radius_rearth <= 0.0:
        return EvaporationValleyResult(planet_radius_rearth, period_days, float("nan"),
                                        float("nan"), "UNKNOWN", "UNKNOWN", "INVALID_RADIUS")
    if period_days <= 0.0:
        return EvaporationValleyResult(planet_radius_rearth, period_days, float("nan"),
                                        float("nan"), "UNKNOWN", "UNKNOWN", "INVALID_PERIOD")
    if stellar_mass_msun <= 0.0:
        return EvaporationValleyResult(planet_radius_rearth, period_days, float("nan"),
                                        float("nan"), "UNKNOWN", "UNKNOWN", "INVALID_STELLAR_MASS")

    r_gap = 1.9 * (period_days / 10.0) ** (-0.11) * stellar_mass_msun**0.26
    gap_half_width = 0.3

    ratio = planet_radius_rearth / r_gap

    if planet_radius_rearth > r_gap + gap_half_width:
        position = "ABOVE_GAP"
        composition = "GAS_ENVELOPE"
    elif planet_radius_rearth < r_gap - gap_half_width:
        position = "BELOW_GAP"
        composition = "ROCKY_CORE"
    else:
        position = "IN_GAP"
        composition = "TRANSITION"

    return EvaporationValleyResult(
        planet_radius_rearth=planet_radius_rearth,
        period_days=period_days,
        gap_radius_rearth=r_gap,
        radius_ratio=ratio,
        valley_position=position,
        composition_hint=composition,
        flag="OK",
    )


def format_evaporation_valley_result(r: EvaporationValleyResult) -> str:
    if r.flag != "OK":
        return f"EvaporationValley | flag={r.flag}"
    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| Planet radius | {r.planet_radius_rearth:.3f} R⊕ |\n"
        f"| Gap centre radius | {r.gap_radius_rearth:.3f} R⊕ |\n"
        f"| Radius ratio (Rp/Rgap) | {r.radius_ratio:.3f} |\n"
        f"| Valley position | {r.valley_position} |\n"
        f"| Composition hint | {r.composition_hint} |\n"
        f"| Flag | {r.flag} |"
    )


def _cli() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Evaporation valley locator")
    p.add_argument("radius_rearth", type=float)
    p.add_argument("period_days", type=float)
    p.add_argument("--mstar", type=float, default=1.0)
    args = p.parse_args()
    r = locate_evaporation_valley(args.radius_rearth, args.period_days,
                                   stellar_mass_msun=args.mstar)
    print(format_evaporation_valley_result(r))


if __name__ == "__main__":
    _cli()
