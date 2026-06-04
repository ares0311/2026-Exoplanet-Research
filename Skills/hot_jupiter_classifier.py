"""Classify planetary regime from orbital period and planet radius."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

# Regime boundaries (Rp in R_Earth, Period in days)
# Based on Fulton et al. (2017) radius gap + Mazeh et al. (2016) period boundaries
_RADIUS_EARTH_TO_JUPITER = 1.0 / 11.209


@dataclass(frozen=True)
class PlanetRegimeResult:
    period_days: float
    radius_rearth: float
    radius_rjupiter: float
    regime: str
    period_class: str   # ULTRA_HOT / HOT / WARM / COLD / LONG_PERIOD
    radius_class: str   # SUPER_EARTH / SUB_NEPTUNE / NEPTUNIAN / JOVIAN / SUPER_JOVIAN
    flag: str


def classify_planet_regime(
    period_days: float,
    radius_rearth: float,
) -> PlanetRegimeResult:
    """
    Classify exoplanet into standard regime from period and radius.

    Period classes:
    - ULTRA_HOT: P < 1 d
    - HOT: 1 ≤ P < 10 d
    - WARM: 10 ≤ P < 100 d
    - COLD: 100 ≤ P < 365 d
    - LONG_PERIOD: P ≥ 365 d

    Radius classes (Fulton gap at ~1.7 R_Earth, Neptune at ~3.9):
    - SUPER_EARTH: Rp < 1.7 R_Earth
    - SUB_NEPTUNE: 1.7 ≤ Rp < 3.9 R_Earth
    - NEPTUNIAN: 3.9 ≤ Rp < 6.0 R_Earth
    - JOVIAN: 6.0 ≤ Rp < 15.0 R_Earth  (~0.53 – 1.34 Rjup)
    - SUPER_JOVIAN: Rp ≥ 15 R_Earth

    Combined regime: e.g. HOT_JUPITER, WARM_NEPTUNE, COLD_SUPER_EARTH.
    """
    if not math.isfinite(period_days) or period_days <= 0.0:
        return PlanetRegimeResult(
            period_days=period_days, radius_rearth=radius_rearth,
            radius_rjupiter=float("nan"),
            regime="UNKNOWN", period_class="UNKNOWN", radius_class="UNKNOWN",
            flag="INVALID_PERIOD",
        )
    if not math.isfinite(radius_rearth) or radius_rearth <= 0.0:
        return PlanetRegimeResult(
            period_days=period_days, radius_rearth=radius_rearth,
            radius_rjupiter=float("nan"),
            regime="UNKNOWN", period_class="UNKNOWN", radius_class="UNKNOWN",
            flag="INVALID_RADIUS",
        )

    # Period class
    if period_days < 1.0:
        p_class = "ULTRA_HOT"
    elif period_days < 10.0:
        p_class = "HOT"
    elif period_days < 100.0:
        p_class = "WARM"
    elif period_days < 365.0:
        p_class = "COLD"
    else:
        p_class = "LONG_PERIOD"

    # Radius class
    if radius_rearth < 1.7:
        r_class = "SUPER_EARTH"
    elif radius_rearth < 3.9:
        r_class = "SUB_NEPTUNE"
    elif radius_rearth < 6.0:
        r_class = "NEPTUNIAN"
    elif radius_rearth < 15.0:
        r_class = "JOVIAN"
    else:
        r_class = "SUPER_JOVIAN"

    # Combined regime name
    regime_map: dict[tuple[str, str], str] = {
        ("ULTRA_HOT", "JOVIAN"): "ULTRA_HOT_JUPITER",
        ("HOT", "JOVIAN"): "HOT_JUPITER",
        ("WARM", "JOVIAN"): "WARM_JUPITER",
        ("COLD", "JOVIAN"): "COLD_JUPITER",
        ("LONG_PERIOD", "JOVIAN"): "LONG_PERIOD_GIANT",
        ("ULTRA_HOT", "SUPER_JOVIAN"): "ULTRA_HOT_SUPER_JOVIAN",
        ("HOT", "SUPER_JOVIAN"): "HOT_SUPER_JOVIAN",
    }
    regime = regime_map.get(
        (p_class, r_class),
        f"{p_class}_{r_class}",
    )

    rjup = radius_rearth * _RADIUS_EARTH_TO_JUPITER

    return PlanetRegimeResult(
        period_days=period_days,
        radius_rearth=radius_rearth,
        radius_rjupiter=round(rjup, 4),
        regime=regime,
        period_class=p_class,
        radius_class=r_class,
        flag="OK",
    )


def format_regime_result(r: PlanetRegimeResult) -> str:
    rjup = f"{r.radius_rjupiter:.4f}" if math.isfinite(r.radius_rjupiter) else "N/A"
    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| Period (days) | {r.period_days:.4f} |\n"
        f"| Radius (R_Earth) | {r.radius_rearth:.4f} |\n"
        f"| Radius (R_Jupiter) | {rjup} |\n"
        f"| Period class | {r.period_class} |\n"
        f"| Radius class | {r.radius_class} |\n"
        f"| Regime | {r.regime} |\n"
        f"| Flag | {r.flag} |\n"
    )


def _cli() -> int:
    p = argparse.ArgumentParser(description="Classify planet regime from period and radius.")
    p.add_argument("period_days", type=float)
    p.add_argument("radius_rearth", type=float)
    args = p.parse_args()
    r = classify_planet_regime(args.period_days, args.radius_rearth)
    print(format_regime_result(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
