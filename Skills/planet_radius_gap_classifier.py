"""Classify a planet radius into Fulton+2017 radius gap categories.

Boundaries (Earth radii):
  rocky      < 1.5
  gap        [1.5, 2.0)
  sub_neptune [2.0, 3.5)
  neptune    [3.5, 6.0)
  giant      >= 6.0

Public API
----------
RadiusGapResult
classify_radius_gap(radius_rearth, *, period_days) -> RadiusGapResult
format_radius_gap(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass

_BOUNDARIES = [
    (0.0, 1.5, "rocky", "Rocky / super-Earth; likely solid composition"),
    (1.5, 2.0, "gap", "Radius gap (photoevaporation desert); ambiguous composition"),
    (2.0, 3.5, "sub_neptune", "Sub-Neptune; volatile-rich envelope likely"),
    (3.5, 6.0, "neptune", "Neptune-sized; substantial gas envelope"),
    (6.0, math.inf, "giant", "Giant planet (Saturn/Jupiter class)"),
]


@dataclass(frozen=True)
class RadiusGapResult:
    radius_rearth: float
    category: str
    description: str
    period_days: float | None
    flag: str  # "OK" | "INVALID"


def classify_radius_gap(
    radius_rearth: float,
    *,
    period_days: float | None = None,
) -> RadiusGapResult:
    """Classify a planet radius into Fulton+2017 radius gap categories.

    Args:
        radius_rearth: Planet radius in Earth radii.
        period_days: Orbital period in days (informational only).

    Returns:
        :class:`RadiusGapResult`.
    """
    if not math.isfinite(radius_rearth) or radius_rearth <= 0:
        return RadiusGapResult(
            radius_rearth=radius_rearth,
            category="unknown",
            description="Invalid radius",
            period_days=period_days,
            flag="INVALID",
        )

    for lo, hi, cat, desc in _BOUNDARIES:
        if lo <= radius_rearth < hi:
            return RadiusGapResult(
                radius_rearth=radius_rearth,
                category=cat,
                description=desc,
                period_days=period_days,
                flag="OK",
            )

    # Fallback (should not be reached given math.inf upper bound)
    return RadiusGapResult(
        radius_rearth=radius_rearth,
        category="unknown",
        description="Radius outside known boundaries",
        period_days=period_days,
        flag="INVALID",
    )


def format_radius_gap(result: RadiusGapResult) -> str:
    """Format radius gap classification result as Markdown."""
    period_str = f"{result.period_days} days" if result.period_days is not None else "N/A"
    lines = [
        "## Planet Radius Gap Classifier",
        "",
        f"- Radius: {result.radius_rearth} R_Earth",
        f"- Period: {period_str}",
        f"- **Category: {result.category}**",
        f"- Description: {result.description}",
        f"- **Flag: {result.flag}**",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="planet_radius_gap_classifier",
        description="Classify a planet radius into Fulton+2017 radius gap categories.",
    )
    parser.add_argument("radius_rearth", type=float)
    parser.add_argument("--period-days", type=float, default=None)
    args = parser.parse_args(argv)

    result = classify_radius_gap(args.radius_rearth, period_days=args.period_days)
    print(format_radius_gap(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
