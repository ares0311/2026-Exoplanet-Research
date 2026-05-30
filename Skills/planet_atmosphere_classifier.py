"""Classify likely atmosphere type from planet radius and insolation flux.

Classification rules:
- "rocky_no_atm"  : R < 1.5 and S > 10
- "rocky_thin_atm": R < 1.5 and S <= 10
- "water_world"   : 1.5 <= R < 2.5 and S < 4
- "sub_neptune"   : 1.5 <= R < 4.0 (S >= 4 OR 2.5 <= R < 4.0)
- "neptune_like"  : 4.0 <= R < 8.0
- "gas_giant"     : R >= 8.0

Public API
----------
AtmosphereClassResult(class_label, confidence, rationale, flag)
classify_atmosphere(radius_rearth, insolation_searth) -> AtmosphereClassResult
format_atmosphere_class(result) -> str
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AtmosphereClassResult:
    class_label: str
    confidence: str  # "high" / "medium" / "low"
    rationale: str
    flag: str  # "OK" always


def classify_atmosphere(
    radius_rearth: float,
    insolation_searth: float = 1.0,
) -> AtmosphereClassResult:
    """Classify likely atmosphere type.

    Args:
        radius_rearth: Planet radius in Earth radii.
        insolation_searth: Stellar insolation in Earth units (default 1.0).

    Returns:
        :class:`AtmosphereClassResult`.
    """
    r = float(radius_rearth)
    s = float(insolation_searth)

    if r >= 8.0:
        label = "gas_giant"
        conf = "high"
        rationale = "Very large radius strongly suggests a hydrogen/helium envelope."
    elif r >= 4.0:
        label = "neptune_like"
        conf = "high"
        rationale = "Neptune-sized planet likely retains a substantial gas envelope."
    elif r >= 2.5:
        label = "sub_neptune"
        conf = "medium"
        rationale = "Sub-Neptune size consistent with thick volatile envelope."
    elif r >= 1.5:
        if s < 4.0:
            label = "water_world"
            conf = "medium"
            rationale = "Low insolation and intermediate radius suggest a water-rich composition."
        else:
            label = "sub_neptune"
            conf = "medium"
            rationale = (
                "High insolation may strip volatiles; likely a sub-Neptune or volatile-rich world."
            )
    else:
        # r < 1.5
        if s > 10.0:
            label = "rocky_no_atm"
            conf = "high"
            rationale = "High insolation on a small planet likely drives atmospheric escape."
        else:
            label = "rocky_thin_atm"
            conf = "medium"
            rationale = "Small radius with moderate insolation; may retain a thin atmosphere."

    return AtmosphereClassResult(
        class_label=label,
        confidence=conf,
        rationale=rationale,
        flag="OK",
    )


def format_atmosphere_class(result: AtmosphereClassResult) -> str:
    """Format atmosphere classification result as Markdown."""
    lines = [
        "## Atmosphere Classification",
        "",
        f"- Class: **{result.class_label}**",
        f"- Confidence: {result.confidence}",
        f"- Rationale: {result.rationale}",
        f"- Flag: {result.flag}",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        prog="planet_atmosphere_classifier",
        description=__doc__,
    )
    p.add_argument("radius_rearth", type=float, help="Planet radius in Earth radii")
    p.add_argument("--insolation", type=float, default=1.0, help="Insolation in S_earth")
    args = p.parse_args(argv)
    r = classify_atmosphere(args.radius_rearth, args.insolation)
    print(format_atmosphere_class(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
