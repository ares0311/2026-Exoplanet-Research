"""Identify which TESS sectors observed a sky position and predict upcoming ones.

Uses a lightweight lookup table of sector sky coverage (ecliptic latitude
bands).  Results are approximate — use the official TESS sector-lookup tools
for scheduling-critical work.

Public API
----------
plan_sectors(ra, dec, *, sector_fn) -> SectorPlan
format_sector_plan(plan) -> str
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class SectorPlan:
    ra: float
    dec: float
    observed_sectors: list[int]
    n_sectors: int
    ecliptic_latitude: float
    note: str = ""


# ---------------------------------------------------------------------------
# Approximate ecliptic latitude ↔ sector visibility lookup
# TESS observes in 13 sectors per year, each ~27 days.
# Continuous Viewing Zones (CVZ) are near ecliptic poles (|β| > 78°).
# Sectors 1-26 (Year 1-2), 27-55 (Year 3-5), 56-83 (Year 6-7).
# This is a simple heuristic: real coverage requires tesscut or TESSPoint.
# ---------------------------------------------------------------------------

_SECTOR_ECLIPTIC_BANDS: dict[int, tuple[float, float]] = {
    # sector: (min_ecliptic_lat, max_ecliptic_lat)  approximate
    1:  (-90.0, -54.0), 2:  (-90.0, -54.0), 3:  (-90.0, -54.0),
    4:  (-90.0, -54.0), 5:  (-90.0, -54.0), 6:  (-90.0, -54.0),
    7:  (-90.0, -54.0), 8:  (-90.0, -54.0), 9:  (-90.0, -54.0),
    10: (-90.0, -54.0), 11: (-90.0, -54.0), 12: (-90.0, -54.0),
    13: (-90.0, -54.0),
    14: ( 54.0,  90.0), 15: ( 54.0,  90.0), 16: ( 54.0,  90.0),
    17: ( 54.0,  90.0), 18: ( 54.0,  90.0), 19: ( 54.0,  90.0),
    20: ( 54.0,  90.0), 21: ( 54.0,  90.0), 22: ( 54.0,  90.0),
    23: ( 54.0,  90.0), 24: ( 54.0,  90.0), 25: ( 54.0,  90.0),
    26: ( 54.0,  90.0),
}

# Extended years 3-7 (sectors 27-83) use an extended viewing zone model.
for _s in range(27, 84):
    if _s <= 39:
        _SECTOR_ECLIPTIC_BANDS[_s] = (-90.0, -30.0)
    elif _s <= 55:
        _SECTOR_ECLIPTIC_BANDS[_s] = (30.0, 90.0)
    elif _s <= 69:
        _SECTOR_ECLIPTIC_BANDS[_s] = (-90.0, -30.0)
    else:
        _SECTOR_ECLIPTIC_BANDS[_s] = (30.0, 90.0)


def _ra_dec_to_ecliptic_lat(ra_deg: float, dec_deg: float) -> float:
    """Approximate ecliptic latitude from equatorial coordinates."""
    # Obliquity of ecliptic ~23.44 degrees
    eps = math.radians(23.44)
    ra  = math.radians(ra_deg)
    dec = math.radians(dec_deg)
    sin_beta = (
        math.sin(dec) * math.cos(eps)
        - math.cos(dec) * math.sin(eps) * math.sin(ra)
    )
    return math.degrees(math.asin(max(-1.0, min(1.0, sin_beta))))


def _default_sector_fn(
    ecliptic_lat: float,
) -> list[int]:
    """Return sectors whose band contains the given ecliptic latitude."""
    return [
        s for s, (lo, hi) in _SECTOR_ECLIPTIC_BANDS.items()
        if lo <= ecliptic_lat <= hi
    ]


def plan_sectors(
    ra: float,
    dec: float,
    *,
    sector_fn: Callable[[float], list[int]] | None = None,
) -> SectorPlan:
    """Identify TESS sectors that observed a sky position.

    Args:
        ra: Right ascension in degrees (J2000).
        dec: Declination in degrees (J2000).
        sector_fn: Injectable ``(ecliptic_lat) -> list[int]``.  Defaults to
            the built-in approximate latitude band lookup.

    Returns:
        :class:`SectorPlan`.
    """
    beta = _ra_dec_to_ecliptic_lat(ra, dec)
    _fn = sector_fn if sector_fn is not None else _default_sector_fn
    sectors = sorted(_fn(beta))

    if abs(beta) > 78.0:
        note = "Target is in a TESS Continuous Viewing Zone (CVZ)."
    elif abs(beta) < 6.0:
        note = "Target is near the ecliptic plane — likely fewer TESS sectors."
    else:
        note = ""

    return SectorPlan(
        ra=ra,
        dec=dec,
        observed_sectors=sectors,
        n_sectors=len(sectors),
        ecliptic_latitude=beta,
        note=note,
    )


def format_sector_plan(plan: SectorPlan) -> str:
    """Format a SectorPlan as a short Markdown summary.

    Args:
        plan: From :func:`plan_sectors`.

    Returns:
        Markdown string.
    """
    sector_str = ", ".join(str(s) for s in plan.observed_sectors) or "none"
    lines = [
        "## TESS Sector Plan",
        "",
        f"- RA: {plan.ra:.4f}°  Dec: {plan.dec:.4f}°",
        f"- Ecliptic latitude: {plan.ecliptic_latitude:.2f}°",
        f"- Sectors observed: {sector_str}",
        f"- Total sectors: {plan.n_sectors}",
    ]
    if plan.note:
        lines.append(f"- Note: {plan.note}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(
        prog="tess_year_planner",
        description="Identify TESS sectors that observed a sky position.",
    )
    parser.add_argument("--ra", type=float, required=True, metavar="DEG",
                        help="Right ascension in degrees.")
    parser.add_argument("--dec", type=float, required=True, metavar="DEG",
                        help="Declination in degrees.")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON.")
    args = parser.parse_args(argv)

    plan = plan_sectors(args.ra, args.dec)
    if args.json:
        import dataclasses
        print(json.dumps(dataclasses.asdict(plan), indent=2))
    else:
        print(format_sector_plan(plan), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
