"""Build finder chart data from a catalog of stars.

Selects all stars within a given radius of the target using the Haversine formula,
sorted by magnitude (brightest first).

Public API
----------
FinderChartData(target_ra, target_dec, field_arcmin, n_stars, stars_in_field, flag)
build_finder_chart_data(target_ra_deg, target_dec_deg, stars, field_arcmin) -> FinderChartData
format_finder_chart_data(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class FinderChartData:
    target_ra: float
    target_dec: float
    field_arcmin: float
    n_stars: int
    stars_in_field: tuple[dict, ...]
    flag: str = "OK"  # "OK" | "EMPTY_FIELD" | "NO_CATALOG"


def _angular_sep_arcmin(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    """Haversine angular separation in arcmin."""
    ra1_r, dec1_r, ra2_r, dec2_r = (
        math.radians(ra1),
        math.radians(dec1),
        math.radians(ra2),
        math.radians(dec2),
    )
    delta_ra = ra2_r - ra1_r
    delta_dec = dec2_r - dec1_r
    a = math.sin(delta_dec / 2) ** 2 + math.cos(dec1_r) * math.cos(dec2_r) * math.sin(
        delta_ra / 2
    ) ** 2
    c = 2 * math.asin(min(1.0, math.sqrt(a)))
    return math.degrees(c) * 60.0


def build_finder_chart_data(
    target_ra_deg: float,
    target_dec_deg: float,
    stars: list[dict],
    field_arcmin: float = 10.0,
) -> FinderChartData:
    """Build finder chart data.

    Args:
        target_ra_deg: Target RA in degrees.
        target_dec_deg: Target declination in degrees.
        stars: List of dicts with keys ``ra_deg``, ``dec_deg``, ``mag``.
        field_arcmin: Field radius in arcminutes.

    Returns:
        :class:`FinderChartData`.
    """
    if not stars:
        return FinderChartData(
            target_ra=target_ra_deg,
            target_dec=target_dec_deg,
            field_arcmin=field_arcmin,
            n_stars=0,
            stars_in_field=(),
            flag="NO_CATALOG",
        )

    in_field = []
    for s in stars:
        sep = _angular_sep_arcmin(target_ra_deg, target_dec_deg, s["ra_deg"], s["dec_deg"])
        if sep <= field_arcmin:
            in_field.append({**s, "sep_arcmin": round(sep, 4)})

    in_field.sort(key=lambda x: x["mag"])
    flag = "EMPTY_FIELD" if not in_field else "OK"

    return FinderChartData(
        target_ra=target_ra_deg,
        target_dec=target_dec_deg,
        field_arcmin=field_arcmin,
        n_stars=len(in_field),
        stars_in_field=tuple(in_field),
        flag=flag,
    )


def format_finder_chart_data(result: FinderChartData) -> str:
    """Format finder chart data as Markdown."""
    lines = [
        "## Finder Chart Data",
        "",
        f"- Target RA: {result.target_ra:.4f}°",
        f"- Target Dec: {result.target_dec:.4f}°",
        f"- Field radius: {result.field_arcmin:.1f}'",
        f"- Stars in field: {result.n_stars}",
        f"- **Flag: {result.flag}**",
    ]
    if result.stars_in_field:
        lines += ["", "| # | RA (°) | Dec (°) | Mag | Sep (') |",
                  "|---|--------|---------|-----|---------|"]
        for i, s in enumerate(result.stars_in_field, 1):
            lines.append(
                f"| {i} | {s['ra_deg']:.4f} | {s['dec_deg']:.4f} |"
                f" {s['mag']:.2f} | {s['sep_arcmin']:.2f} |"
            )
    return "\n".join(lines) + "\n"


# keep field import to avoid F401 when imported elsewhere
_field_ref = field


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(
        prog="finder_chart_data_builder",
        description="Build finder chart data from a star catalog.",
    )
    parser.add_argument("target_ra_deg", type=float)
    parser.add_argument("target_dec_deg", type=float)
    parser.add_argument("stars_json", type=str, help="JSON file with list of star dicts")
    parser.add_argument("--field-arcmin", type=float, default=10.0)
    args = parser.parse_args(argv)

    with open(args.stars_json) as fh:
        stars = json.load(fh)

    result = build_finder_chart_data(
        args.target_ra_deg, args.target_dec_deg, stars, args.field_arcmin
    )
    print(format_finder_chart_data(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
