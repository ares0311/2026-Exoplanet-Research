"""Select good comparison stars from a list of neighbours.

Filters by:
- magnitude similarity (within delta_mag of target)
- minimum angular separation (min_sep_arcsec)
- maximum angular separation (max_sep_arcmin)

Angular separation computed via the haversine formula.

Public API
----------
NeighbourStar(mag, ra_deg, dec_deg)  -- input type
ComparisonStarResult(n_candidates, n_selected, selected_indices, flag)
find_comparison_stars(target_mag, target_ra_deg, target_dec_deg, neighbours, ...)
    -> ComparisonStarResult
format_comparison_star_result(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class NeighbourStar:
    mag: float
    ra_deg: float
    dec_deg: float


@dataclass(frozen=True)
class ComparisonStarResult:
    n_candidates: int
    n_selected: int
    selected_indices: tuple[int, ...]
    flag: str = "OK"


def _angular_sep_arcsec(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    """Angular separation in arcseconds using the haversine formula."""
    r = math.pi / 180.0
    d1, d2 = dec1 * r, dec2 * r
    dra = (ra2 - ra1) * r
    a = math.sin(dra / 2) ** 2
    b = math.cos(d1) * math.cos(d2) * a
    c = math.sin((d2 - d1) / 2) ** 2 + b
    sep_rad = 2.0 * math.asin(math.sqrt(max(c, 0.0)))
    return sep_rad * (180.0 / math.pi) * 3600.0


def find_comparison_stars(
    target_mag: float,
    target_ra_deg: float,
    target_dec_deg: float,
    neighbours: list[NeighbourStar],
    *,
    delta_mag: float = 1.0,
    min_sep_arcsec: float = 10.0,
    max_sep_arcmin: float = 15.0,
) -> ComparisonStarResult:
    """Select good comparison stars from a list of neighbours.

    Args:
        target_mag: Apparent magnitude of the target star.
        target_ra_deg: Target RA in degrees.
        target_dec_deg: Target Dec in degrees.
        neighbours: List of NeighbourStar objects.
        delta_mag: Maximum magnitude difference allowed.
        min_sep_arcsec: Minimum angular separation in arcseconds.
        max_sep_arcmin: Maximum angular separation in arcminutes.

    Returns:
        :class:`ComparisonStarResult`.
    """
    max_sep_arcsec = max_sep_arcmin * 60.0
    selected: list[int] = []

    for i, star in enumerate(neighbours):
        if abs(star.mag - target_mag) > delta_mag:
            continue
        sep = _angular_sep_arcsec(target_ra_deg, target_dec_deg, star.ra_deg, star.dec_deg)
        if sep < min_sep_arcsec or sep > max_sep_arcsec:
            continue
        selected.append(i)

    flag = "OK" if selected else "WARNING"
    return ComparisonStarResult(
        n_candidates=len(neighbours),
        n_selected=len(selected),
        selected_indices=tuple(selected),
        flag=flag,
    )


def format_comparison_star_result(result: ComparisonStarResult) -> str:
    """Format comparison star result as Markdown."""
    idx_str = (
        ", ".join(str(i) for i in result.selected_indices) if result.selected_indices else "none"
    )
    lines = [
        "## Comparison Star Selection",
        "",
        f"- Candidates considered: {result.n_candidates}",
        f"- Stars selected: **{result.n_selected}**",
        f"- Selected indices: {idx_str}",
        f"- Flag: {result.flag}",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(
        prog="comparison_star_finder",
        description="Select comparison stars for differential photometry.",
    )
    parser.add_argument("target_mag", type=float)
    parser.add_argument("target_ra", type=float)
    parser.add_argument("target_dec", type=float)
    parser.add_argument(
        "--neighbours",
        type=str,
        default="[]",
        help='JSON list of [{"mag":X,"ra_deg":Y,"dec_deg":Z}]',
    )
    parser.add_argument("--delta-mag", type=float, default=1.0)
    parser.add_argument("--min-sep-arcsec", type=float, default=10.0)
    parser.add_argument("--max-sep-arcmin", type=float, default=15.0)
    args = parser.parse_args(argv)

    raw = json.loads(args.neighbours)
    neighbours = [NeighbourStar(**s) for s in raw]
    result = find_comparison_stars(
        args.target_mag, args.target_ra, args.target_dec, neighbours,
        delta_mag=args.delta_mag,
        min_sep_arcsec=args.min_sep_arcsec,
        max_sep_arcmin=args.max_sep_arcmin,
    )
    print(format_comparison_star_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
