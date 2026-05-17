"""Check for nearby TIC stars that could contaminate a target's aperture.

Queries the TESS Input Catalog within a given radius and returns stars bright
enough to cause measurable transit depth dilution.

Public API
----------
NearbyStar(tic_id, separation_arcsec, tmag, delta_tmag, dilution_fraction)
NearbyStarResult(target_tic_id, target_tmag, neighbors, total_dilution_fraction,
                 n_significant)
check_nearby_stars(target_tic_id, target_ra, target_dec, target_tmag, *,
                   radius_arcsec, delta_tmag_limit, catalog_fn) -> NearbyStarResult
format_nearby_result(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class NearbyStar:
    tic_id: int
    separation_arcsec: float
    tmag: float
    delta_tmag: float          # tmag - target_tmag (positive = fainter)
    dilution_fraction: float   # flux_neighbor / flux_target


@dataclass(frozen=True)
class NearbyStarResult:
    target_tic_id: int
    target_tmag: float
    neighbors: tuple[NearbyStar, ...]
    total_dilution_fraction: float  # sum of all neighbor dilutions
    n_significant: int              # neighbors with delta_tmag < 5


def _angular_separation(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    """Great-circle separation in arcseconds."""
    d2r = math.pi / 180.0
    dlat = (dec2 - dec1) * d2r
    dlon = (ra2 - ra1) * d2r * math.cos((dec1 + dec2) / 2.0 * d2r)
    return math.sqrt(dlat ** 2 + dlon ** 2) / d2r * 3600.0


def _default_catalog_fn(
    ra: float, dec: float, radius_arcsec: float
) -> list[dict]:
    """Stub — returns empty list; real implementation uses astroquery."""
    return []


def check_nearby_stars(
    target_tic_id: int,
    target_ra: float,
    target_dec: float,
    target_tmag: float,
    *,
    radius_arcsec: float = 63.0,
    delta_tmag_limit: float = 8.0,
    catalog_fn=None,
) -> NearbyStarResult:
    """Find nearby TIC stars that could dilute the target's transit depth.

    Args:
        target_tic_id: TIC ID of the target star.
        target_ra: Right ascension in degrees.
        target_dec: Declination in degrees.
        target_tmag: Target TESS magnitude.
        radius_arcsec: Search radius (default 63″ = 3 TESS pixels).
        delta_tmag_limit: Only include stars fainter than target by this much.
        catalog_fn: Injectable; called as ``catalog_fn(ra, dec, radius_arcsec)``
            returning list of dicts with keys ``tic_id``, ``ra``, ``dec``, ``tmag``.

    Returns:
        :class:`NearbyStarResult`.
    """
    if catalog_fn is None:
        catalog_fn = _default_catalog_fn

    rows = catalog_fn(target_ra, target_dec, radius_arcsec)

    neighbors: list[NearbyStar] = []
    for row in rows:
        try:
            rid = int(row["tic_id"])
            rra = float(row["ra"])
            rdec = float(row["dec"])
            rtmag = float(row["tmag"])
        except (KeyError, TypeError, ValueError):
            continue

        if rid == target_tic_id:
            continue

        sep = _angular_separation(target_ra, target_dec, rra, rdec)
        if sep > radius_arcsec:
            continue

        dmag = rtmag - target_tmag
        if dmag > delta_tmag_limit:
            continue

        flux_ratio = 10.0 ** (-0.4 * dmag)
        neighbors.append(NearbyStar(
            tic_id=rid,
            separation_arcsec=round(sep, 2),
            tmag=rtmag,
            delta_tmag=round(dmag, 3),
            dilution_fraction=round(flux_ratio, 6),
        ))

    neighbors.sort(key=lambda s: s.separation_arcsec)
    total_dil = sum(s.dilution_fraction for s in neighbors)
    n_sig = sum(1 for s in neighbors if s.delta_tmag < 5.0)

    return NearbyStarResult(
        target_tic_id=target_tic_id,
        target_tmag=target_tmag,
        neighbors=tuple(neighbors),
        total_dilution_fraction=round(total_dil, 6),
        n_significant=n_sig,
    )


def format_nearby_result(result: NearbyStarResult) -> str:
    """Format nearby-star result as Markdown."""
    lines = [
        "## Nearby Star Contamination Check",
        "",
        f"- Target TIC {result.target_tic_id} (Tmag={result.target_tmag:.2f})",
        f"- Neighbors found: {len(result.neighbors)}",
        f"- Significant neighbors (ΔTmag < 5): {result.n_significant}",
        f"- Total dilution fraction: {result.total_dilution_fraction:.4f}",
    ]
    if result.neighbors:
        lines += ["", "| TIC ID | Sep (″) | Tmag | ΔTmag | Dilution |",
                  "|---|---|---|---|---|"]
        for s in result.neighbors[:10]:
            lines.append(
                f"| {s.tic_id} | {s.separation_arcsec:.1f} | {s.tmag:.2f} |"
                f" {s.delta_tmag:+.2f} | {s.dilution_fraction:.4f} |"
            )
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="nearby_star_checker",
        description="Check for nearby contaminating TIC stars.",
    )
    parser.add_argument("tic_id", type=int)
    parser.add_argument("--ra", type=float, required=True)
    parser.add_argument("--dec", type=float, required=True)
    parser.add_argument("--tmag", type=float, required=True)
    parser.add_argument("--radius", type=float, default=63.0)
    args = parser.parse_args(argv)

    result = check_nearby_stars(
        args.tic_id, args.ra, args.dec, args.tmag,
        radius_arcsec=args.radius,
    )
    print(format_nearby_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
