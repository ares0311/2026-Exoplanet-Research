"""Map sky coordinates to TESS observation sectors using an ecliptic model.

Uses a simplified ecliptic-coordinate sector model to predict which TESS
sectors observed a target.  Sectors are organised into two hemispheres
(north/south) with 13 sectors per year.  This module provides a fast,
dependency-free approximation; for precise sector coverage use
``tess_visibility_checker.py`` or lightkurve.

Public API
----------
SectorMapResult(ra_deg, dec_deg, ecliptic_lat_deg, n_sectors, sector_ids,
                years_observed, flag)
get_sector_map(ra_deg, dec_deg, *, n_years) -> SectorMapResult
format_sector_map(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SectorMapResult:
    ra_deg: float
    dec_deg: float
    ecliptic_lat_deg: float
    n_sectors: int
    sector_ids: tuple[int, ...]
    years_observed: tuple[int, ...]   # TESS year number (1-based) for each sector
    flag: str  # "OK" | "ECLIPTIC_PLANE" | "INVALID"


_DEG2RAD = math.pi / 180.0


def _equatorial_to_ecliptic_lat(ra_deg: float, dec_deg: float) -> float:
    """Approximate ecliptic latitude from equatorial coords (J2000)."""
    eps = 23.4393  # obliquity
    ra = ra_deg * _DEG2RAD
    dec = dec_deg * _DEG2RAD
    e = eps * _DEG2RAD
    sin_beta = math.sin(dec) * math.cos(e) - math.cos(dec) * math.sin(e) * math.sin(ra)
    return math.degrees(math.asin(max(-1.0, min(1.0, sin_beta))))


def _ecliptic_lon(ra_deg: float, dec_deg: float) -> float:
    """Approximate ecliptic longitude."""
    eps = 23.4393
    ra = ra_deg * _DEG2RAD
    dec = dec_deg * _DEG2RAD
    e = eps * _DEG2RAD
    num = math.sin(ra) * math.cos(e) + math.tan(dec) * math.sin(e)
    lam = math.atan2(num, math.cos(ra))
    return math.degrees(lam) % 360.0


def get_sector_map(
    ra_deg: float,
    dec_deg: float,
    *,
    n_years: int = 6,
) -> SectorMapResult:
    """Predict which TESS sectors observed a sky position.

    Uses a simplified model: each year has 13 sectors, each covering ~27 days
    and a ~24° ecliptic longitude slice.  Northern ecliptic hemisphere targets
    are observed in the northern viewing season (sectors 14–26 per cycle) and
    southern in the southern season (sectors 1–13 per cycle).  CVZ targets
    (|ecliptic lat| > 78°) are observed in every sector of their hemisphere.

    Args:
        ra_deg: Right ascension in degrees (J2000).
        dec_deg: Declination in degrees (J2000).
        n_years: Number of TESS years to search (default 6).

    Returns:
        :class:`SectorMapResult`.
    """
    if not (-90 <= dec_deg <= 90) or not (0 <= ra_deg < 360):
        return SectorMapResult(ra_deg, dec_deg, 0.0, 0, (), (), "INVALID")

    beta = _equatorial_to_ecliptic_lat(ra_deg, dec_deg)
    lam = _ecliptic_lon(ra_deg, dec_deg)

    # Ecliptic plane: |beta| < 6° → hard to observe
    ecliptic_plane = abs(beta) < 6.0

    # Each ecliptic longitude slice maps to a sector within a year
    # 13 sectors per year × 360°/13 ≈ 27.7° per sector
    deg_per_sector = 360.0 / 13.0

    sector_ids: list[int] = []
    years: list[int] = []

    for year in range(1, n_years + 1):
        # Northern hemisphere: sectors 14–26 in each cycle (odd years: 1-based)
        # Southern hemisphere: sectors 1–13
        # CVZ: all 13 sectors of the relevant hemisphere
        is_north = beta > 0

        if abs(beta) > 78.0:
            # Continuous viewing zone — visible all year
            if is_north:
                base = 14 + (year - 1) * 26
                for s in range(13):
                    sid = base + s
                    sector_ids.append(sid)
                    years.append(year)
            else:
                base = 1 + (year - 1) * 26
                for s in range(13):
                    sid = base + s
                    sector_ids.append(sid)
                    years.append(year)
        elif abs(beta) >= 6.0:
            # Observed in one sector per year based on ecliptic longitude
            s_idx = int(lam / deg_per_sector) % 13
            sid = 14 + (year - 1) * 26 + s_idx if is_north else 1 + (year - 1) * 26 + s_idx
            sector_ids.append(sid)
            years.append(year)

    flag = "ECLIPTIC_PLANE" if ecliptic_plane else "OK"
    return SectorMapResult(
        ra_deg=round(ra_deg, 4),
        dec_deg=round(dec_deg, 4),
        ecliptic_lat_deg=round(beta, 4),
        n_sectors=len(sector_ids),
        sector_ids=tuple(sector_ids),
        years_observed=tuple(years),
        flag=flag,
    )


def format_sector_map(result: SectorMapResult) -> str:
    """Format sector map result as Markdown."""
    lines = [
        "## TESS Sector Map",
        "",
        f"- RA: {result.ra_deg:.4f}°  Dec: {result.dec_deg:.4f}°",
        f"- Ecliptic latitude: {result.ecliptic_lat_deg:.2f}°",
        f"- Sectors predicted: {result.n_sectors}",
        f"- Sector IDs: {list(result.sector_ids[:20])}{'...' if result.n_sectors > 20 else ''}",
        f"- **Flag: {result.flag}**",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="tess_sector_map",
        description="Map sky coordinates to TESS sectors.",
    )
    parser.add_argument("ra_deg", type=float)
    parser.add_argument("dec_deg", type=float)
    parser.add_argument("--n-years", type=int, default=6)
    args = parser.parse_args(argv)

    result = get_sector_map(args.ra_deg, args.dec_deg, n_years=args.n_years)
    print(format_sector_map(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
