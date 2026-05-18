"""Check TESS visibility for a sky position across sectors.

Uses a simplified ecliptic-coordinate model to determine which TESS sectors
(1–96) observe a given RA/Dec. Each TESS sector covers ~27 days and observes
a 24°×96° field of view near the ecliptic poles.

Public API
----------
TESSVisibilityResult(ra_deg, dec_deg, n_sectors_visible, sector_list,
                     continuous_viewing_zone, flag)
check_tess_visibility(ra_deg, dec_deg, *, max_sector) -> TESSVisibilityResult
format_visibility_result(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TESSVisibilityResult:
    ra_deg: float
    dec_deg: float
    n_sectors_visible: int
    sector_list: tuple[int, ...]
    continuous_viewing_zone: bool    # |ecliptic lat| > 78° → CVZ
    flag: str                        # "OK", "INVALID"


def _equatorial_to_ecliptic(ra_deg: float, dec_deg: float) -> tuple[float, float]:
    """Convert RA/Dec (J2000) to ecliptic latitude/longitude (degrees)."""
    eps = math.radians(23.439)
    ra = math.radians(ra_deg)
    dec = math.radians(dec_deg)
    sin_beta = math.sin(dec) * math.cos(eps) - math.cos(dec) * math.sin(eps) * math.sin(ra)
    beta = math.degrees(math.asin(max(-1.0, min(1.0, sin_beta))))
    cos_lam = math.cos(dec) * math.cos(ra) / math.cos(math.radians(beta))
    lam = math.degrees(math.acos(max(-1.0, min(1.0, cos_lam))))
    if math.sin(dec) * math.sin(eps) + math.cos(dec) * math.cos(eps) * math.sin(ra) < 0:
        lam = 360.0 - lam
    return lam, beta


def check_tess_visibility(
    ra_deg: float,
    dec_deg: float,
    *,
    max_sector: int = 96,
) -> TESSVisibilityResult:
    """Check TESS visibility for a sky position.

    Uses an ecliptic-latitude model: TESS observes from |β| ≈ 6° to 90° in
    alternating hemispheres. Each 13-sector year covers one hemisphere.

    Args:
        ra_deg: Right ascension in degrees [0, 360).
        dec_deg: Declination in degrees [-90, 90].
        max_sector: Maximum sector number to consider.

    Returns:
        :class:`TESSVisibilityResult`.
    """
    if not (0 <= ra_deg < 360) or not (-90 <= dec_deg <= 90):
        return TESSVisibilityResult(ra_deg, dec_deg, 0, (), False, "INVALID")

    lam, beta = _equatorial_to_ecliptic(ra_deg, dec_deg)
    abs_beta = abs(beta)
    cvz = abs_beta > 78.0

    # Simplified model: each year = 13 sectors, alternating hemispheres
    # Northern ecliptic (β > 0) → odd years; Southern (β < 0) → even years
    # Sectors per year visible to target ≈ proportional to ecliptic latitude
    sectors: list[int] = []

    for s in range(1, max_sector + 1):
        year = (s - 1) // 13
        hemisphere = "N" if year % 2 == 0 else "S"
        target_hemisphere = "N" if beta >= 0 else "S"

        # Minimum ecliptic latitude for visibility in this sector
        # TESS camera footprint: each sector spans ~6°–90° in |β|
        # Sectors within a year rotate in ecliptic longitude by ~27.7°/sector
        sector_in_year = (s - 1) % 13
        lam_center = (sector_in_year * (360.0 / 13)) % 360.0
        lam_diff = abs(((lam - lam_center + 180) % 360) - 180)

        if hemisphere == target_hemisphere and abs_beta > 6.0 and lam_diff < 96.0:
            sectors.append(s)

    return TESSVisibilityResult(
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        n_sectors_visible=len(sectors),
        sector_list=tuple(sectors[:20]),  # cap at 20 for readability
        continuous_viewing_zone=cvz,
        flag="OK",
    )


def format_visibility_result(result: TESSVisibilityResult) -> str:
    """Format TESS visibility result as Markdown."""
    lines = [
        "## TESS Visibility",
        "",
        f"- RA: {result.ra_deg:.4f}°",
        f"- Dec: {result.dec_deg:.4f}°",
        f"- Sectors visible: {result.n_sectors_visible}",
        f"- Continuous viewing zone: {'Yes' if result.continuous_viewing_zone else 'No'}",
        f"- **Flag: {result.flag}**",
    ]
    if result.sector_list:
        sectors_str = ", ".join(str(s) for s in result.sector_list)
        lines.append(f"- Sectors (first 20): {sectors_str}")
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="tess_visibility_checker",
        description="Check TESS visibility for a sky position.",
    )
    parser.add_argument("ra_deg", type=float)
    parser.add_argument("dec_deg", type=float)
    parser.add_argument("--max-sector", type=int, default=96)
    args = parser.parse_args(argv)

    result = check_tess_visibility(
        args.ra_deg, args.dec_deg, max_sector=args.max_sector,
    )
    print(format_visibility_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
