"""Convert sky coordinates between equatorial, ecliptic, and galactic systems.

Uses stdlib math only with fixed rotation matrix constants.  Equatorial
(J2000 RA/Dec) ↔ ecliptic uses the mean obliquity ε = 23.4393°.
Equatorial ↔ galactic uses the IAU 1958 constants.

Public API
----------
CoordinateResult(ra_deg, dec_deg, ecl_lon_deg, ecl_lat_deg,
                 gal_lon_deg, gal_lat_deg, flag)
convert_coordinates(ra_deg, dec_deg) -> CoordinateResult
format_coordinate_result(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# Mean obliquity of the ecliptic (J2000), degrees
_OBLIQUITY_DEG = 23.4393

# IAU 1958 galactic pole in J2000 equatorial coordinates
_POLE_RA_DEG = 192.85948   # RA of north galactic pole
_POLE_DEC_DEG = 27.12825   # Dec of north galactic pole
_ASC_NODE_DEG = 32.93192   # ascending node of galactic equator on equatorial equator


@dataclass(frozen=True)
class CoordinateResult:
    ra_deg: float
    dec_deg: float
    ecl_lon_deg: float | None    # ecliptic longitude λ
    ecl_lat_deg: float | None    # ecliptic latitude β
    gal_lon_deg: float | None    # galactic longitude l
    gal_lat_deg: float | None    # galactic latitude b
    flag: str  # "OK" | "INVALID"


def _eq_to_ecl(ra_deg: float, dec_deg: float) -> tuple[float, float]:
    """Convert equatorial (J2000) to ecliptic coordinates."""
    eps = math.radians(_OBLIQUITY_DEG)
    ra = math.radians(ra_deg)
    dec = math.radians(dec_deg)

    sin_lat = (math.sin(dec) * math.cos(eps)
               - math.cos(dec) * math.sin(eps) * math.sin(ra))
    lat = math.asin(max(-1.0, min(1.0, sin_lat)))

    y = math.sin(ra) * math.cos(eps) + math.tan(dec) * math.sin(eps)
    x = math.cos(ra)
    lon = math.atan2(y, x)

    return math.degrees(lon) % 360.0, math.degrees(lat)


def _eq_to_gal(ra_deg: float, dec_deg: float) -> tuple[float, float]:
    """Convert equatorial (J2000) to galactic coordinates (IAU 1958)."""
    ra = math.radians(ra_deg)
    dec = math.radians(dec_deg)
    pole_ra = math.radians(_POLE_RA_DEG)
    pole_dec = math.radians(_POLE_DEC_DEG)
    sin_b = (math.sin(dec) * math.sin(pole_dec)
             + math.cos(dec) * math.cos(pole_dec) * math.cos(ra - pole_ra))
    b = math.asin(max(-1.0, min(1.0, sin_b)))

    y = math.cos(dec) * math.sin(ra - pole_ra)
    x = (math.sin(dec) * math.cos(pole_dec)
         - math.cos(dec) * math.sin(pole_dec) * math.cos(ra - pole_ra))
    lon = (math.degrees(math.atan2(y, x)) + (270.0 - _ASC_NODE_DEG)) % 360.0

    return lon, math.degrees(b)


def convert_coordinates(ra_deg: float, dec_deg: float) -> CoordinateResult:
    """Convert J2000 RA/Dec to ecliptic and galactic coordinates.

    Args:
        ra_deg: Right ascension (degrees, J2000), in [0, 360).
        dec_deg: Declination (degrees, J2000), in [-90, 90].

    Returns:
        :class:`CoordinateResult`.
    """
    if not (math.isfinite(ra_deg) and math.isfinite(dec_deg)):
        return CoordinateResult(ra_deg, dec_deg, None, None, None, None, "INVALID")
    if not (-90.0 <= dec_deg <= 90.0):
        return CoordinateResult(ra_deg, dec_deg, None, None, None, None, "INVALID")

    ra_norm = ra_deg % 360.0

    ecl_lon, ecl_lat = _eq_to_ecl(ra_norm, dec_deg)
    gal_lon, gal_lat = _eq_to_gal(ra_norm, dec_deg)

    return CoordinateResult(
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        ecl_lon_deg=round(ecl_lon, 6),
        ecl_lat_deg=round(ecl_lat, 6),
        gal_lon_deg=round(gal_lon, 6),
        gal_lat_deg=round(gal_lat, 6),
        flag="OK",
    )


def format_coordinate_result(result: CoordinateResult) -> str:
    """Format coordinate conversion result as Markdown."""
    lines = [
        "## Target Coordinates Converter",
        "",
        f"- RA (J2000): {result.ra_deg}°",
        f"- Dec (J2000): {result.dec_deg}°",
        f"- **Ecliptic lon/lat: {result.ecl_lon_deg}° / {result.ecl_lat_deg}°**",
        f"- **Galactic l/b: {result.gal_lon_deg}° / {result.gal_lat_deg}°**",
        f"- **Flag: {result.flag}**",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="target_coordinates_converter",
        description="Convert RA/Dec to ecliptic and galactic coordinates.",
    )
    parser.add_argument("ra_deg", type=float)
    parser.add_argument("dec_deg", type=float)
    args = parser.parse_args(argv)

    result = convert_coordinates(args.ra_deg, args.dec_deg)
    print(format_coordinate_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
