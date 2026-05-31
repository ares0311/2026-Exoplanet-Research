"""Validate sky coordinates (RA, Dec) and compute derived properties."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SkyCoordinateResult:
    ra_deg: float
    dec_deg: float
    ecliptic_lat_deg: float
    galactic_lat_deg: float
    in_continuous_viewing_zone: bool
    flag: str


# Obliquity of ecliptic (J2000)
_EPS = math.radians(23.4393)

# IAU 1958 galactic pole (J2000 approx): RA=192.8595°, Dec=27.1284°, l_NCP=122.9320°
_GP_RA = math.radians(192.8595)
_GP_DEC = math.radians(27.1284)


def _ecliptic_latitude(ra_deg: float, dec_deg: float) -> float:
    ra = math.radians(ra_deg)
    dec = math.radians(dec_deg)
    sin_b = math.sin(dec) * math.cos(_EPS) - math.cos(dec) * math.sin(_EPS) * math.sin(ra)
    return math.degrees(math.asin(max(-1.0, min(1.0, sin_b))))


def _galactic_latitude(ra_deg: float, dec_deg: float) -> float:
    ra = math.radians(ra_deg)
    dec = math.radians(dec_deg)
    sin_b = (
        math.sin(dec) * math.sin(_GP_DEC)
        + math.cos(dec) * math.cos(_GP_DEC) * math.cos(ra - _GP_RA)
    )
    return math.degrees(math.asin(max(-1.0, min(1.0, sin_b))))


def validate_sky_coordinates(
    ra_deg: float,
    dec_deg: float,
) -> SkyCoordinateResult:
    """
    Validate RA/Dec and compute ecliptic/galactic latitudes.

    TESS continuous viewing zones (CVZ): |ecliptic latitude| > 78°.
    """
    if not math.isfinite(ra_deg):
        return SkyCoordinateResult(
            ra_deg=ra_deg, dec_deg=dec_deg,
            ecliptic_lat_deg=float("nan"), galactic_lat_deg=float("nan"),
            in_continuous_viewing_zone=False, flag="INVALID_RA",
        )
    if not math.isfinite(dec_deg):
        return SkyCoordinateResult(
            ra_deg=ra_deg, dec_deg=dec_deg,
            ecliptic_lat_deg=float("nan"), galactic_lat_deg=float("nan"),
            in_continuous_viewing_zone=False, flag="INVALID_DEC",
        )
    if ra_deg < 0.0 or ra_deg >= 360.0:
        return SkyCoordinateResult(
            ra_deg=ra_deg, dec_deg=dec_deg,
            ecliptic_lat_deg=float("nan"), galactic_lat_deg=float("nan"),
            in_continuous_viewing_zone=False, flag="RA_OUT_OF_RANGE",
        )
    if dec_deg < -90.0 or dec_deg > 90.0:
        return SkyCoordinateResult(
            ra_deg=ra_deg, dec_deg=dec_deg,
            ecliptic_lat_deg=float("nan"), galactic_lat_deg=float("nan"),
            in_continuous_viewing_zone=False, flag="DEC_OUT_OF_RANGE",
        )

    elat = _ecliptic_latitude(ra_deg, dec_deg)
    blat = _galactic_latitude(ra_deg, dec_deg)
    in_cvz = abs(elat) > 78.0

    return SkyCoordinateResult(
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        ecliptic_lat_deg=round(elat, 4),
        galactic_lat_deg=round(blat, 4),
        in_continuous_viewing_zone=in_cvz,
        flag="OK",
    )


def format_coordinate_validation(r: SkyCoordinateResult) -> str:
    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| RA (deg) | {r.ra_deg:.4f} |\n"
        f"| Dec (deg) | {r.dec_deg:.4f} |\n"
        f"| Ecliptic lat (deg) | {r.ecliptic_lat_deg:.4f} |\n"
        f"| Galactic lat (deg) | {r.galactic_lat_deg:.4f} |\n"
        f"| In CVZ | {r.in_continuous_viewing_zone} |\n"
        f"| Flag | {r.flag} |\n"
    )


def _cli() -> int:
    p = argparse.ArgumentParser(description="Validate sky coordinates.")
    p.add_argument("ra_deg", type=float)
    p.add_argument("dec_deg", type=float)
    args = p.parse_args()
    r = validate_sky_coordinates(args.ra_deg, args.dec_deg)
    print(format_coordinate_validation(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
