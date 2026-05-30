"""Assign a TESS camera number to a target's RA/Dec using ecliptic-coordinate sector geometry.

TESS has 4 cameras per sector arranged in a 24°x96° strip along a line of ecliptic longitude.
Each camera covers a 24°x24° square. Camera 1 is nearest the ecliptic, camera 4 is near the pole.

Public API:
    CameraAssignmentResult  -- frozen dataclass
    assign_tess_camera(ra_deg, dec_deg) -> CameraAssignmentResult
    format_camera_assignment(result) -> str
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

_OBL = math.radians(23.439)  # IAU obliquity


def _ecliptic_coords(ra_deg: float, dec_deg: float) -> tuple[float, float]:
    ra = math.radians(ra_deg)
    dec = math.radians(dec_deg)
    sin_lat = math.sin(dec) * math.cos(_OBL) - math.cos(dec) * math.sin(_OBL) * math.sin(ra)
    lat = math.degrees(math.asin(max(-1.0, min(1.0, sin_lat))))
    lon_num = math.sin(ra) * math.cos(_OBL) + math.tan(dec) * math.sin(_OBL)
    lon = math.degrees(math.atan2(lon_num, math.cos(ra))) % 360.0
    return lon, lat


@dataclass(frozen=True)
class CameraAssignmentResult:
    ra_deg: float
    dec_deg: float
    ecliptic_lon_deg: float
    ecliptic_lat_deg: float
    camera: int
    abs_ecliptic_lat_deg: float
    flag: str


def assign_tess_camera(ra_deg: float, dec_deg: float) -> CameraAssignmentResult:
    if not (0.0 <= ra_deg < 360.0):
        return CameraAssignmentResult(
            ra_deg=ra_deg, dec_deg=dec_deg, ecliptic_lon_deg=0.0, ecliptic_lat_deg=0.0,
            camera=0, abs_ecliptic_lat_deg=0.0, flag="INVALID_RA",
        )
    if not (-90.0 <= dec_deg <= 90.0):
        return CameraAssignmentResult(
            ra_deg=ra_deg, dec_deg=dec_deg, ecliptic_lon_deg=0.0, ecliptic_lat_deg=0.0,
            camera=0, abs_ecliptic_lat_deg=0.0, flag="INVALID_DEC",
        )
    elon, elat = _ecliptic_coords(ra_deg, dec_deg)
    abs_lat = abs(elat)
    # TESS cameras 1-4 cover |elat| from ~6° to ~90° in four 24° strips
    # Camera 1: 6°-30°, Camera 2: 30°-54°, Camera 3: 54°-78°, Camera 4: 78°-90°
    # (approximate model; actual sky coverage is more complex)
    if abs_lat < 6.0:
        camera = 0  # in ecliptic gap, not covered
        flag = "ECLIPTIC_GAP"
    elif abs_lat < 30.0:
        camera = 1
        flag = "OK"
    elif abs_lat < 54.0:
        camera = 2
        flag = "OK"
    elif abs_lat < 78.0:
        camera = 3
        flag = "OK"
    else:
        camera = 4
        flag = "OK"
    return CameraAssignmentResult(
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        ecliptic_lon_deg=elon,
        ecliptic_lat_deg=elat,
        camera=camera,
        abs_ecliptic_lat_deg=abs_lat,
        flag=flag,
    )


def format_camera_assignment(result: CameraAssignmentResult) -> str:
    lines = [
        "## TESS Camera Assignment",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| RA (deg) | {result.ra_deg:.4f} |",
        f"| Dec (deg) | {result.dec_deg:.4f} |",
        f"| Ecliptic Lon (deg) | {result.ecliptic_lon_deg:.4f} |",
        f"| Ecliptic Lat (deg) | {result.ecliptic_lat_deg:.4f} |",
        f"| |Ecliptic Lat| (deg) | {result.abs_ecliptic_lat_deg:.4f} |",
        f"| Camera | {result.camera} |",
        f"| Flag | {result.flag} |",
    ]
    return "\n".join(lines)


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Assign TESS camera to RA/Dec.")
    parser.add_argument("ra_deg", type=float)
    parser.add_argument("dec_deg", type=float)
    args = parser.parse_args()
    result = assign_tess_camera(args.ra_deg, args.dec_deg)
    print(format_camera_assignment(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
