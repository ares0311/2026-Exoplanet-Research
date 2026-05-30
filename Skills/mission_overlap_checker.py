"""Check whether a target has archival coverage from TESS, Kepler, and/or K2.

Public API:
    MissionOverlapResult  -- frozen dataclass
    check_mission_overlap(ra_deg, dec_deg) -> MissionOverlapResult
    format_mission_overlap(result) -> str
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class MissionOverlapResult:
    ra_deg: float
    dec_deg: float
    ecliptic_lat_deg: float
    in_kepler_field: bool
    in_k2_accessible: bool
    in_tess_cvz: bool
    missions: list[str]
    flag: str


def _ecliptic_lat(ra_deg: float, dec_deg: float) -> float:
    ra = math.radians(ra_deg)
    dec = math.radians(dec_deg)
    eps = math.radians(23.439)
    sin_lat = math.sin(dec) * math.cos(eps) - math.cos(dec) * math.sin(eps) * math.sin(ra)
    return math.degrees(math.asin(max(-1.0, min(1.0, sin_lat))))


def check_mission_overlap(ra_deg: float, dec_deg: float) -> MissionOverlapResult:
    if not (0.0 <= ra_deg < 360.0):
        return MissionOverlapResult(
            ra_deg=ra_deg, dec_deg=dec_deg, ecliptic_lat_deg=0.0,
            in_kepler_field=False, in_k2_accessible=False, in_tess_cvz=False,
            missions=[], flag="INVALID_RA",
        )
    if not (-90.0 <= dec_deg <= 90.0):
        return MissionOverlapResult(
            ra_deg=ra_deg, dec_deg=dec_deg, ecliptic_lat_deg=0.0,
            in_kepler_field=False, in_k2_accessible=False, in_tess_cvz=False,
            missions=[], flag="INVALID_DEC",
        )
    ecl_lat = _ecliptic_lat(ra_deg, dec_deg)
    in_kepler = (
        290.0 <= ra_deg <= 305.0 and 36.0 <= dec_deg <= 52.0
    )
    in_k2 = abs(ecl_lat) < 5.0
    in_cvz = abs(dec_deg) > 78.0 or (abs(dec_deg) < -78.0)
    missions: list[str] = ["TESS"]
    if in_kepler:
        missions.append("Kepler")
    if in_k2:
        missions.append("K2")
    if in_cvz:
        missions.append("TESS-CVZ")
    flag = "MULTI_MISSION" if len(missions) > 1 else "TESS_ONLY"
    return MissionOverlapResult(
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        ecliptic_lat_deg=ecl_lat,
        in_kepler_field=in_kepler,
        in_k2_accessible=in_k2,
        in_tess_cvz=in_cvz,
        missions=missions,
        flag=flag,
    )


def format_mission_overlap(result: MissionOverlapResult) -> str:
    lines = [
        "## Mission Overlap Check",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| RA (deg) | {result.ra_deg:.4f} |",
        f"| Dec (deg) | {result.dec_deg:.4f} |",
        f"| Ecliptic Lat (deg) | {result.ecliptic_lat_deg:.4f} |",
        f"| In Kepler Field | {result.in_kepler_field} |",
        f"| K2 Accessible | {result.in_k2_accessible} |",
        f"| TESS CVZ | {result.in_tess_cvz} |",
        f"| Missions | {', '.join(result.missions)} |",
        f"| Flag | {result.flag} |",
    ]
    return "\n".join(lines)


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Check mission overlap for a target.")
    parser.add_argument("ra_deg", type=float)
    parser.add_argument("dec_deg", type=float)
    args = parser.parse_args()
    result = check_mission_overlap(args.ra_deg, args.dec_deg)
    print(format_mission_overlap(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
