"""Check if a full transit (ingress + egress) is visible from a ground site before twilight.

Public API:
    GroundVisibilityResult  -- frozen dataclass
    check_ground_transit_visibility(transit_start_jd, transit_end_jd, sunset_jd,
                                     sunrise_jd, min_altitude_deg) -> GroundVisibilityResult
    format_ground_visibility_result(result) -> str
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class GroundVisibilityResult:
    transit_start_jd: float
    transit_end_jd: float
    transit_duration_hours: float
    overlap_start_jd: float
    overlap_end_jd: float
    overlap_hours: float
    fully_visible: bool
    flag: str


def check_ground_transit_visibility(
    transit_start_jd: float,
    transit_end_jd: float,
    sunset_jd: float,
    sunrise_jd: float,
    min_altitude_deg: float = 30.0,
) -> GroundVisibilityResult:
    if transit_end_jd <= transit_start_jd:
        return GroundVisibilityResult(
            transit_start_jd=transit_start_jd, transit_end_jd=transit_end_jd,
            transit_duration_hours=0.0, overlap_start_jd=0.0, overlap_end_jd=0.0,
            overlap_hours=0.0, fully_visible=False, flag="INVALID_TRANSIT_WINDOW",
        )
    if sunrise_jd <= sunset_jd:
        return GroundVisibilityResult(
            transit_start_jd=transit_start_jd, transit_end_jd=transit_end_jd,
            transit_duration_hours=0.0, overlap_start_jd=0.0, overlap_end_jd=0.0,
            overlap_hours=0.0, fully_visible=False, flag="INVALID_NIGHT_WINDOW",
        )
    if min_altitude_deg < 0 or min_altitude_deg >= 90:
        return GroundVisibilityResult(
            transit_start_jd=transit_start_jd, transit_end_jd=transit_end_jd,
            transit_duration_hours=0.0, overlap_start_jd=0.0, overlap_end_jd=0.0,
            overlap_hours=0.0, fully_visible=False, flag="INVALID_ALTITUDE",
        )
    transit_dur = (transit_end_jd - transit_start_jd) * 24.0
    overlap_start = max(transit_start_jd, sunset_jd)
    overlap_end = min(transit_end_jd, sunrise_jd)
    if overlap_end <= overlap_start:
        return GroundVisibilityResult(
            transit_start_jd=transit_start_jd, transit_end_jd=transit_end_jd,
            transit_duration_hours=transit_dur, overlap_start_jd=0.0, overlap_end_jd=0.0,
            overlap_hours=0.0, fully_visible=False, flag="NO_OVERLAP",
        )
    overlap_hours = (overlap_end - overlap_start) * 24.0
    fully_visible = (transit_start_jd >= sunset_jd) and (transit_end_jd <= sunrise_jd)
    flag = "FULLY_VISIBLE" if fully_visible else "PARTIAL"
    return GroundVisibilityResult(
        transit_start_jd=transit_start_jd,
        transit_end_jd=transit_end_jd,
        transit_duration_hours=transit_dur,
        overlap_start_jd=overlap_start,
        overlap_end_jd=overlap_end,
        overlap_hours=overlap_hours,
        fully_visible=fully_visible,
        flag=flag,
    )


def format_ground_visibility_result(result: GroundVisibilityResult) -> str:
    lines = [
        "## Ground Transit Visibility",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Transit Duration (hr) | {result.transit_duration_hours:.3f} |",
        f"| Overlap (hr) | {result.overlap_hours:.3f} |",
        f"| Fully Visible | {result.fully_visible} |",
        f"| Flag | {result.flag} |",
    ]
    return "\n".join(lines)


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Check ground-based transit visibility.")
    parser.add_argument("transit_start_jd", type=float)
    parser.add_argument("transit_end_jd", type=float)
    parser.add_argument("sunset_jd", type=float)
    parser.add_argument("sunrise_jd", type=float)
    parser.add_argument("--min-altitude", type=float, default=30.0)
    args = parser.parse_args()
    result = check_ground_transit_visibility(
        args.transit_start_jd, args.transit_end_jd,
        args.sunset_jd, args.sunrise_jd, args.min_altitude,
    )
    print(format_ground_visibility_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
