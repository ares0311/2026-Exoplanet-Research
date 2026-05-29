"""Plan multi-site telescope network coverage for a transit observation.

Uses the same simplified solar twilight model as twilight_constraint_calculator:
  - Solar declination: 23.45 * sin(2pi/365 * (doy - 81)) degrees
  - Astronomical dark window per site: [sunset + 1.5 h, next sunrise - 1.5 h]
  - Transit UTC window converted to local solar time via longitude offset

Public API
----------
SiteObservability(site_name, lat_deg, lon_deg, can_observe, overlap_hours)
NetworkPlanResult(n_sites_total, n_sites_viable, viable_sites,
                  coverage_fraction, flag)
plan_network_coverage(sites, transit_start_utc, transit_end_utc, date_utc)
    -> NetworkPlanResult
format_network_plan(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SiteObservability:
    site_name: str
    lat_deg: float
    lon_deg: float
    can_observe: bool
    overlap_hours: float


@dataclass(frozen=True)
class NetworkPlanResult:
    n_sites_total: int
    n_sites_viable: int
    viable_sites: tuple[SiteObservability, ...]
    coverage_fraction: float
    flag: str  # "FULL_COVERAGE", "PARTIAL_COVERAGE", "NO_COVERAGE"


def _day_of_year(date_utc: str) -> int:
    """Compute day of year from 'YYYY-MM-DD'."""
    try:
        parts = date_utc.split("-")
        year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
    except (IndexError, ValueError):
        return 80
    days_per_month = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
        days_per_month[2] = 29
    return sum(days_per_month[:month]) + day


def _parse_datetime_utc(dt_str: str) -> float:
    """Parse 'YYYY-MM-DD HH:MM' into fractional UTC hours (0–48 range)."""
    try:
        date_part, time_part = dt_str.split()
        h, m = time_part.split(":")
        return int(h) + int(m) / 60.0
    except (ValueError, AttributeError):
        return 22.0


def _dark_window_local(doy: int, lat_deg: float, lon_deg: float) -> tuple[float, float]:
    """Compute astronomical dark window in local solar hours.

    Returns (dark_start_h, dark_end_h) where dark_end_h may exceed 24.
    """
    delta_rad = math.radians(23.45 * math.sin(math.radians(360.0 / 365.0 * (doy - 81))))
    lat_rad = math.radians(lat_deg)

    cos_h = -math.tan(lat_rad) * math.tan(delta_rad)
    cos_h = max(-1.0, min(1.0, cos_h))
    h_hours = math.degrees(math.acos(cos_h)) / 15.0

    sunrise_local = 12.0 - h_hours
    sunset_local = 12.0 + h_hours

    dark_start = sunset_local + 1.5
    dark_end = sunrise_local + 24.0 - 1.5
    return dark_start, dark_end


def _utc_to_local_solar(utc_h: float, lon_deg: float) -> float:
    """Convert UTC hour to local solar time (approximate)."""
    return utc_h + lon_deg / 15.0


def plan_network_coverage(
    sites: list[dict],
    transit_start_utc: str,
    transit_end_utc: str,
    date_utc: str,
) -> NetworkPlanResult:
    """Plan multi-site transit coverage.

    Args:
        sites: List of dicts with keys ``name`` (str), ``lat_deg`` (float),
            ``lon_deg`` (float).
        transit_start_utc: Transit ingress UTC 'YYYY-MM-DD HH:MM'.
        transit_end_utc: Transit egress UTC 'YYYY-MM-DD HH:MM'.
        date_utc: Observation date 'YYYY-MM-DD'.

    Returns:
        :class:`NetworkPlanResult`.
    """
    doy = _day_of_year(date_utc)

    transit_start_h = _parse_datetime_utc(transit_start_utc)
    transit_end_h = _parse_datetime_utc(transit_end_utc)
    if transit_end_h < transit_start_h:
        transit_end_h += 24.0
    transit_duration = transit_end_h - transit_start_h

    observabilities: list[SiteObservability] = []
    for site in sites:
        name = str(site.get("name", "Unknown"))
        lat = float(site.get("lat_deg", 0.0))
        lon = float(site.get("lon_deg", 0.0))

        dark_start_local, dark_end_local = _dark_window_local(doy, lat, lon)

        # Convert transit window to local solar time for this site
        transit_start_local = _utc_to_local_solar(transit_start_h, lon)
        transit_end_local = _utc_to_local_solar(transit_end_h, lon)

        # Compute overlap of transit window with dark window
        overlap_start = max(dark_start_local, transit_start_local)
        overlap_end = min(dark_end_local, transit_end_local)
        overlap = max(0.0, overlap_end - overlap_start)

        can_observe = overlap > 0.0
        observabilities.append(
            SiteObservability(
                site_name=name,
                lat_deg=lat,
                lon_deg=lon,
                can_observe=can_observe,
                overlap_hours=round(overlap, 3),
            )
        )

    viable = tuple(s for s in observabilities if s.can_observe)
    n_viable = len(viable)

    # Coverage fraction: fraction of transit covered by at least one site
    if transit_duration <= 0 or n_viable == 0:
        coverage_fraction = 0.0
    else:
        # Best single-site overlap as proxy for coverage fraction
        best_overlap = max(s.overlap_hours for s in viable)
        coverage_fraction = min(1.0, best_overlap / transit_duration)

    if coverage_fraction >= 1.0:
        flag = "FULL_COVERAGE"
    elif coverage_fraction > 0.0:
        flag = "PARTIAL_COVERAGE"
    else:
        flag = "NO_COVERAGE"

    return NetworkPlanResult(
        n_sites_total=len(sites),
        n_sites_viable=n_viable,
        viable_sites=viable,
        coverage_fraction=round(coverage_fraction, 3),
        flag=flag,
    )


def format_network_plan(result: NetworkPlanResult) -> str:
    """Format network plan as Markdown."""
    lines = [
        "## Telescope Network Plan",
        "",
        f"- Sites total: {result.n_sites_total}",
        f"- Sites viable: {result.n_sites_viable}",
        f"- Coverage fraction: {result.coverage_fraction:.1%}",
        f"- **Flag: {result.flag}**",
    ]
    if result.viable_sites:
        lines += [
            "",
            "| Site | Lat | Lon | Overlap (h) |",
            "|------|-----|-----|-------------|",
        ]
        for s in result.viable_sites:
            lines.append(
                f"| {s.site_name} | {s.lat_deg:.1f} | {s.lon_deg:.1f}"
                f" | {s.overlap_hours:.2f} |"
            )
    return "\n".join(lines)


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("sites_json", help="JSON file with list of site dicts")
    p.add_argument("--transit-start", default="2026-06-01 22:00")
    p.add_argument("--transit-end", default="2026-06-02 00:30")
    p.add_argument("--date", default="2026-06-01")
    args = p.parse_args(argv)

    with open(args.sites_json) as fh:
        sites = json.load(fh)

    r = plan_network_coverage(sites, args.transit_start, args.transit_end, args.date)
    print(format_network_plan(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
