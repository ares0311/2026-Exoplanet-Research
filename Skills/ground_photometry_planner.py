"""Plan a ground-based photometry observation given site and transit parameters.

Computes observation start/end with 1x-duration baseline padding.
Uses altitude ≈ 90 - |lat - dec| approximation (no spherical trig).

Public API
----------
GroundPhotPlan(obs_start_utc, obs_end_utc, duration_with_baseline_hours,
               min_altitude_deg, max_airmass, flag)
plan_ground_photometry(target_ra_deg, target_dec_deg, transit_mid_bjd,
                       duration_hours, site_lat_deg, site_lon_deg,
                       site_elev_m, date_utc) -> GroundPhotPlan
format_ground_phot_plan(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class GroundPhotPlan:
    obs_start_utc: str
    obs_end_utc: str
    duration_with_baseline_hours: float
    min_altitude_deg: float
    max_airmass: float
    flag: str  # "OK" | "LOW_ALTITUDE" | "UNOBSERVABLE"


def _bjd_to_utc_approx(bjd: float) -> str:
    """Convert BJD to approximate UTC string (ignores ~8 min light-travel)."""
    jd_int = int(bjd + 0.5)
    frac = bjd + 0.5 - jd_int
    a = jd_int + 32044
    b = (4 * a + 3) // 146097
    c = a - (146097 * b) // 4
    d = (4 * c + 3) // 1461
    e = c - (1461 * d) // 4
    m = (5 * e + 2) // 153
    day = e - (153 * m + 2) // 5 + 1
    month = m + 3 - 12 * (m // 10)
    year = 100 * b + d - 4800 + m // 10
    total_minutes = frac * 1440.0
    hour = int(total_minutes // 60) % 24
    minute = int(total_minutes % 60)
    second = int((total_minutes * 60) % 60)
    return f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:{second:02d}Z"


def plan_ground_photometry(
    target_ra_deg: float,
    target_dec_deg: float,
    transit_mid_bjd: float,
    duration_hours: float,
    site_lat_deg: float,
    site_lon_deg: float = 0.0,
    site_elev_m: float = 0.0,
    date_utc: str = "",
    *,
    min_observable_alt_deg: float = 20.0,
) -> GroundPhotPlan:
    """Plan a ground-based photometry observation.

    Args:
        target_ra_deg: Target right ascension in degrees.
        target_dec_deg: Target declination in degrees.
        transit_mid_bjd: Transit mid-time in BJD.
        duration_hours: Transit duration in hours.
        site_lat_deg: Site latitude in degrees.
        site_lon_deg: Site longitude in degrees (East positive).
        site_elev_m: Site elevation in metres (informational only).
        date_utc: Observation date string (informational).
        min_observable_alt_deg: Minimum altitude for OK flag.

    Returns:
        :class:`GroundPhotPlan`.
    """
    # Suppress unused-arg warnings — stored for future use
    _ = site_lon_deg, site_elev_m, date_utc

    if duration_hours <= 0:
        return GroundPhotPlan(
            obs_start_utc="",
            obs_end_utc="",
            duration_with_baseline_hours=0.0,
            min_altitude_deg=0.0,
            max_airmass=99.9,
            flag="UNOBSERVABLE",
        )

    half_dur_days = duration_hours / 48.0  # half-duration in days
    baseline_days = duration_hours / 24.0   # 1x duration each side
    obs_start_bjd = transit_mid_bjd - half_dur_days - baseline_days
    obs_end_bjd = transit_mid_bjd + half_dur_days + baseline_days

    total_hours = (obs_end_bjd - obs_start_bjd) * 24.0

    # Simplified altitude: max altitude approximation at meridian transit
    alt_deg = 90.0 - abs(site_lat_deg - target_dec_deg)
    alt_deg = max(-90.0, min(90.0, alt_deg))

    if alt_deg <= 0:
        airmass = 99.9
        flag = "UNOBSERVABLE"
    else:
        airmass = round(min(1.0 / math.sin(math.radians(alt_deg)), 99.9), 3)
        flag = "OK" if alt_deg >= min_observable_alt_deg else "LOW_ALTITUDE"

    return GroundPhotPlan(
        obs_start_utc=_bjd_to_utc_approx(obs_start_bjd),
        obs_end_utc=_bjd_to_utc_approx(obs_end_bjd),
        duration_with_baseline_hours=round(total_hours, 3),
        min_altitude_deg=round(alt_deg, 2),
        max_airmass=airmass,
        flag=flag,
    )


def format_ground_phot_plan(result: GroundPhotPlan) -> str:
    """Format ground photometry plan as Markdown."""
    lines = [
        "## Ground Photometry Plan",
        "",
        f"- Obs start (UTC): {result.obs_start_utc}",
        f"- Obs end (UTC): {result.obs_end_utc}",
        f"- Total duration (with baseline): {result.duration_with_baseline_hours:.2f} h",
        f"- Min altitude: {result.min_altitude_deg:.1f}°",
        f"- Max airmass: {result.max_airmass:.3f}",
        f"- **Flag: {result.flag}**",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="ground_photometry_planner",
        description="Plan a ground-based photometry observation.",
    )
    parser.add_argument("target_ra_deg", type=float)
    parser.add_argument("target_dec_deg", type=float)
    parser.add_argument("transit_mid_bjd", type=float)
    parser.add_argument("duration_hours", type=float)
    parser.add_argument("site_lat_deg", type=float)
    parser.add_argument("site_lon_deg", type=float)
    parser.add_argument("--site-elev-m", type=float, default=0.0)
    parser.add_argument("--date-utc", type=str, default="")
    parser.add_argument("--min-observable-alt-deg", type=float, default=20.0)
    args = parser.parse_args(argv)

    result = plan_ground_photometry(
        args.target_ra_deg,
        args.target_dec_deg,
        args.transit_mid_bjd,
        args.duration_hours,
        args.site_lat_deg,
        args.site_lon_deg,
        args.site_elev_m,
        args.date_utc,
        min_observable_alt_deg=args.min_observable_alt_deg,
    )
    print(format_ground_phot_plan(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
