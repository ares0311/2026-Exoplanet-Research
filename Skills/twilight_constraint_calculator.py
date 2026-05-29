"""Check whether a planned observation falls within astronomical dark time.

Uses a simplified solar model:
  - Solar declination: delta = 23.45 * sin(2pi/365 * (doy - 81)) degrees
  - Sunrise hour (local solar): H = arccos(-tan(lat)*tan(delta)) / 15
  - Sunset hour: 12 - (H - 12) = 24 - H  (symmetric around noon)
  - Astronomical dark: [sunset + 1.5 h, sunrise - 1.5 h] next day

Public API
----------
TwilightResult(dark_start_local, dark_end_local, obs_start, obs_end,
               overlap_hours, in_dark_time, flag)
check_twilight_constraints(date_utc, obs_start_local, obs_end_local,
                           site_lat_deg, utc_offset_h) -> TwilightResult
format_twilight_result(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TwilightResult:
    dark_start_local: str    # "HH:MM"
    dark_end_local: str      # "HH:MM"
    obs_start: str
    obs_end: str
    overlap_hours: float
    in_dark_time: bool       # obs fully within dark time
    flag: str  # "OK", "PARTIAL_TWILIGHT", "IN_TWILIGHT"


def _day_of_year(date_utc: str) -> int:
    """Compute day of year from 'YYYY-MM-DD'."""
    try:
        parts = date_utc.split("-")
        year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
    except (IndexError, ValueError):
        return 80  # default ~spring equinox

    days_per_month = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    # leap year
    if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
        days_per_month[2] = 29
    return sum(days_per_month[:month]) + day


def _hhmm_to_hours(hhmm: str) -> float:
    """Convert 'HH:MM' string to fractional hours."""
    parts = hhmm.split(":")
    return int(parts[0]) + int(parts[1]) / 60.0


def _hours_to_hhmm(h: float) -> str:
    """Convert fractional hours to 'HH:MM' (mod 24)."""
    h = h % 24.0
    hh = int(h)
    mm = int(round((h - hh) * 60.0))
    if mm == 60:
        hh += 1
        mm = 0
    return f"{hh:02d}:{mm:02d}"


def _overlap_hours(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Compute overlap in hours between two intervals (handles wrap-around naively)."""
    start = max(a_start, b_start)
    end = min(a_end, b_end)
    return max(0.0, end - start)


def check_twilight_constraints(
    date_utc: str,
    obs_start_local: str,  # "HH:MM"
    obs_end_local: str,
    site_lat_deg: float,
    utc_offset_h: float = 0.0,
) -> TwilightResult:
    """Check if observation falls within astronomical dark time.

    Args:
        date_utc: Date string 'YYYY-MM-DD'.
        obs_start_local: Observation start time 'HH:MM' (local).
        obs_end_local: Observation end time 'HH:MM' (local).
        site_lat_deg: Site latitude in degrees.
        utc_offset_h: UTC offset in hours (positive = East).

    Returns:
        :class:`TwilightResult`.
    """
    doy = _day_of_year(date_utc)

    # Solar declination
    delta_rad = math.radians(23.45 * math.sin(math.radians(360.0 / 365.0 * (doy - 81))))
    lat_rad = math.radians(site_lat_deg)

    # Hour angle at sunrise/sunset: cos(H) = -tan(lat)*tan(delta)
    cos_h = -math.tan(lat_rad) * math.tan(delta_rad)

    # Clamp to handle polar regions
    cos_h = max(-1.0, min(1.0, cos_h))
    h_hours = math.degrees(math.acos(cos_h)) / 15.0  # hours from solar noon

    sunrise_solar = 12.0 - h_hours   # solar local time
    sunset_solar = 12.0 + h_hours

    # Convert solar time to clock time (UTC offset used as a proxy for longitude)
    # Simple approximation: local clock ≈ solar time + utc_offset (ignores eqn of time)
    sunrise_local = sunrise_solar  # already in local solar hours approx
    sunset_local = sunset_solar

    # Astronomical dark window (1.5 h after sunset to 1.5 h before sunrise next day)
    astro_dark_start = sunset_local + 1.5
    astro_dark_end = sunrise_local + 24.0 - 1.5  # treat as next-day sunrise

    obs_start_h = _hhmm_to_hours(obs_start_local)
    obs_end_h = _hhmm_to_hours(obs_end_local)

    # If obs end is before start, assume next-day (e.g. 23:00 to 02:00)
    if obs_end_h < obs_start_h:
        obs_end_h += 24.0

    dark_start_str = _hours_to_hhmm(astro_dark_start)
    dark_end_str = _hours_to_hhmm(astro_dark_end % 24.0)

    overlap = _overlap_hours(astro_dark_start, astro_dark_end, obs_start_h, obs_end_h)
    obs_duration = obs_end_h - obs_start_h

    in_dark = obs_start_h >= astro_dark_start and obs_end_h <= astro_dark_end

    if in_dark:
        flag = "OK"
    elif overlap > 0:
        flag = "PARTIAL_TWILIGHT"
    else:
        flag = "IN_TWILIGHT"

    return TwilightResult(
        dark_start_local=dark_start_str,
        dark_end_local=dark_end_str,
        obs_start=obs_start_local,
        obs_end=obs_end_local,
        overlap_hours=round(min(overlap, obs_duration), 3),
        in_dark_time=in_dark,
        flag=flag,
    )


def format_twilight_result(result: TwilightResult) -> str:
    """Format twilight result as Markdown."""
    lines = [
        "## Twilight Constraint Check",
        "",
        f"- Astronomical dark: {result.dark_start_local} – {result.dark_end_local} (local)",
        f"- Observation: {result.obs_start} – {result.obs_end} (local)",
        f"- Overlap: {result.overlap_hours:.2f} h",
        f"- Fully in dark time: {'Yes' if result.in_dark_time else 'No'}",
        f"- **Flag: {result.flag}**",
    ]
    return "\n".join(lines)


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--date", default="2026-06-01")
    p.add_argument("--obs-start", default="22:00")
    p.add_argument("--obs-end", default="01:00")
    p.add_argument("--lat", type=float, default=30.0)
    p.add_argument("--utc-offset", type=float, default=0.0)
    args = p.parse_args(argv)
    r = check_twilight_constraints(
        args.date, args.obs_start, args.obs_end, args.lat, args.utc_offset
    )
    print(format_twilight_result(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
