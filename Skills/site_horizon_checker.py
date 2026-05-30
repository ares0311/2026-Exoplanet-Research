"""Check whether a target is observable given a site horizon limit.

Uses spherical trig altitude formula:
  alt = arcsin(sin(lat)*sin(dec) + cos(lat)*cos(dec)*cos(ha))
where ha = hour_angle in radians (0 = meridian).

Public API
----------
HorizonCheckResult(meridian_altitude_deg, n_hours_above, n_hours_below,
                   min_altitude_achieved, max_altitude_achieved, flag)
check_site_horizon(target_dec_deg, site_lat_deg, *, min_altitude_deg,
                   hour_angles) -> HorizonCheckResult
format_horizon_check(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class HorizonCheckResult:
    meridian_altitude_deg: float
    n_hours_above: int
    n_hours_below: int
    min_altitude_achieved: float
    max_altitude_achieved: float
    flag: str  # "OK", "LOW_TARGET", "NEVER_RISES"


def _altitude_deg(lat_rad: float, dec_rad: float, ha_rad: float) -> float:
    """Compute altitude in degrees given latitude, declination, and hour angle."""
    sin_alt = (
        math.sin(lat_rad) * math.sin(dec_rad)
        + math.cos(lat_rad) * math.cos(dec_rad) * math.cos(ha_rad)
    )
    return math.degrees(math.asin(max(-1.0, min(1.0, sin_alt))))


def check_site_horizon(
    target_dec_deg: float,
    site_lat_deg: float,
    *,
    min_altitude_deg: float = 20.0,
    hour_angles: list[float] | None = None,
) -> HorizonCheckResult:
    """Check target visibility against site horizon limit.

    Args:
        target_dec_deg: Target declination in degrees.
        site_lat_deg: Site latitude in degrees.
        min_altitude_deg: Minimum observable altitude in degrees.
        hour_angles: Hour angles (in hours) at which to evaluate. Defaults to
            [-6, -5, ..., 5, 6].

    Returns:
        :class:`HorizonCheckResult`.
    """
    if hour_angles is None:
        hour_angles = list(range(-6, 7))

    lat_rad = math.radians(site_lat_deg)
    dec_rad = math.radians(target_dec_deg)

    meridian_alt = _altitude_deg(lat_rad, dec_rad, 0.0)

    altitudes = [
        _altitude_deg(lat_rad, dec_rad, math.radians(ha * 15.0))
        for ha in hour_angles
    ]

    n_above = sum(1 for a in altitudes if a >= min_altitude_deg)
    n_below = len(altitudes) - n_above

    min_alt = min(altitudes)
    max_alt = max(altitudes)

    if meridian_alt <= 0.0:
        flag = "NEVER_RISES"
    elif max_alt < min_altitude_deg:
        flag = "LOW_TARGET"
    else:
        flag = "OK"

    return HorizonCheckResult(
        meridian_altitude_deg=round(meridian_alt, 2),
        n_hours_above=n_above,
        n_hours_below=n_below,
        min_altitude_achieved=round(min_alt, 2),
        max_altitude_achieved=round(max_alt, 2),
        flag=flag,
    )


def format_horizon_check(result: HorizonCheckResult) -> str:
    """Format horizon check result as Markdown."""
    lines = [
        "## Site Horizon Check",
        "",
        f"- Meridian altitude: {result.meridian_altitude_deg:.1f}°",
        f"- Hours above limit: {result.n_hours_above}",
        f"- Hours below limit: {result.n_hours_below}",
        f"- Min altitude achieved: {result.min_altitude_achieved:.1f}°",
        f"- Max altitude achieved: {result.max_altitude_achieved:.1f}°",
        f"- **Flag: {result.flag}**",
    ]
    return "\n".join(lines)


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dec", type=float, default=30.0, help="Target declination (deg)")
    p.add_argument("--lat", type=float, default=35.0, help="Site latitude (deg)")
    p.add_argument("--min-alt", type=float, default=20.0)
    args = p.parse_args(argv)
    r = check_site_horizon(args.dec, args.lat, min_altitude_deg=args.min_alt)
    print(format_horizon_check(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
