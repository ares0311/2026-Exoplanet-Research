"""Plan monthly ground-based observability for a sky target.

Estimates whether a target is observable each calendar month given a ground
site, minimum altitude constraint, and approximate Sun position.  Uses a
purely analytic (no external library) solar RA approximation and hour-angle
geometry.  Distinct from ``tess_visibility_checker`` (space-based TESS sectors)
and ``multi_observatory_coordinator`` (single-epoch airmass curves).

Public API
----------
MonthlyVisibility(month, month_name, sun_ra_deg, target_ha_at_midnight_deg,
                  max_altitude_deg, is_observable, observable_hours_approx)
SeasonalVisibilityResult(ra_deg, dec_deg, site_lat_deg, min_altitude_deg,
                         year, monthly, n_observable_months, flag)
plan_seasonal_visibility(ra_deg, dec_deg, *, site_lat_deg, min_altitude_deg,
                         year) -> SeasonalVisibilityResult
format_seasonal_visibility(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass

_MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

# Day-of-year of the 15th of each month (non-leap year)
_MID_MONTH_DOY = [15, 46, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349]


def _sun_ra_deg(day_of_year: int) -> float:
    """Approximate solar RA (degrees) from day of year (1=Jan 1)."""
    # Vernal equinox ~March 20 = day 79
    mean_anomaly_deg = 360.0 * (day_of_year - 3) / 365.25
    mean_anomaly_rad = math.radians(mean_anomaly_deg)
    # Equation of centre (first-order correction, ~2°)
    eoc = 1.915 * math.sin(mean_anomaly_rad) + 0.020 * math.sin(2 * mean_anomaly_rad)
    ecliptic_lon = mean_anomaly_deg + eoc + 102.9372  # longitude of perihelion
    ecliptic_lon_rad = math.radians(ecliptic_lon % 360.0)
    eps = math.radians(23.4393)
    ra_rad = math.atan2(math.cos(eps) * math.sin(ecliptic_lon_rad), math.cos(ecliptic_lon_rad))
    return math.degrees(ra_rad) % 360.0


def _altitude(dec_deg: float, ha_deg: float, lat_deg: float) -> float:
    """Altitude (degrees) of object at hour angle *ha_deg* from latitude *lat_deg*."""
    dec = math.radians(dec_deg)
    ha = math.radians(ha_deg)
    lat = math.radians(lat_deg)
    sin_alt = (math.sin(dec) * math.sin(lat)
               + math.cos(dec) * math.cos(lat) * math.cos(ha))
    return math.degrees(math.asin(max(-1.0, min(1.0, sin_alt))))


def _observable_hours(dec_deg: float, lat_deg: float, min_alt_deg: float) -> float:
    """Approximate hours above *min_alt_deg* for a target (ignoring solar constraint)."""
    dec = math.radians(dec_deg)
    lat = math.radians(lat_deg)
    cos_ha = (math.sin(math.radians(min_alt_deg)) - math.sin(dec) * math.sin(lat)) / (
        math.cos(dec) * math.cos(lat)
    )
    if cos_ha >= 1.0:
        return 0.0
    if cos_ha <= -1.0:
        return 24.0
    ha_hours = math.degrees(math.acos(cos_ha)) / 15.0
    return 2.0 * ha_hours


@dataclass(frozen=True)
class MonthlyVisibility:
    month: int           # 1–12
    month_name: str
    sun_ra_deg: float
    target_ha_at_midnight_deg: float  # HA of target at local midnight
    max_altitude_deg: float           # altitude at transit (HA = 0)
    is_observable: bool               # above min_alt for >1 h and not too close to Sun
    observable_hours_approx: float    # hours above min_alt per night


@dataclass(frozen=True)
class SeasonalVisibilityResult:
    ra_deg: float
    dec_deg: float
    site_lat_deg: float
    min_altitude_deg: float
    year: int
    monthly: tuple[MonthlyVisibility, ...]
    n_observable_months: int
    flag: str  # "OK" | "INVALID"


def plan_seasonal_visibility(
    ra_deg: float,
    dec_deg: float,
    *,
    site_lat_deg: float = 28.3,     # La Palma default
    min_altitude_deg: float = 30.0,
    year: int = 2026,
) -> SeasonalVisibilityResult:
    """Estimate monthly ground-based observability across a year.

    Args:
        ra_deg: Target right ascension (degrees).
        dec_deg: Target declination (degrees).
        site_lat_deg: Observer latitude (degrees, + north).
        min_altitude_deg: Minimum altitude for observation (degrees).
        year: Calendar year for day-of-year calculations.

    Returns:
        :class:`SeasonalVisibilityResult`.
    """
    if not all(math.isfinite(v) for v in (ra_deg, dec_deg, site_lat_deg, min_altitude_deg)):
        return SeasonalVisibilityResult(ra_deg, dec_deg, site_lat_deg, min_altitude_deg,
                                        year, (), 0, "INVALID")
    if not (-90.0 <= dec_deg <= 90.0):
        return SeasonalVisibilityResult(ra_deg, dec_deg, site_lat_deg, min_altitude_deg,
                                        year, (), 0, "INVALID")
    if not (-90.0 <= site_lat_deg <= 90.0):
        return SeasonalVisibilityResult(ra_deg, dec_deg, site_lat_deg, min_altitude_deg,
                                        year, (), 0, "INVALID")

    transit_alt = _altitude(dec_deg, 0.0, site_lat_deg)  # max altitude at transit
    obs_hours = _observable_hours(dec_deg, site_lat_deg, min_altitude_deg)

    monthly: list[MonthlyVisibility] = []
    for i, doy in enumerate(_MID_MONTH_DOY):
        sun_ra = _sun_ra_deg(doy)
        # HA of target at local midnight: at midnight, LST ≈ sun_ra + 180°
        # HA = LST - target_RA
        lst_midnight = (sun_ra + 180.0) % 360.0
        ha_deg = (lst_midnight - ra_deg + 360.0) % 360.0
        # Normalise to (-180, 180]
        if ha_deg > 180.0:
            ha_deg -= 360.0

        # Sun distance in RA (rough anti-Sun separation)
        sun_sep = abs((ra_deg - sun_ra + 540.0) % 360.0 - 180.0)
        # Observable if: transit alt > min_alt AND enough dark hours AND not within 30° of Sun
        is_obs = (
            transit_alt >= min_altitude_deg
            and obs_hours >= 1.0
            and sun_sep >= 30.0
        )

        monthly.append(MonthlyVisibility(
            month=i + 1,
            month_name=_MONTH_NAMES[i],
            sun_ra_deg=round(sun_ra, 2),
            target_ha_at_midnight_deg=round(ha_deg, 2),
            max_altitude_deg=round(transit_alt, 2),
            is_observable=is_obs,
            observable_hours_approx=round(obs_hours, 2),
        ))

    n_obs = sum(1 for m in monthly if m.is_observable)
    return SeasonalVisibilityResult(
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        site_lat_deg=site_lat_deg,
        min_altitude_deg=min_altitude_deg,
        year=year,
        monthly=tuple(monthly),
        n_observable_months=n_obs,
        flag="OK",
    )


def format_seasonal_visibility(result: SeasonalVisibilityResult) -> str:
    """Format seasonal visibility result as Markdown."""
    lines = [
        "## Seasonal Visibility Planner",
        "",
        f"- RA: {result.ra_deg}°, Dec: {result.dec_deg}°",
        f"- Site latitude: {result.site_lat_deg}°",
        f"- Min altitude: {result.min_altitude_deg}°",
        f"- **Observable months: {result.n_observable_months} / 12**",
        f"- **Flag: {result.flag}**",
    ]
    if result.monthly:
        lines += [
            "", "| Month | Sun RA° | HA at midnight° | Max Alt° | Observable |",
            "|---|---|---|---|---|",
        ]
        for m in result.monthly:
            obs_str = "**Yes**" if m.is_observable else "No"
            lines.append(
                f"| {m.month_name[:3]} | {m.sun_ra_deg:.1f} | {m.target_ha_at_midnight_deg:.1f}"
                f" | {m.max_altitude_deg:.1f} | {obs_str} |"
            )
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="seasonal_visibility_planner",
        description="Plan monthly ground-based observability for a sky target.",
    )
    parser.add_argument("ra_deg", type=float)
    parser.add_argument("dec_deg", type=float)
    parser.add_argument("--lat", type=float, default=28.3)
    parser.add_argument("--min-alt", type=float, default=30.0)
    parser.add_argument("--year", type=int, default=2026)
    args = parser.parse_args(argv)

    result = plan_seasonal_visibility(
        args.ra_deg, args.dec_deg,
        site_lat_deg=args.lat, min_altitude_deg=args.min_alt, year=args.year,
    )
    print(format_seasonal_visibility(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
