"""Select the best ground-based observatory site for a predicted transit.

Given a list of observer sites and a predicted transit mid-time, determines
which sites can usefully cover the transit window based on airmass and Moon
separation constraints.  Distinct from ``obs_request_formatter`` (formats a
single request) and ``follow_up_scheduler`` (schedules events over time).

Public API
----------
ObservatorySlot(site_name, lat_deg, lon_deg, utc_offset_hours,
                airmass_at_transit, moon_sep_deg, usable, reason)
CoordinationResult(tic_id, transit_mid_bjd, n_sites, usable_slots,
                   best_site, flag)
coordinate_observations(tic_id, transit_mid_bjd, sites, *,
                        duration_hours, max_airmass,
                        min_moon_sep_deg) -> CoordinationResult
format_coordination_result(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ObservatorySlot:
    site_name: str
    lat_deg: float
    lon_deg: float
    utc_offset_hours: float
    airmass_at_transit: float | None  # sec(z) approximation
    moon_sep_deg: float | None
    usable: bool
    reason: str  # why usable or not


@dataclass(frozen=True)
class CoordinationResult:
    tic_id: int | str | None
    transit_mid_bjd: float
    n_sites: int
    usable_slots: tuple[ObservatorySlot, ...]
    best_site: str | None
    flag: str  # "OK" | "NO_USABLE_SITES" | "INVALID"


_BJD_TO_JD = 0.0  # BJD ≈ JD for this approximation (TDB offset < 8 min)
_J2000_JD = 2_451_545.0


def _lst_from_jd(jd: float, lon_deg: float) -> float:
    """Approximate Local Sidereal Time (degrees) from JD and longitude."""
    d = jd - _J2000_JD
    gmst_deg = (280.46061837 + 360.98564736629 * d) % 360.0
    return (gmst_deg + lon_deg) % 360.0


def _hour_angle_deg(lst_deg: float, ra_deg: float) -> float:
    return (lst_deg - ra_deg) % 360.0


def _altitude_deg(lat_deg: float, dec_deg: float, ha_deg: float) -> float:
    lat = math.radians(lat_deg)
    dec = math.radians(dec_deg)
    ha = math.radians(ha_deg)
    sin_alt = (math.sin(lat) * math.sin(dec) +
               math.cos(lat) * math.cos(dec) * math.cos(ha))
    return math.degrees(math.asin(max(-1.0, min(1.0, sin_alt))))


def _airmass(alt_deg: float) -> float | None:
    if alt_deg <= 0:
        return None
    z = math.radians(90.0 - alt_deg)
    return round(1.0 / math.cos(z), 3)


def _evaluate_site(
    site: dict,
    transit_mid_bjd: float,
    target_ra_deg: float,
    target_dec_deg: float,
    max_airmass: float,
    min_moon_sep_deg: float,
) -> ObservatorySlot:
    lat = site.get("lat_deg", 0.0)
    lon = site.get("lon_deg", 0.0)
    utc_off = site.get("utc_offset_hours", 0.0)
    name = site.get("name", "unknown")

    jd = transit_mid_bjd + _BJD_TO_JD
    lst = _lst_from_jd(jd, lon)
    ha = _hour_angle_deg(lst, target_ra_deg)
    # Map HA to [-180, 180]
    if ha > 180:
        ha -= 360.0

    alt = _altitude_deg(lat, target_dec_deg, ha)
    am = _airmass(alt)

    # Simplified Moon position (J2000 epoch, mean longitude)
    d = jd - _J2000_JD
    moon_lon_deg = (218.316 + 13.176396 * d) % 360.0
    moon_lat_deg = 0.0  # simplified
    # Angular separation target–moon (rough)
    delta_lon = math.radians(target_ra_deg - moon_lon_deg)
    moon_sep = math.degrees(math.acos(max(-1.0, min(1.0,
        math.sin(math.radians(target_dec_deg)) * math.sin(math.radians(moon_lat_deg)) +
        math.cos(math.radians(target_dec_deg)) * math.cos(math.radians(moon_lat_deg)) *
        math.cos(delta_lon)
    ))))

    # Determine usability
    reasons: list[str] = []
    if am is None:
        reasons.append("target below horizon")
    elif am > max_airmass:
        reasons.append(f"airmass {am:.2f} > max {max_airmass}")
    if moon_sep < min_moon_sep_deg:
        reasons.append(f"moon sep {moon_sep:.1f}° < min {min_moon_sep_deg}°")

    usable = not reasons
    reason = "; ".join(reasons) if reasons else "OK"

    return ObservatorySlot(
        site_name=name,
        lat_deg=lat,
        lon_deg=lon,
        utc_offset_hours=utc_off,
        airmass_at_transit=am,
        moon_sep_deg=round(moon_sep, 2),
        usable=usable,
        reason=reason,
    )


def coordinate_observations(
    tic_id: int | str | None,
    transit_mid_bjd: float,
    sites: list[dict],
    *,
    duration_hours: float = 2.0,
    max_airmass: float = 2.5,
    min_moon_sep_deg: float = 30.0,
    target_ra_deg: float = 0.0,
    target_dec_deg: float = 0.0,
) -> CoordinationResult:
    """Evaluate all sites for a predicted transit observation.

    Each site dict must contain at minimum ``name``, ``lat_deg``, ``lon_deg``.
    Optional: ``utc_offset_hours``.

    Args:
        tic_id: TIC identifier (for labelling).
        transit_mid_bjd: Predicted transit mid-time (BJD).
        sites: List of site dicts.
        duration_hours: Transit duration (hours; used for context only).
        max_airmass: Reject sites where airmass exceeds this value.
        min_moon_sep_deg: Reject sites where Moon separation is below this.
        target_ra_deg: Target RA (degrees, J2000).
        target_dec_deg: Target Dec (degrees, J2000).

    Returns:
        :class:`CoordinationResult`.
    """
    if not isinstance(sites, list):
        return CoordinationResult(tic_id, transit_mid_bjd, 0, (), None, "INVALID")

    slots = [
        _evaluate_site(s, transit_mid_bjd, target_ra_deg, target_dec_deg,
                       max_airmass, min_moon_sep_deg)
        for s in sites
    ]

    usable = tuple(s for s in slots if s.usable)

    # Best site = lowest airmass among usable
    best_site: str | None = None
    if usable:
        best = min(usable, key=lambda s: s.airmass_at_transit or 999)
        best_site = best.site_name

    flag = "OK" if usable else ("NO_USABLE_SITES" if slots else "INVALID")

    return CoordinationResult(
        tic_id=tic_id,
        transit_mid_bjd=transit_mid_bjd,
        n_sites=len(slots),
        usable_slots=usable,
        best_site=best_site,
        flag=flag,
    )


def format_coordination_result(result: CoordinationResult) -> str:
    """Format coordination result as Markdown."""
    lines = [
        f"## Multi-Observatory Coordinator — TIC {result.tic_id}",
        "",
        f"- Transit mid-time: {result.transit_mid_bjd} BJD",
        f"- Sites evaluated: {result.n_sites}",
        f"- **Usable sites: {len(result.usable_slots)}**",
        f"- **Best site: {result.best_site or '—'}**",
        f"- **Flag: {result.flag}**",
    ]
    if result.usable_slots:
        lines += ["", "| Site | Airmass | Moon sep (°) |",
                  "|---|---|---|"]
        for s in result.usable_slots:
            am = f"{s.airmass_at_transit:.2f}" if s.airmass_at_transit else "—"
            ms = f"{s.moon_sep_deg:.1f}" if s.moon_sep_deg is not None else "—"
            lines.append(f"| {s.site_name} | {am} | {ms} |")
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(
        prog="multi_observatory_coordinator",
        description="Select best observatory site for a predicted transit.",
    )
    parser.add_argument("transit_bjd", type=float)
    parser.add_argument("--sites", type=str, default=None, help="JSON array of site dicts")
    args = parser.parse_args(argv)

    sites = json.loads(args.sites) if args.sites else []
    result = coordinate_observations(None, args.transit_bjd, sites)
    print(format_coordination_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
