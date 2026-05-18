"""Compute airmass for a target at a given time and observatory location.

Uses the plane-parallel approximation: X = 1 / sin(altitude).
LST is derived from BJD via the GMST formula (no astropy required).

Public API
----------
AirmassResult(ra_deg, dec_deg, obs_lat_deg, obs_lon_deg, bjd_mid,
              hour_angle_deg, altitude_deg, airmass, is_observable, flag)
compute_airmass(ra_deg, dec_deg, obs_lat_deg, obs_lon_deg, bjd_mid, *,
                horizon_limit_deg) -> AirmassResult
compute_airmass_curve(ra_deg, dec_deg, obs_lat_deg, obs_lon_deg,
                      bjd_start, bjd_end, *, n_steps,
                      horizon_limit_deg) -> list[AirmassResult]
format_airmass_result(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass

_J2000_JD = 2451545.0  # JD of J2000.0
_SIDEREAL_DAY = 0.9972696  # sidereal day in solar days


def _bjd_to_gmst_deg(bjd: float) -> float:
    """Approximate GMST in degrees from BJD (accurate to ~0.1°)."""
    d = bjd - _J2000_JD
    gmst_deg = (280.46061837 + 360.98564736629 * d) % 360.0
    return gmst_deg


def _compute_altitude(ra_deg: float, dec_deg: float,
                      lat_deg: float, lst_deg: float) -> tuple[float, float]:
    """Return (hour_angle_deg, altitude_deg)."""
    ha_deg = (lst_deg - ra_deg) % 360.0
    if ha_deg > 180:
        ha_deg -= 360.0

    ha_rad = math.radians(ha_deg)
    dec_rad = math.radians(dec_deg)
    lat_rad = math.radians(lat_deg)

    sin_alt = (math.sin(dec_rad) * math.sin(lat_rad)
               + math.cos(dec_rad) * math.cos(lat_rad) * math.cos(ha_rad))
    sin_alt = max(-1.0, min(1.0, sin_alt))
    alt_deg = math.degrees(math.asin(sin_alt))
    return ha_deg, alt_deg


@dataclass(frozen=True)
class AirmassResult:
    ra_deg: float
    dec_deg: float
    obs_lat_deg: float
    obs_lon_deg: float
    bjd_mid: float
    hour_angle_deg: float
    altitude_deg: float
    airmass: float
    is_observable: bool
    flag: str  # "OK" | "BELOW_HORIZON" | "INVALID"


def compute_airmass(
    ra_deg: float,
    dec_deg: float,
    obs_lat_deg: float,
    obs_lon_deg: float,
    bjd_mid: float,
    *,
    horizon_limit_deg: float = 20.0,
) -> AirmassResult:
    """Compute airmass at a given BJD.

    Args:
        ra_deg: Target right ascension in degrees [0, 360).
        dec_deg: Target declination in degrees [-90, 90].
        obs_lat_deg: Observer latitude in degrees.
        obs_lon_deg: Observer longitude in degrees (East positive).
        bjd_mid: Barycentric Julian Date of the observation.
        horizon_limit_deg: Minimum observable altitude in degrees.

    Returns:
        :class:`AirmassResult`.
    """
    if not (-90 <= obs_lat_deg <= 90) or not (-90 <= dec_deg <= 90):
        return AirmassResult(
            ra_deg, dec_deg, obs_lat_deg, obs_lon_deg, bjd_mid,
            0.0, 0.0, 10.0, False, "INVALID",
        )

    gmst_deg = _bjd_to_gmst_deg(bjd_mid)
    lst_deg = (gmst_deg + obs_lon_deg) % 360.0
    ha_deg, alt_deg = _compute_altitude(ra_deg, dec_deg, obs_lat_deg, lst_deg)

    if alt_deg <= 0:
        return AirmassResult(
            ra_deg, dec_deg, obs_lat_deg, obs_lon_deg, bjd_mid,
            ha_deg, alt_deg, 10.0, False, "BELOW_HORIZON",
        )

    airmass = 1.0 / math.sin(math.radians(alt_deg))
    airmass = min(airmass, 10.0)
    is_observable = alt_deg >= horizon_limit_deg

    flag = "OK" if is_observable else "BELOW_HORIZON"
    return AirmassResult(
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        obs_lat_deg=obs_lat_deg,
        obs_lon_deg=obs_lon_deg,
        bjd_mid=bjd_mid,
        hour_angle_deg=round(ha_deg, 4),
        altitude_deg=round(alt_deg, 4),
        airmass=round(airmass, 4),
        is_observable=is_observable,
        flag=flag,
    )


def compute_airmass_curve(
    ra_deg: float,
    dec_deg: float,
    obs_lat_deg: float,
    obs_lon_deg: float,
    bjd_start: float,
    bjd_end: float,
    *,
    n_steps: int = 48,
    horizon_limit_deg: float = 20.0,
) -> list[AirmassResult]:
    """Compute airmass time-series over a BJD range.

    Returns a list of :class:`AirmassResult` with exactly ``n_steps`` entries.
    """
    if n_steps <= 0 or bjd_end <= bjd_start:
        return []
    step = (bjd_end - bjd_start) / n_steps
    return [
        compute_airmass(
            ra_deg, dec_deg, obs_lat_deg, obs_lon_deg,
            bjd_start + i * step, horizon_limit_deg=horizon_limit_deg,
        )
        for i in range(n_steps)
    ]


def format_airmass_result(result: AirmassResult) -> str:
    """Format airmass result as Markdown."""
    lines = [
        "## Airmass",
        "",
        f"- RA: {result.ra_deg:.4f}°",
        f"- Dec: {result.dec_deg:.4f}°",
        f"- Obs lat: {result.obs_lat_deg:.4f}°",
        f"- Obs lon: {result.obs_lon_deg:.4f}°",
        f"- BJD: {result.bjd_mid:.4f}",
    ]
    if result.flag == "INVALID":
        lines.append("- **Flag: INVALID**")
    else:
        lines += [
            f"- Hour angle: {result.hour_angle_deg:.2f}°",
            f"- Altitude: {result.altitude_deg:.2f}°",
            f"- Airmass: {result.airmass:.3f}",
            f"- Observable: {'Yes' if result.is_observable else 'No'}",
            f"- **Flag: {result.flag}**",
        ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="airmass_calculator",
        description="Compute airmass for a target at a given time.",
    )
    parser.add_argument("ra_deg", type=float)
    parser.add_argument("dec_deg", type=float)
    parser.add_argument("obs_lat_deg", type=float)
    parser.add_argument("obs_lon_deg", type=float)
    parser.add_argument("bjd_mid", type=float)
    parser.add_argument("--horizon-limit-deg", type=float, default=20.0)
    args = parser.parse_args(argv)

    result = compute_airmass(
        args.ra_deg, args.dec_deg, args.obs_lat_deg, args.obs_lon_deg, args.bjd_mid,
        horizon_limit_deg=args.horizon_limit_deg,
    )
    print(format_airmass_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
