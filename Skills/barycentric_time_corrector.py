"""Compute and apply barycentric time corrections (BJD − JD).

Converts between Julian Date systems using a simplified barycentric correction
that accounts for Earth's orbital position relative to the Solar System
barycentre.  Accuracy is ~10 s, sufficient for transit-timing purposes when
no external ephemeris service is available.

Public API
----------
BarycentricResult(utc_jd, bjd_tdb, correction_seconds, flag)
compute_barycentric_correction(utc_jd, ra_deg, dec_deg) -> float
apply_barycentric_correction(times, ra_deg, dec_deg, *,
                             from_system) -> BarycentricResult
format_barycentric_result(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BarycentricResult:
    utc_jd: float | None            # input time (JD UTC)
    bjd_tdb: float | None           # output BJD(TDB)
    correction_seconds: float | None  # BJD - JD in seconds
    flag: str  # "OK" | "INVALID"


_DEG2RAD = math.pi / 180.0
_J2000 = 2451545.0   # JD of J2000.0
_AU_LIGHT_SECONDS = 499.00478384  # light travel time per AU in seconds


def _earth_position_au(jd: float) -> tuple[float, float, float]:
    """Approximate Earth barycentre position in AU (VSOP87 low-order terms)."""
    T = (jd - _J2000) / 36525.0  # Julian centuries from J2000

    # Mean longitude and anomaly
    L = math.radians(280.46646 + 36000.76983 * T) % (2 * math.pi)
    M = math.radians(357.52911 + 35999.05029 * T - 0.0001537 * T * T)

    # Equation of centre
    C = (math.radians(1.914602 - 0.004817 * T - 0.000014 * T * T) * math.sin(M)
         + math.radians(0.019993 - 0.000101 * T) * math.sin(2 * M)
         + math.radians(0.000289) * math.sin(3 * M))

    # Sun's true longitude and radius vector
    sun_lon = L + C
    r = 1.000001018 * (1 - 0.016708634 ** 2) / (1 + 0.016708634 * math.cos(M + C))

    # Convert to ecliptic rectangular coordinates
    eps = math.radians(23.439291 - 0.013004 * T)  # obliquity
    # Earth is opposite the Sun
    x = -r * math.cos(sun_lon)
    y = -r * math.sin(sun_lon) * math.cos(eps)
    z = -r * math.sin(sun_lon) * math.sin(eps)
    return x, y, z


def compute_barycentric_correction(
    utc_jd: float,
    ra_deg: float,
    dec_deg: float,
) -> float:
    """Compute BJD(TDB) - JD(UTC) in seconds.

    Args:
        utc_jd: Julian Date (UTC).
        ra_deg: Target right ascension (degrees, J2000).
        dec_deg: Target declination (degrees, J2000).

    Returns:
        Correction in seconds (add to JD UTC to get BJD TDB).
    """
    ra_r = ra_deg * _DEG2RAD
    dec_r = dec_deg * _DEG2RAD

    # Unit vector from barycentre toward target
    sx = math.cos(dec_r) * math.cos(ra_r)
    sy = math.cos(dec_r) * math.sin(ra_r)
    sz = math.sin(dec_r)

    # Earth barycentre position
    ex, ey, ez = _earth_position_au(utc_jd)

    # Roemer delay: dot product of earth position with target direction
    roemer_au = ex * sx + ey * sy + ez * sz
    roemer_seconds = roemer_au * _AU_LIGHT_SECONDS

    # TDB - TT correction (~1.7 ms, negligible for transit work)
    # Approximate TT - UTC = 69.184 s (as of 2026, 37 leap seconds + 32.184)
    tt_utc = 69.184

    return roemer_seconds + tt_utc


def apply_barycentric_correction(
    times: list[float],
    ra_deg: float,
    dec_deg: float,
    *,
    from_system: str = "JD_UTC",
) -> BarycentricResult:
    """Apply barycentric correction to a list of times.

    Args:
        times: Input time array.
        ra_deg: Target RA (degrees, J2000).
        dec_deg: Target Dec (degrees, J2000).
        from_system: Input time system — ``"JD_UTC"`` or ``"BJD_TDB"`` (identity).

    Returns:
        :class:`BarycentricResult` with the first input time and its BJD.
    """
    if from_system not in ("JD_UTC", "BJD_TDB"):
        return BarycentricResult(None, None, None, "INVALID")
    if not (-90 <= dec_deg <= 90) or not (0 <= ra_deg <= 360):
        return BarycentricResult(None, None, None, "INVALID")
    if not times:
        return BarycentricResult(None, None, None, "INVALID")

    t0 = times[0]
    if from_system == "BJD_TDB":
        return BarycentricResult(t0, t0, 0.0, "OK")

    corr = compute_barycentric_correction(t0, ra_deg, dec_deg)
    bjd = t0 + corr / 86400.0  # convert seconds → days

    return BarycentricResult(
        utc_jd=t0,
        bjd_tdb=round(bjd, 8),
        correction_seconds=round(corr, 3),
        flag="OK",
    )


def format_barycentric_result(result: BarycentricResult) -> str:
    """Format barycentric correction result as Markdown."""
    lines = [
        "## Barycentric Time Correction",
        "",
        f"- Input JD (UTC): {result.utc_jd}",
        f"- BJD (TDB): {result.bjd_tdb}",
        f"- Correction: {result.correction_seconds} seconds",
        f"- **Flag: {result.flag}**",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="barycentric_time_corrector",
        description="Compute BJD(TDB) - JD(UTC) barycentric correction.",
    )
    parser.add_argument("jd", type=float)
    parser.add_argument("ra_deg", type=float)
    parser.add_argument("dec_deg", type=float)
    args = parser.parse_args(argv)

    result = apply_barycentric_correction([args.jd], args.ra_deg, args.dec_deg)
    print(format_barycentric_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
