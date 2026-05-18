"""Check angular separation between a target and the Moon.

Uses a low-precision lunar position model (Meeus Ch. 47 simplified,
accurate to ~1°) to estimate the Moon's RA/Dec at a given BJD.

Public API
----------
MoonSeparationResult(ra_deg, dec_deg, bjd, moon_ra_deg, moon_dec_deg,
                     moon_separation_deg, moon_illumination_fraction,
                     moon_phase_name, is_problematic, flag)
check_moon_separation(ra_deg, dec_deg, bjd, *, min_separation_deg,
                      illumination_threshold) -> MoonSeparationResult
format_moon_separation_result(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass

_J2000_JD = 2451545.0
_DEG = math.pi / 180.0


def _moon_position(bjd: float) -> tuple[float, float, float]:
    """Approximate Moon RA (deg), Dec (deg), and phase angle (deg).

    Based on Meeus 'Astronomical Algorithms' Ch. 47 low-precision formulae.
    Accurate to ~1° in ecliptic longitude.
    """
    d = bjd - _J2000_JD

    # Mean longitude and anomaly
    L0 = (218.3164477 + 13.17639650 * d) % 360.0
    M0 = (134.9633964 + 13.06499295 * d) % 360.0  # Moon mean anomaly
    Ms = (357.5291092 + 0.98560028 * d) % 360.0   # Sun mean anomaly
    D = (297.8501921 + 12.19074912 * d) % 360.0   # Moon elongation
    F = (93.2720950 + 13.22935024 * d) % 360.0    # Moon argument of latitude

    M0r = M0 * _DEG
    Msr = Ms * _DEG
    Dr = D * _DEG
    Fr = F * _DEG

    # Ecliptic longitude correction
    dlon = (6.289 * math.sin(M0r)
            - 1.274 * math.sin(2 * Dr - M0r)
            + 0.658 * math.sin(2 * Dr)
            - 0.214 * math.sin(2 * M0r)
            - 0.186 * math.sin(Msr))
    lam = (L0 + dlon) % 360.0

    # Ecliptic latitude
    beta = 5.128 * math.sin(Fr)

    # Phase angle (elongation-based)
    phase_angle = (180.0 - D % 360.0) % 360.0
    if phase_angle > 180:
        phase_angle = 360 - phase_angle

    # Convert ecliptic to equatorial (obliquity 23.4393°)
    eps = 23.4393 * _DEG
    lam_r = lam * _DEG
    beta_r = beta * _DEG

    sin_dec = (math.sin(beta_r) * math.cos(eps)
               + math.cos(beta_r) * math.sin(eps) * math.sin(lam_r))
    sin_dec = max(-1.0, min(1.0, sin_dec))
    dec = math.degrees(math.asin(sin_dec))

    cos_ra_cos_dec = math.cos(lam_r) * math.cos(beta_r)
    sin_ra_cos_dec = (math.sin(lam_r) * math.cos(beta_r) * math.cos(eps)
                      - math.sin(beta_r) * math.sin(eps))
    ra = math.degrees(math.atan2(sin_ra_cos_dec, cos_ra_cos_dec)) % 360.0

    return ra, dec, phase_angle


def _angular_separation(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    """Great-circle separation in degrees."""
    ra1r, dec1r, ra2r, dec2r = (x * _DEG for x in (ra1, dec1, ra2, dec2))
    cos_sep = (math.sin(dec1r) * math.sin(dec2r)
               + math.cos(dec1r) * math.cos(dec2r) * math.cos(ra1r - ra2r))
    cos_sep = max(-1.0, min(1.0, cos_sep))
    return math.degrees(math.acos(cos_sep))


def _phase_name(illum: float) -> str:
    if illum < 0.03:
        return "new"
    if illum < 0.40:
        return "crescent"
    if illum < 0.60:
        return "quarter"
    if illum < 0.97:
        return "gibbous"
    return "full"


@dataclass(frozen=True)
class MoonSeparationResult:
    ra_deg: float
    dec_deg: float
    bjd: float
    moon_ra_deg: float
    moon_dec_deg: float
    moon_separation_deg: float
    moon_illumination_fraction: float
    moon_phase_name: str
    is_problematic: bool
    flag: str  # "OK" | "MOON_WARN" | "MOON_SEVERE" | "INVALID"


def check_moon_separation(
    ra_deg: float,
    dec_deg: float,
    bjd: float,
    *,
    min_separation_deg: float = 30.0,
    illumination_threshold: float = 0.50,
) -> MoonSeparationResult:
    """Check angular separation between target and Moon.

    Args:
        ra_deg: Target RA in degrees.
        dec_deg: Target Dec in degrees.
        bjd: Barycentric Julian Date.
        min_separation_deg: Minimum acceptable separation in degrees.
        illumination_threshold: Illumination fraction above which bright moon matters.

    Returns:
        :class:`MoonSeparationResult`.
    """
    if not (-90 <= dec_deg <= 90):
        return MoonSeparationResult(
            ra_deg, dec_deg, bjd, 0.0, 0.0, 0.0, 0.0, "new", False, "INVALID",
        )

    moon_ra, moon_dec, phase_angle = _moon_position(bjd)
    separation = _angular_separation(ra_deg, dec_deg, moon_ra, moon_dec)
    illum = (1.0 + math.cos(math.radians(phase_angle))) / 2.0

    is_problematic = (separation < min_separation_deg
                      and illum > illumination_threshold)

    if illum > illumination_threshold and separation < min_separation_deg / 2:
        flag = "MOON_SEVERE"
    elif is_problematic:
        flag = "MOON_WARN"
    else:
        flag = "OK"

    return MoonSeparationResult(
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        bjd=bjd,
        moon_ra_deg=round(moon_ra, 4),
        moon_dec_deg=round(moon_dec, 4),
        moon_separation_deg=round(separation, 4),
        moon_illumination_fraction=round(illum, 4),
        moon_phase_name=_phase_name(illum),
        is_problematic=is_problematic,
        flag=flag,
    )


def format_moon_separation_result(result: MoonSeparationResult) -> str:
    """Format moon separation result as Markdown."""
    lines = [
        "## Moon Separation",
        "",
        f"- Target RA: {result.ra_deg:.4f}°",
        f"- Target Dec: {result.dec_deg:.4f}°",
        f"- BJD: {result.bjd:.4f}",
    ]
    if result.flag == "INVALID":
        lines.append("- **Flag: INVALID**")
    else:
        lines += [
            f"- Moon RA: {result.moon_ra_deg:.2f}°",
            f"- Moon Dec: {result.moon_dec_deg:.2f}°",
            f"- Separation: {result.moon_separation_deg:.2f}°",
            f"- Illumination: {result.moon_illumination_fraction:.2%} ({result.moon_phase_name})",
            f"- Problematic: {'Yes' if result.is_problematic else 'No'}",
            f"- **Flag: {result.flag}**",
        ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="moon_separation_checker",
        description="Check Moon angular separation for a target.",
    )
    parser.add_argument("ra_deg", type=float)
    parser.add_argument("dec_deg", type=float)
    parser.add_argument("bjd", type=float)
    parser.add_argument("--min-separation-deg", type=float, default=30.0)
    parser.add_argument("--illumination-threshold", type=float, default=0.50)
    args = parser.parse_args(argv)

    result = check_moon_separation(
        args.ra_deg, args.dec_deg, args.bjd,
        min_separation_deg=args.min_separation_deg,
        illumination_threshold=args.illumination_threshold,
    )
    print(format_moon_separation_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
