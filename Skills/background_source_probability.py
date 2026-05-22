"""Estimate the background eclipsing binary prior from galactic source density.

Uses a simplified galactic latitude model to estimate the surface density of
background stars brighter than a given magnitude limit, then computes the
probability that any such source falls within the photometric aperture.

Public API
----------
BackgroundSourceResult(galactic_lat_deg, source_density_per_sq_deg,
                        n_expected_in_aperture, bgeb_prior,
                        is_crowded, flag)
estimate_bg_source_prob(ra_deg, dec_deg, *, aperture_arcsec2,
                         mag_limit, bgeb_eclipse_rate) -> BackgroundSourceResult
format_bg_prob_result(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BackgroundSourceResult:
    galactic_lat_deg: float | None
    source_density_per_sq_deg: float | None   # stars/deg² brighter than mag_limit
    n_expected_in_aperture: float | None       # expected N in aperture
    bgeb_prior: float                          # P(bgEB) in [0, 1]
    is_crowded: bool                           # n_expected >= 1
    flag: str  # "OK" | "INVALID"


# Approximate equatorial→galactic conversion (J2000)
_RA_NGP = 192.85948   # deg
_DEC_NGP = 27.12825   # deg
_L_NCP = 122.93192    # deg
_DEG2RAD = math.pi / 180.0


def _equatorial_to_galactic(ra_deg: float, dec_deg: float) -> tuple[float, float]:
    """Approximate equatorial to galactic coordinate conversion."""
    ra_r = ra_deg * _DEG2RAD
    dec_r = dec_deg * _DEG2RAD
    ngp_ra = _RA_NGP * _DEG2RAD
    ngp_dec = _DEC_NGP * _DEG2RAD

    sin_b = (math.sin(dec_r) * math.sin(ngp_dec)
             + math.cos(dec_r) * math.cos(ngp_dec) * math.cos(ra_r - ngp_ra))
    b_rad = math.asin(max(-1.0, min(1.0, sin_b)))

    cos_b = math.cos(b_rad)
    if abs(cos_b) < 1e-9:
        return 0.0, math.degrees(b_rad)

    sin_l_ncp = math.cos(dec_r) * math.sin(ra_r - ngp_ra) / cos_b
    cos_l_ncp = (math.sin(dec_r) - sin_b * math.sin(ngp_dec)) / (cos_b * math.cos(ngp_dec))
    l_rad = math.atan2(sin_l_ncp, cos_l_ncp)
    l_deg = (math.degrees(l_rad) + _L_NCP) % 360.0

    return l_deg, math.degrees(b_rad)


def _source_density(b_deg: float, mag_limit: float) -> float:
    """Approximate stellar surface density (stars/deg²) at galactic latitude b."""
    # Simple model: n(b, m) = n0 * exp(-|b| / H_b) * 10^(0.6*(m - m0))
    # Parameters tuned to rough TRILEGAL/2MASS statistics
    n0 = 3000.0          # stars/deg² at b=0, T<14
    h_b = 15.0           # galactic scale height in degrees
    m0 = 14.0            # reference magnitude
    density = n0 * math.exp(-abs(b_deg) / h_b) * 10 ** (0.6 * (mag_limit - m0))
    return max(0.0, density)


def estimate_bg_source_prob(
    ra_deg: float,
    dec_deg: float,
    *,
    aperture_arcsec2: float = 1385.4,  # TESS 21″/px × 3px radius ≈ π×21²
    mag_limit: float = 19.0,
    bgeb_eclipse_rate: float = 0.005,  # ~0.5% of stars are EBs
) -> BackgroundSourceResult:
    """Estimate background source probability within the photometric aperture.

    Args:
        ra_deg: Right ascension (degrees).
        dec_deg: Declination (degrees).
        aperture_arcsec2: Effective aperture area (arcsec²).
        mag_limit: Limiting magnitude for background source search.
        bgeb_eclipse_rate: Fraction of background stars that are EBs.

    Returns:
        :class:`BackgroundSourceResult`.
    """
    if not (-90 <= dec_deg <= 90) or not (0 <= ra_deg <= 360):
        return BackgroundSourceResult(None, None, None, 0.0, False, "INVALID")

    _, b_deg = _equatorial_to_galactic(ra_deg, dec_deg)
    density = _source_density(b_deg, mag_limit)

    # Convert aperture from arcsec² to deg²
    aperture_deg2 = aperture_arcsec2 / (3600.0 ** 2)
    n_expected = density * aperture_deg2

    # P(bgEB) = 1 - P(no bgEB in aperture)
    # Using Poisson: P(N≥1) = 1 - exp(-lambda) where lambda = n_expected * rate
    lam = n_expected * bgeb_eclipse_rate
    p_bgeb = 1.0 - math.exp(-lam)
    p_bgeb = min(1.0, max(0.0, p_bgeb))

    return BackgroundSourceResult(
        galactic_lat_deg=round(b_deg, 3),
        source_density_per_sq_deg=round(density, 1),
        n_expected_in_aperture=round(n_expected, 4),
        bgeb_prior=round(p_bgeb, 6),
        is_crowded=n_expected >= 1.0,
        flag="OK",
    )


def format_bg_prob_result(result: BackgroundSourceResult) -> str:
    """Format background source probability result as Markdown."""
    lines = [
        "## Background Source Probability",
        "",
        f"- Galactic latitude: {result.galactic_lat_deg}°",
        f"- Source density: {result.source_density_per_sq_deg} stars/deg²",
        f"- Expected sources in aperture: {result.n_expected_in_aperture}",
        f"- **bgEB prior: {result.bgeb_prior:.5f}**",
        f"- Crowded field: {'Yes' if result.is_crowded else 'No'}",
        f"- **Flag: {result.flag}**",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="background_source_probability",
        description="Estimate background EB probability from galactic source density.",
    )
    parser.add_argument("ra_deg", type=float)
    parser.add_argument("dec_deg", type=float)
    parser.add_argument("--mag-limit", type=float, default=19.0)
    args = parser.parse_args(argv)

    result = estimate_bg_source_prob(args.ra_deg, args.dec_deg, mag_limit=args.mag_limit)
    print(format_bg_prob_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
