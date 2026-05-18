"""Derive planet radius from transit depth and stellar radius.

Propagates both measurement uncertainties to final Rp/Re and Rp/Rjup values
and classifies the companion by size.

Public API
----------
PlanetRadiusResult(depth_ppm, stellar_radius_rsun, planet_radius_rearth,
                   planet_radius_rjup, planet_radius_rsun, depth_err_ppm,
                   stellar_radius_err_rsun, planet_radius_err_rearth,
                   radius_ratio, classification)
estimate_planet_radius(depth_ppm, stellar_radius_rsun, *, depth_err_ppm,
                       stellar_radius_err_rsun) -> PlanetRadiusResult
format_planet_radius_result(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass

_RSUN_TO_REARTH = 109.076
_RSUN_TO_RJUP = 9.731


@dataclass(frozen=True)
class PlanetRadiusResult:
    depth_ppm: float
    stellar_radius_rsun: float
    planet_radius_rearth: float
    planet_radius_rjup: float
    planet_radius_rsun: float
    depth_err_ppm: float | None
    stellar_radius_err_rsun: float | None
    planet_radius_err_rearth: float | None
    radius_ratio: float                 # Rp / R_star = sqrt(depth/1e6)
    classification: str                 # sub-Earth / super-Earth / Neptune / Jupiter / stellar


def _classify(rearth: float) -> str:
    if rearth < 1.25:
        return "sub-Earth"
    if rearth < 2.0:
        return "super-Earth"
    if rearth < 6.0:
        return "Neptune"
    if rearth < 15.0:
        return "Jupiter"
    return "stellar"


def estimate_planet_radius(
    depth_ppm: float,
    stellar_radius_rsun: float,
    *,
    depth_err_ppm: float | None = None,
    stellar_radius_err_rsun: float | None = None,
) -> PlanetRadiusResult:
    """Derive companion radius from transit depth and stellar radius.

    Args:
        depth_ppm: Transit depth in parts per million.
        stellar_radius_rsun: Host-star radius in solar radii.
        depth_err_ppm: 1-sigma depth uncertainty (ppm).
        stellar_radius_err_rsun: 1-sigma stellar radius uncertainty (R_sun).

    Returns:
        :class:`PlanetRadiusResult`.
    """
    depth = max(float(depth_ppm), 0.0)
    r_star = max(float(stellar_radius_rsun), 1e-9)

    radius_ratio = math.sqrt(depth / 1e6)
    rp_rsun = radius_ratio * r_star
    rp_rearth = rp_rsun * _RSUN_TO_REARTH
    rp_rjup = rp_rsun * _RSUN_TO_RJUP

    # Error propagation: (δRp/Rp)² = (0.5 * δdepth/depth)² + (δR*/R*)²
    rp_err: float | None = None
    if depth > 0:
        terms: list[float] = []
        if depth_err_ppm is not None:
            terms.append((0.5 * float(depth_err_ppm) / depth) ** 2)
        if stellar_radius_err_rsun is not None:
            terms.append((float(stellar_radius_err_rsun) / r_star) ** 2)
        if terms:
            rp_err = rp_rearth * math.sqrt(sum(terms))

    return PlanetRadiusResult(
        depth_ppm=depth_ppm,
        stellar_radius_rsun=stellar_radius_rsun,
        planet_radius_rearth=round(rp_rearth, 4),
        planet_radius_rjup=round(rp_rjup, 5),
        planet_radius_rsun=round(rp_rsun, 6),
        depth_err_ppm=depth_err_ppm,
        stellar_radius_err_rsun=stellar_radius_err_rsun,
        planet_radius_err_rearth=round(rp_err, 4) if rp_err is not None else None,
        radius_ratio=round(radius_ratio, 6),
        classification=_classify(rp_rearth),
    )


def format_planet_radius_result(result: PlanetRadiusResult) -> str:
    """Format planet radius result as Markdown."""
    err_str = (
        f" ± {result.planet_radius_err_rearth:.3f}"
        if result.planet_radius_err_rearth is not None else ""
    )
    lines = [
        "## Planet Radius Estimate",
        "",
        f"- Depth: {result.depth_ppm:.1f} ppm",
        f"- Stellar radius: {result.stellar_radius_rsun:.3f} R☉",
        f"- Radius ratio (Rp/R★): {result.radius_ratio:.5f}",
        f"- Planet radius: **{result.planet_radius_rearth:.3f}{err_str} R⊕**",
        f"  ({result.planet_radius_rjup:.4f} R♃ / {result.planet_radius_rsun:.5f} R☉)",
        f"- Classification: **{result.classification}**",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="planet_radius_estimator",
        description="Estimate planet radius from transit depth and stellar radius.",
    )
    parser.add_argument("depth_ppm", type=float)
    parser.add_argument("stellar_radius_rsun", type=float)
    parser.add_argument("--depth-err", type=float, default=None)
    parser.add_argument("--radius-err", type=float, default=None)
    args = parser.parse_args(argv)

    result = estimate_planet_radius(
        args.depth_ppm, args.stellar_radius_rsun,
        depth_err_ppm=args.depth_err,
        stellar_radius_err_rsun=args.radius_err,
    )
    print(format_planet_radius_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
