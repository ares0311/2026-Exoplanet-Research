"""Derive stellar radius from transit depth and known planet radius."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class StellarRadiusResult:
    depth_ppm: float
    planet_radius_rearth: float
    stellar_radius_rsun: float
    stellar_radius_rsun_err: float
    radius_ratio: float
    flag: str


def derive_stellar_radius(
    depth_ppm: float,
    planet_radius_rearth: float,
    depth_err_ppm: float = 0.0,
) -> StellarRadiusResult:
    """
    Derive stellar radius from transit depth and known planet radius.

    depth = (Rp/R*)^2  =>  R* = Rp / sqrt(depth)

    Parameters
    ----------
    depth_ppm:            Transit depth in parts-per-million.
    planet_radius_rearth: Planet radius in Earth radii.
    depth_err_ppm:        1-sigma uncertainty on depth (ppm); used for error propagation.
    """
    _RSUN_REARTH = 109.076  # 1 R_sun in Earth radii

    if not math.isfinite(depth_ppm) or depth_ppm <= 0.0:
        return StellarRadiusResult(
            depth_ppm=depth_ppm,
            planet_radius_rearth=planet_radius_rearth,
            stellar_radius_rsun=float("nan"),
            stellar_radius_rsun_err=float("nan"),
            radius_ratio=float("nan"),
            flag="INVALID_DEPTH",
        )
    if not math.isfinite(planet_radius_rearth) or planet_radius_rearth <= 0.0:
        return StellarRadiusResult(
            depth_ppm=depth_ppm,
            planet_radius_rearth=planet_radius_rearth,
            stellar_radius_rsun=float("nan"),
            stellar_radius_rsun_err=float("nan"),
            radius_ratio=float("nan"),
            flag="INVALID_PLANET_RADIUS",
        )

    depth_frac = depth_ppm * 1e-6
    rp_r = math.sqrt(depth_frac)  # Rp/R*
    rstar_rearth = planet_radius_rearth / rp_r
    rstar_rsun = rstar_rearth / _RSUN_REARTH

    # Error propagation: sigma_R* = R* * sigma_depth / (2 * depth)
    if depth_err_ppm > 0.0 and math.isfinite(depth_err_ppm):
        rstar_err = rstar_rsun * (depth_err_ppm / (2.0 * depth_ppm))
    else:
        rstar_err = rstar_rsun * 0.05  # default 5% systematic

    flag = "OK"
    if rstar_rsun < 0.08 or rstar_rsun > 1000.0:
        flag = "IMPLAUSIBLE_STELLAR_RADIUS"

    return StellarRadiusResult(
        depth_ppm=depth_ppm,
        planet_radius_rearth=planet_radius_rearth,
        stellar_radius_rsun=round(rstar_rsun, 4),
        stellar_radius_rsun_err=round(rstar_err, 4),
        radius_ratio=round(rp_r, 5),
        flag=flag,
    )


def format_radius_result(r: StellarRadiusResult) -> str:
    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| Depth (ppm) | {r.depth_ppm:.1f} |\n"
        f"| Planet radius (R⊕) | {r.planet_radius_rearth:.3f} |\n"
        f"| Rp/R★ | {r.radius_ratio:.5f} |\n"
        f"| Stellar radius (R☉) | {r.stellar_radius_rsun:.4f} ± {r.stellar_radius_rsun_err:.4f} |\n"
        f"| Flag | {r.flag} |\n"
    )


def _cli() -> int:
    p = argparse.ArgumentParser(description="Derive stellar radius from transit depth.")
    p.add_argument("depth_ppm", type=float)
    p.add_argument("planet_radius_rearth", type=float)
    p.add_argument("--depth-err", type=float, default=0.0)
    args = p.parse_args()
    r = derive_stellar_radius(args.depth_ppm, args.planet_radius_rearth, args.depth_err)
    print(format_radius_result(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
