"""Estimate stellar surface density from galactic latitude (stars per sq-arcmin)."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

# Scale height model: rho ~ rho0 * exp(-|b| / b_scale)
# Calibrated to ~3000 stars/sq-deg at b=0 (galactic plane, shallow survey)
# and ~50 stars/sq-deg at |b|=90 (poles).
_RHO0_PER_SQARCMIN = 3000.0 / 3600.0  # stars/sq-arcmin at b=0
_B_SCALE_DEG = 20.0                     # galactic latitude scale height in degrees
_TESS_PIXEL_SQARCMIN = 4.0 * 4.0       # TESS pixel ≈ 21 arcsec, ~4'×4' area


@dataclass(frozen=True)
class FieldStarDensityResult:
    galactic_lat_deg: float
    stars_per_sqarcmin: float
    stars_in_tess_pixel: float
    crowding_risk: str
    flag: str


def estimate_field_star_density(
    galactic_lat_deg: float,
    limiting_tmag: float = 14.0,
) -> FieldStarDensityResult:
    """
    Estimate stellar surface density from galactic latitude.

    Uses an exponential disk model calibrated to TESS-depth surveys.
    Magnitude limit scales density by 10^(0.6*(Tmag - 14)).

    crowding_risk: LOW (< 1 star/pixel), MODERATE (1–5), HIGH (> 5).
    """
    if not math.isfinite(galactic_lat_deg) or abs(galactic_lat_deg) > 90.0:
        return FieldStarDensityResult(
            galactic_lat_deg=galactic_lat_deg,
            stars_per_sqarcmin=float("nan"),
            stars_in_tess_pixel=float("nan"),
            crowding_risk="",
            flag="INVALID_GALACTIC_LAT",
        )
    if not math.isfinite(limiting_tmag):
        return FieldStarDensityResult(
            galactic_lat_deg=galactic_lat_deg,
            stars_per_sqarcmin=float("nan"),
            stars_in_tess_pixel=float("nan"),
            crowding_risk="",
            flag="INVALID_LIMITING_MAG",
        )

    b_abs = abs(galactic_lat_deg)
    density = _RHO0_PER_SQARCMIN * math.exp(-b_abs / _B_SCALE_DEG)

    # Scale for magnitude limit relative to reference Tmag=14
    mag_scale = 10.0 ** (0.6 * (limiting_tmag - 14.0))
    density *= mag_scale

    stars_per_pixel = density * _TESS_PIXEL_SQARCMIN

    if stars_per_pixel < 1.0:
        risk = "LOW"
    elif stars_per_pixel < 5.0:
        risk = "MODERATE"
    else:
        risk = "HIGH"

    return FieldStarDensityResult(
        galactic_lat_deg=galactic_lat_deg,
        stars_per_sqarcmin=round(density, 4),
        stars_in_tess_pixel=round(stars_per_pixel, 3),
        crowding_risk=risk,
        flag="OK",
    )


def format_field_density_result(r: FieldStarDensityResult) -> str:
    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| Galactic latitude (deg) | {r.galactic_lat_deg:.2f} |\n"
        f"| Density (stars/sq-arcmin) | {r.stars_per_sqarcmin:.4f} |\n"
        f"| Stars in TESS pixel | {r.stars_in_tess_pixel:.3f} |\n"
        f"| Crowding risk | {r.crowding_risk} |\n"
        f"| Flag | {r.flag} |\n"
    )


def _cli() -> int:
    p = argparse.ArgumentParser(description="Estimate field star density from galactic latitude.")
    p.add_argument("galactic_lat_deg", type=float)
    p.add_argument("--limiting-tmag", type=float, default=14.0)
    args = p.parse_args()
    r = estimate_field_star_density(args.galactic_lat_deg, args.limiting_tmag)
    print(format_field_density_result(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
