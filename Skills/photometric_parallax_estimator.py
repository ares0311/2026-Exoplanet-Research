"""Estimate photometric parallax distance from apparent and absolute magnitude."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

# Absolute magnitude calibrations for main-sequence (luminosity class V)
# From Pecaut & Mamajek (2013): M_V as function of B-V for FGK stars
# M_V ≈ a0 + a1*(B-V) + a2*(B-V)^2   for 0.30 <= B-V <= 1.40
_MV_COEFFS = (1.24, 4.60, -1.15)   # polynomial in B-V

# Absolute magnitude in Gaia G as function of Bp-Rp (Gaia DR2 main sequence)
_MG_COEFFS = (3.45, 1.80, -0.20)   # polynomial in Bp-Rp, 0.5 <= Bp-Rp <= 2.5


@dataclass(frozen=True)
class PhotoParallaxResult:
    apparent_mag: float
    color_index: str
    color_value: float
    abs_mag_est: float
    distance_pc: float
    distance_err_pc: float
    flag: str


def _poly(coeffs: tuple[float, ...], x: float) -> float:
    return sum(c * x**i for i, c in enumerate(coeffs))


def estimate_parallax_distance(
    apparent_mag: float,
    color_index: str,
    color_value: float,
    extinction_mag: float = 0.0,
) -> PhotoParallaxResult:
    """
    Estimate distance from photometric parallax.

    distance_modulus = apparent_mag - extinction - abs_mag
    distance_pc = 10^((DM + 5) / 5)

    Supported colour indices: B-V, Bp-Rp. Valid for FGK main-sequence stars.
    """
    if not math.isfinite(apparent_mag):
        return PhotoParallaxResult(
            apparent_mag=apparent_mag, color_index=color_index,
            color_value=color_value, abs_mag_est=float("nan"),
            distance_pc=float("nan"), distance_err_pc=float("nan"),
            flag="INVALID_APPARENT_MAG",
        )
    if not math.isfinite(color_value):
        return PhotoParallaxResult(
            apparent_mag=apparent_mag, color_index=color_index,
            color_value=color_value, abs_mag_est=float("nan"),
            distance_pc=float("nan"), distance_err_pc=float("nan"),
            flag="INVALID_COLOR_VALUE",
        )

    if color_index == "B-V":
        in_range = 0.30 <= color_value <= 1.40
        abs_mag = _poly(_MV_COEFFS, color_value)
    elif color_index == "Bp-Rp":
        in_range = 0.50 <= color_value <= 2.50
        abs_mag = _poly(_MG_COEFFS, color_value)
    else:
        return PhotoParallaxResult(
            apparent_mag=apparent_mag, color_index=color_index,
            color_value=color_value, abs_mag_est=float("nan"),
            distance_pc=float("nan"), distance_err_pc=float("nan"),
            flag="UNKNOWN_COLOR_INDEX",
        )

    ext = extinction_mag if math.isfinite(extinction_mag) else 0.0
    dm = apparent_mag - ext - abs_mag
    distance_pc = 10.0 ** ((dm + 5.0) / 5.0)

    # ~0.5 mag systematic → ~23% distance error
    dist_err = distance_pc * 0.23
    if not in_range:
        dist_err *= 2.0

    flag = "OK" if in_range else "OUT_OF_RANGE"

    return PhotoParallaxResult(
        apparent_mag=apparent_mag,
        color_index=color_index,
        color_value=color_value,
        abs_mag_est=round(abs_mag, 3),
        distance_pc=round(distance_pc, 2),
        distance_err_pc=round(dist_err, 2),
        flag=flag,
    )


def format_parallax_result(r: PhotoParallaxResult) -> str:
    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| Apparent mag | {r.apparent_mag:.3f} |\n"
        f"| Color ({r.color_index}) | {r.color_value:.3f} |\n"
        f"| Abs mag (est) | {r.abs_mag_est:.3f} |\n"
        f"| Distance (pc) | {r.distance_pc:.2f} ± {r.distance_err_pc:.2f} |\n"
        f"| Flag | {r.flag} |\n"
    )


def _cli() -> int:
    p = argparse.ArgumentParser(description="Estimate photometric parallax distance.")
    p.add_argument("apparent_mag", type=float)
    p.add_argument("color_index", choices=["B-V", "Bp-Rp"])
    p.add_argument("color_value", type=float)
    p.add_argument("--extinction-mag", type=float, default=0.0)
    args = p.parse_args()
    r = estimate_parallax_distance(
        args.apparent_mag, args.color_index, args.color_value, args.extinction_mag
    )
    print(format_parallax_result(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
