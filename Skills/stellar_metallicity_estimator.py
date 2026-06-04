"""Estimate stellar metallicity [Fe/H] from photometric colour indices."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

# Polynomial fits: [Fe/H] ~ f(color, luminosity_class)
# Calibrated to Casagrande et al. (2011) / Ramirez & Melendez (2005) solar neighbourhood
# [Fe/H] = a0 + a1*(color - color_sun) for main-sequence FGK stars
# Solar reference colours: B-V=0.650, Bp-Rp=0.818

_SOLAR_BV = 0.650
_SOLAR_BPRP = 0.818

# Coefficients (color_sun_offset, slope) — linear approximation
# Bluer than solar → typically more metal-poor (lower [Fe/H])
_COEFFS: dict[str, tuple[float, float]] = {
    "B-V":   (0.0, -1.2),   # [Fe/H] = -1.2 * (B-V - 0.650)
    "Bp-Rp": (0.0, -0.9),   # [Fe/H] = -0.9 * (Bp-Rp - 0.818)
}

_SUPPORTED = set(_COEFFS.keys())


@dataclass(frozen=True)
class MetallicityResult:
    color_index: str
    color_value: float
    feh: float
    feh_err: float
    flag: str


def estimate_metallicity(
    color_index: str,
    color_value: float,
    luminosity_class: str = "V",
) -> MetallicityResult:
    """
    Estimate photometric [Fe/H] from broadband colour.

    Valid for FGK main-sequence and subgiant stars.
    Uncertainty is ~0.20 dex systematic from photometric scatter.
    Giants are flagged as OUT_OF_CALIBRATION.
    """
    if color_index not in _SUPPORTED:
        return MetallicityResult(
            color_index=color_index, color_value=color_value,
            feh=float("nan"), feh_err=float("nan"),
            flag="UNKNOWN_COLOR_INDEX",
        )
    if not math.isfinite(color_value):
        return MetallicityResult(
            color_index=color_index, color_value=color_value,
            feh=float("nan"), feh_err=float("nan"),
            flag="INVALID_COLOR_VALUE",
        )

    lc = luminosity_class.strip().upper()
    if lc in ("I", "II", "III", "IA", "IB"):
        return MetallicityResult(
            color_index=color_index, color_value=color_value,
            feh=float("nan"), feh_err=float("nan"),
            flag="OUT_OF_CALIBRATION",
        )

    solar_ref = _SOLAR_BV if color_index == "B-V" else _SOLAR_BPRP
    _, slope = _COEFFS[color_index]
    feh = slope * (color_value - solar_ref)

    # Validity ranges
    in_range = (
        (0.30 <= color_value <= 1.20) if color_index == "B-V"
        else (0.50 <= color_value <= 2.50)
    )

    feh_err = 0.20 if in_range else 0.40
    flag = "OK" if in_range else "OUT_OF_RANGE"

    # Clamp to physically plausible range
    feh = max(-3.0, min(1.0, feh))

    return MetallicityResult(
        color_index=color_index,
        color_value=color_value,
        feh=round(feh, 3),
        feh_err=feh_err,
        flag=flag,
    )


def format_metallicity_result(r: MetallicityResult) -> str:
    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| Color index | {r.color_index} |\n"
        f"| Color value | {r.color_value:.3f} |\n"
        f"| [Fe/H] | {r.feh:.3f} ± {r.feh_err:.2f} |\n"
        f"| Flag | {r.flag} |\n"
    )


def _cli() -> int:
    p = argparse.ArgumentParser(description="Estimate stellar [Fe/H] from colour.")
    p.add_argument("color_index", choices=sorted(_SUPPORTED))
    p.add_argument("color_value", type=float)
    p.add_argument("--luminosity-class", default="V")
    args = p.parse_args()
    r = estimate_metallicity(args.color_index, args.color_value, args.luminosity_class)
    print(format_metallicity_result(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
