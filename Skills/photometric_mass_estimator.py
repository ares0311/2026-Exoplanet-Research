"""Estimate stellar mass from photometric colour without full isochrone fitting."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

# Mass–colour calibration for main-sequence stars (Pecaut & Mamajek 2013)
# log10(M/Msun) = a0 + a1*(B-V) + a2*(B-V)^2
# Valid 0.30 <= B-V <= 1.40 (FGK dwarfs)
_BV_MASS_COEFFS = (0.10, -0.68, 0.18)   # log10(Msun) polynomial in B-V
_BPRP_MASS_COEFFS = (0.08, -0.45, 0.06) # log10(Msun) polynomial in Bp-Rp (0.5–2.5)

_SUPPORTED = {"B-V", "Bp-Rp"}


@dataclass(frozen=True)
class PhotoMassResult:
    color_index: str
    color_value: float
    luminosity_class: str
    mass_msun: float
    mass_err_msun: float
    flag: str


def _poly(coeffs: tuple[float, ...], x: float) -> float:
    return sum(c * x**i for i, c in enumerate(coeffs))


def estimate_mass_from_photometry(
    color_index: str,
    color_value: float,
    luminosity_class: str = "V",
) -> PhotoMassResult:
    """
    Estimate stellar mass from photometric colour for main-sequence stars.

    Uses polynomial fits to M/Msun as function of B-V or Bp-Rp.
    Giants and supergiants are flagged OUT_OF_CALIBRATION.
    Error estimate: ~15% systematic for main-sequence; ~40% out of range.
    """
    if color_index not in _SUPPORTED:
        return PhotoMassResult(
            color_index=color_index, color_value=color_value,
            luminosity_class=luminosity_class,
            mass_msun=float("nan"), mass_err_msun=float("nan"),
            flag="UNKNOWN_COLOR_INDEX",
        )
    if not math.isfinite(color_value):
        return PhotoMassResult(
            color_index=color_index, color_value=color_value,
            luminosity_class=luminosity_class,
            mass_msun=float("nan"), mass_err_msun=float("nan"),
            flag="INVALID_COLOR_VALUE",
        )

    lc = luminosity_class.strip().upper()
    if lc in ("I", "II", "III", "IA", "IB"):
        return PhotoMassResult(
            color_index=color_index, color_value=color_value,
            luminosity_class=luminosity_class,
            mass_msun=float("nan"), mass_err_msun=float("nan"),
            flag="OUT_OF_CALIBRATION",
        )

    if color_index == "B-V":
        in_range = 0.30 <= color_value <= 1.40
        log_mass = _poly(_BV_MASS_COEFFS, color_value)
    else:
        in_range = 0.50 <= color_value <= 2.50
        log_mass = _poly(_BPRP_MASS_COEFFS, color_value)

    mass = 10.0 ** log_mass
    err_frac = 0.15 if in_range else 0.40
    mass_err = mass * err_frac

    flag = "OK" if in_range else "OUT_OF_RANGE"

    return PhotoMassResult(
        color_index=color_index,
        color_value=color_value,
        luminosity_class=luminosity_class,
        mass_msun=round(mass, 4),
        mass_err_msun=round(mass_err, 4),
        flag=flag,
    )


def format_photo_mass_result(r: PhotoMassResult) -> str:
    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| Color index | {r.color_index} |\n"
        f"| Color value | {r.color_value:.3f} |\n"
        f"| Luminosity class | {r.luminosity_class} |\n"
        f"| Mass (M☉) | {r.mass_msun:.4f} ± {r.mass_err_msun:.4f} |\n"
        f"| Flag | {r.flag} |\n"
    )


def _cli() -> int:
    p = argparse.ArgumentParser(description="Estimate stellar mass from photometric colour.")
    p.add_argument("color_index", choices=sorted(_SUPPORTED))
    p.add_argument("color_value", type=float)
    p.add_argument("--luminosity-class", default="V")
    args = p.parse_args()
    r = estimate_mass_from_photometry(args.color_index, args.color_value, args.luminosity_class)
    print(format_photo_mass_result(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
