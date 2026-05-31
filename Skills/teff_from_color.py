"""Estimate stellar effective temperature from broadband colour indices."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

# Polynomial fits: Teff = sum(c_i * color^i)
# B-V → Teff: fit to Pecaut & Mamajek (2013) table, FGK range 0.3 ≤ B-V ≤ 1.4
_BV_COEFFS = (9549.0, -7966.0, 5765.0, -2155.0, 316.0)
# Bp-Rp → Teff: calibrated to match Andrae et al. (2018) Gaia DR2 range 0.5 ≤ Bp-Rp ≤ 3.0
_BPRP_COEFFS = (8017.0, -3544.0, 894.0, -98.0, 4.0)

_SUPPORTED_INDICES = {"B-V", "Bp-Rp"}


@dataclass(frozen=True)
class TeffFromColorResult:
    color_index: str
    color_value: float
    teff_k: float
    teff_err_k: float
    flag: str


def _poly(coeffs: tuple[float, ...], x: float) -> float:
    return sum(c * x**i for i, c in enumerate(coeffs))


def estimate_teff_from_color(
    color_index: str,
    color_value: float,
) -> TeffFromColorResult:
    """
    Estimate Teff from B-V or Gaia Bp-Rp colour index.

    Uses polynomial fits valid for FGK main-sequence stars.
    Uncertainty is ~150 K systematic from scatter in the calibration sample.
    """
    if color_index not in _SUPPORTED_INDICES:
        return TeffFromColorResult(
            color_index=color_index,
            color_value=color_value,
            teff_k=float("nan"),
            teff_err_k=float("nan"),
            flag="UNKNOWN_COLOR_INDEX",
        )
    if not math.isfinite(color_value):
        return TeffFromColorResult(
            color_index=color_index,
            color_value=color_value,
            teff_k=float("nan"),
            teff_err_k=float("nan"),
            flag="INVALID_COLOR_VALUE",
        )

    if color_index == "B-V":
        in_range = 0.3 <= color_value <= 1.4
        teff = _poly(_BV_COEFFS, color_value)
    else:  # Bp-Rp
        in_range = 0.5 <= color_value <= 3.0
        teff = _poly(_BPRP_COEFFS, color_value)

    flag = "OK" if in_range else "OUT_OF_RANGE"
    teff_err = 150.0 if in_range else 300.0

    if teff < 2000.0 or teff > 50000.0:
        return TeffFromColorResult(
            color_index=color_index,
            color_value=color_value,
            teff_k=round(teff, 1),
            teff_err_k=teff_err,
            flag="IMPLAUSIBLE_TEFF",
        )

    return TeffFromColorResult(
        color_index=color_index,
        color_value=color_value,
        teff_k=round(teff, 1),
        teff_err_k=teff_err,
        flag=flag,
    )


def format_teff_result(r: TeffFromColorResult) -> str:
    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| Color index | {r.color_index} |\n"
        f"| Color value | {r.color_value:.3f} |\n"
        f"| Teff (K) | {r.teff_k:.1f} ± {r.teff_err_k:.0f} |\n"
        f"| Flag | {r.flag} |\n"
    )


def _cli() -> int:
    p = argparse.ArgumentParser(description="Estimate Teff from colour index.")
    p.add_argument("color_index", choices=sorted(_SUPPORTED_INDICES))
    p.add_argument("color_value", type=float)
    args = p.parse_args()
    r = estimate_teff_from_color(args.color_index, args.color_value)
    print(format_teff_result(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
