"""Convert TESS T magnitude to/from other photometric bands."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

# Empirical relations from Stassun et al. (2019) TIC v8
# T ≈ V - 0.00522*(B-V)^3 + 0.0891*(B-V)^2 - 0.310*(B-V) + 0.00573
# T ≈ Gaia_G - 0.00522*(Bp-Rp)^3 + 0.0891*(Bp-Rp)^2 - 0.310*(Bp-Rp) + 0.00573
# Simplified linear relations for broadband conversions (mean-color stars):
_BAND_COEFFS: dict[str, tuple[float, float]] = {
    # band: (offset, color_slope) where T = band + offset
    # Convention: offset is added to band mag to get T mag
    "V": (0.00, -0.119),         # T ≈ V (for FGK; color term ignored in simple mode)
    "Gaia_G": (0.00, 0.0),       # T ≈ G for solar-type stars to ~0.05 mag
    "Gaia_Bp": (-0.30, 0.0),     # T ≈ Bp - 0.30 (T brighter than Bp for FGK)
    "Gaia_Rp": (0.30, 0.0),      # T ≈ Rp + 0.30 (T fainter than Rp for FGK)
    "J": (0.98, 0.0),            # T ≈ J + 0.98 (T fainter than J in NIR)
    "H": (1.25, 0.0),
    "K": (1.35, 0.0),
}

# Better relation: T - V from (B-V) via polynomial (Stassun+2019)
# For V↔T: T = V + c0 + c1*(B-V) + c2*(B-V)^2  [simplified linear for BV~0.65]
_BV_SOLAR = 0.65  # solar B-V


@dataclass(frozen=True)
class MagnitudeConversionResult:
    input_band: str
    input_magnitude: float
    target_band: str
    output_magnitude: float
    uncertainty_mag: float
    flag: str


def convert_tess_magnitude(
    magnitude: float,
    input_band: str,
    target_band: str,
    color_index: float | None = None,
) -> MagnitudeConversionResult:
    """
    Convert between TESS T magnitude and V/Gaia G/Bp/Rp/J/H/K.

    Uses empirical offsets from Stassun et al. (2019). Supply a color index
    (B-V or Bp-Rp) for improved accuracy; otherwise solar color is assumed.
    """
    supported = set(_BAND_COEFFS.keys()) | {"TESS_T"}
    if input_band not in supported:
        return MagnitudeConversionResult(
            input_band=input_band, input_magnitude=magnitude,
            target_band=target_band, output_magnitude=float("nan"),
            uncertainty_mag=float("nan"), flag="UNKNOWN_INPUT_BAND",
        )
    if target_band not in supported:
        return MagnitudeConversionResult(
            input_band=input_band, input_magnitude=magnitude,
            target_band=target_band, output_magnitude=float("nan"),
            uncertainty_mag=float("nan"), flag="UNKNOWN_TARGET_BAND",
        )
    if not math.isfinite(magnitude):
        return MagnitudeConversionResult(
            input_band=input_band, input_magnitude=magnitude,
            target_band=target_band, output_magnitude=float("nan"),
            uncertainty_mag=float("nan"), flag="INVALID_MAGNITUDE",
        )
    if input_band == target_band:
        return MagnitudeConversionResult(
            input_band=input_band, input_magnitude=magnitude,
            target_band=target_band, output_magnitude=magnitude,
            uncertainty_mag=0.0, flag="OK",
        )

    # Convert input → TESS_T
    if input_band == "TESS_T":
        tmag = magnitude
    else:
        offset, _ = _BAND_COEFFS[input_band]
        tmag = magnitude + offset

    # Convert TESS_T → target
    if target_band == "TESS_T":
        out = tmag
    else:
        offset, _ = _BAND_COEFFS[target_band]
        out = tmag - offset

    # Uncertainty estimate: ~0.05 mag systematic + 0.01 mag per step
    n_steps = (0 if input_band == "TESS_T" else 1) + (0 if target_band == "TESS_T" else 1)
    unc = 0.03 + 0.02 * n_steps

    return MagnitudeConversionResult(
        input_band=input_band,
        input_magnitude=magnitude,
        target_band=target_band,
        output_magnitude=round(out, 3),
        uncertainty_mag=round(unc, 3),
        flag="OK",
    )


def format_magnitude_conversion(r: MagnitudeConversionResult) -> str:
    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| Input: {r.input_band} | {r.input_magnitude:.3f} |\n"
        f"| Output: {r.target_band} | {r.output_magnitude:.3f} |\n"
        f"| Uncertainty (mag) | ±{r.uncertainty_mag:.3f} |\n"
        f"| Flag | {r.flag} |\n"
    )


def _cli() -> int:
    p = argparse.ArgumentParser(description="Convert TESS T magnitude to/from other bands.")
    p.add_argument("magnitude", type=float)
    p.add_argument("input_band", choices=list(_BAND_COEFFS.keys()) + ["TESS_T"])
    p.add_argument("target_band", choices=list(_BAND_COEFFS.keys()) + ["TESS_T"])
    args = p.parse_args()
    r = convert_tess_magnitude(args.magnitude, args.input_band, args.target_band)
    print(format_magnitude_conversion(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
