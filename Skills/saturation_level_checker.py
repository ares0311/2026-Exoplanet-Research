"""Estimate whether a target saturates TESS detector pixels.

Analytically derives the expected peak pixel ADU count from TESS magnitude
and compares it to the saturation threshold.  Fills a gap in the existing
tool chain — quality-bit tools flag post-hoc saturation, but this module
predicts saturation analytically before downloading data.

Public API
----------
SaturationResult(tmag, flux_ratio, peak_flux_adu, saturation_threshold_adu,
                 is_saturated, saturation_fraction, flag)
check_saturation(tmag, *, saturation_threshold_adu, exposure_sec,
                 zero_point_adu) -> SaturationResult
format_saturation_result(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# TESS instrument defaults
_DEFAULT_EXPOSURE_SEC = 120.0          # 2-minute cadence
_DEFAULT_SATURATION_ADU = 150_000.0   # approximate TESS full-well (16-bit, gain ~5)
_DEFAULT_ZERO_POINT_ADU = 1.5e8       # ADU/s for Tmag = 0 (scaled to 2-min cadence)


@dataclass(frozen=True)
class SaturationResult:
    tmag: float
    flux_ratio: float                  # ratio relative to Tmag=0 reference
    peak_flux_adu: float               # estimated peak pixel ADU
    saturation_threshold_adu: float
    is_saturated: bool
    saturation_fraction: float         # peak_flux / threshold (>1 = saturated)
    flag: str  # "OK" | "INVALID"


def check_saturation(
    tmag: float,
    *,
    saturation_threshold_adu: float = _DEFAULT_SATURATION_ADU,
    exposure_sec: float = _DEFAULT_EXPOSURE_SEC,
    zero_point_adu: float = _DEFAULT_ZERO_POINT_ADU,
) -> SaturationResult:
    """Estimate whether a star saturates the TESS detector.

    Uses the standard magnitude → flux relation:
    ``flux = zero_point × 10^(-0.4 × Tmag) × exposure_sec``

    The peak pixel flux is estimated as ~25 % of the total (assuming the PSF
    spreads over ~4 pixels in the brightest pixel for Tmag > 6).

    Args:
        tmag: TESS magnitude of the target.
        saturation_threshold_adu: ADU count that triggers pixel saturation.
        exposure_sec: Integration time per cadence (seconds).
        zero_point_adu: ADU/s for Tmag = 0 star (instrument zero-point).

    Returns:
        :class:`SaturationResult`.
    """
    if not math.isfinite(tmag):
        return SaturationResult(tmag, 0.0, 0.0, saturation_threshold_adu, False, 0.0, "INVALID")
    if saturation_threshold_adu <= 0 or exposure_sec <= 0:
        return SaturationResult(tmag, 0.0, 0.0, saturation_threshold_adu, False, 0.0, "INVALID")

    flux_ratio = 10.0 ** (-0.4 * tmag)
    total_adu = zero_point_adu * flux_ratio * exposure_sec
    # Peak pixel ≈ 25 % of total for a typical TESS PSF
    peak_fraction = 0.25
    peak_adu = total_adu * peak_fraction

    sat_frac = round(peak_adu / saturation_threshold_adu, 6)
    is_sat = sat_frac >= 1.0

    return SaturationResult(
        tmag=tmag,
        flux_ratio=round(flux_ratio, 8),
        peak_flux_adu=round(peak_adu, 2),
        saturation_threshold_adu=saturation_threshold_adu,
        is_saturated=is_sat,
        saturation_fraction=sat_frac,
        flag="OK",
    )


def format_saturation_result(result: SaturationResult) -> str:
    """Format saturation check result as Markdown."""
    sat_label = "**SATURATED**" if result.is_saturated else "Not saturated"
    lines = [
        "## Saturation Level Checker",
        "",
        f"- Tmag: {result.tmag}",
        f"- Peak pixel flux: {result.peak_flux_adu:.0f} ADU",
        f"- Saturation threshold: {result.saturation_threshold_adu:.0f} ADU",
        f"- Saturation fraction: {result.saturation_fraction:.3f}",
        f"- **Status: {sat_label}**",
        f"- **Flag: {result.flag}**",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="saturation_level_checker",
        description="Estimate TESS saturation level from Tmag.",
    )
    parser.add_argument("tmag", type=float)
    parser.add_argument("--threshold", type=float, default=_DEFAULT_SATURATION_ADU)
    args = parser.parse_args(argv)

    result = check_saturation(args.tmag, saturation_threshold_adu=args.threshold)
    print(format_saturation_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
