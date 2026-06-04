"""Estimate convective zone depth from stellar effective temperature."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

# Approximate convective zone depth (R_conv / R_star) as a function of Teff
# Based on Pinsonneault et al. (2001) / Cranmer & Saar (2011) calibrations:
# Fully convective: Teff < 3900 K (M dwarfs) → Rconv/R* = 1.0
# Transition zone: 3900–6000 K (K dwarfs and late G) → Rconv/R* decreasing
# Shallow CZ: 6000–7500 K (early G, F dwarfs) → Rconv/R* ~0.05–0.20
# Radiative envelopes: Teff > 7500 K (A stars and hotter) → Rconv/R* ~ 0.01 (core)

_TEFF_FULLY_CONVECTIVE = 3900.0
_TEFF_TRANSITION_END = 6000.0
_TEFF_SHALLOW_END = 7500.0
_RCZ_FULLY_CONV = 1.0
_RCZ_AT_6000K = 0.22
_RCZ_AT_7500K = 0.05
_RCZ_HOT = 0.01   # convective core only


@dataclass(frozen=True)
class ConvectiveZoneResult:
    teff_k: float
    rcz_over_rstar: float    # convective zone depth / stellar radius
    convective_type: str     # FULLY_CONVECTIVE / DEEP_CZ / SHALLOW_CZ / RADIATIVE
    dynamo_active: bool      # convective zone deep enough for solar-type dynamo
    flag: str


def estimate_convective_zone(teff_k: float) -> ConvectiveZoneResult:
    """
    Estimate fractional convective zone depth from effective temperature.

    Returns Rconv/R* and a convective type classification.
    Dynamo_active is True when Rconv/R* > 0.10 (solar-type alpha-omega dynamo).
    """
    if not math.isfinite(teff_k) or teff_k <= 0:
        return ConvectiveZoneResult(
            teff_k=teff_k, rcz_over_rstar=float("nan"),
            convective_type="UNKNOWN", dynamo_active=False, flag="INVALID_TEFF",
        )

    if teff_k < _TEFF_FULLY_CONVECTIVE:
        rcz = _RCZ_FULLY_CONV
        conv_type = "FULLY_CONVECTIVE"
    elif teff_k < _TEFF_TRANSITION_END:
        # Linear interpolation from 1.0 at 3900K to 0.22 at 6000K
        frac = (teff_k - _TEFF_FULLY_CONVECTIVE) / (
            _TEFF_TRANSITION_END - _TEFF_FULLY_CONVECTIVE
        )
        rcz = _RCZ_FULLY_CONV + frac * (_RCZ_AT_6000K - _RCZ_FULLY_CONV)
        conv_type = "DEEP_CZ"
    elif teff_k < _TEFF_SHALLOW_END:
        # Linear interpolation from 0.22 at 6000K to 0.05 at 7500K
        frac = (teff_k - _TEFF_TRANSITION_END) / (
            _TEFF_SHALLOW_END - _TEFF_TRANSITION_END
        )
        rcz = _RCZ_AT_6000K + frac * (_RCZ_AT_7500K - _RCZ_AT_6000K)
        conv_type = "SHALLOW_CZ"
    else:
        rcz = _RCZ_HOT
        conv_type = "RADIATIVE"

    dynamo = rcz > 0.10

    return ConvectiveZoneResult(
        teff_k=teff_k,
        rcz_over_rstar=round(rcz, 4),
        convective_type=conv_type,
        dynamo_active=dynamo,
        flag="OK",
    )


def format_convective_zone_result(r: ConvectiveZoneResult) -> str:
    rcz_str = f"{r.rcz_over_rstar:.4f}" if math.isfinite(r.rcz_over_rstar) else "N/A"
    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| Teff (K) | {r.teff_k:.1f} |\n"
        f"| Rconv / R* | {rcz_str} |\n"
        f"| Convective type | {r.convective_type} |\n"
        f"| Dynamo active | {r.dynamo_active} |\n"
        f"| Flag | {r.flag} |\n"
    )


def _cli() -> int:
    p = argparse.ArgumentParser(description="Estimate stellar convective zone depth.")
    p.add_argument("teff_k", type=float)
    args = p.parse_args()
    r = estimate_convective_zone(args.teff_k)
    print(format_convective_zone_result(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
