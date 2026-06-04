"""Estimate stellar RV jitter from activity indicators."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

# Saar & Donahue (1997) / Wright (2005) empirical relations
# log10(sigma_jitter [m/s]) ~ a + b * log10(v_sin_i) + c * (B-V)
# Simplified single-parameter forms calibrated to solar neighbourhood

_TEFF_SUN = 5778.0
_BV_SUN = 0.650


@dataclass(frozen=True)
class RvJitterResult:
    jitter_ms: float
    jitter_err_ms: float
    activity_level: str   # QUIET / MODERATE / ACTIVE
    dominant_source: str  # OSCILLATION / GRANULATION / ROTATION / UNKNOWN
    flag: str


def estimate_rv_jitter(
    bv_color: float | None = None,
    prot_days: float | None = None,
    vsini_kms: float | None = None,
    teff_k: float | None = None,
) -> RvJitterResult:
    """
    Estimate intrinsic stellar RV jitter from available activity indicators.

    Uses empirical scaling relations:
    - Rotation: sigma ~ 3.0 * (25 / Prot)^0.9  (Wright 2005)
    - v sin i:  sigma ~ 1.0 * vsini^0.5         (Saar & Donahue 1997 simplified)
    - B-V:      proxy for spectral type, sets granulation floor
    - Teff:     fallback granulation estimate (Kjeldsen & Bedding 1995)

    Returns lowest jitter (most conservative) when multiple diagnostics available.
    """
    estimates: list[float] = []
    dominant = "UNKNOWN"

    if prot_days is not None and math.isfinite(prot_days) and prot_days > 0:
        sigma_rot = 3.0 * (25.0 / prot_days) ** 0.9
        estimates.append(sigma_rot)
        dominant = "ROTATION"

    if vsini_kms is not None and math.isfinite(vsini_kms) and vsini_kms > 0:
        sigma_vsini = 1.0 * math.sqrt(vsini_kms)
        estimates.append(sigma_vsini)
        if not estimates or sigma_vsini > max(estimates[:-1], default=0.0):
            dominant = "ROTATION"

    if bv_color is not None and math.isfinite(bv_color):
        # Granulation floor from colour proxy: redder → more jitter
        sigma_gran = 0.4 + 1.5 * max(0.0, bv_color - _BV_SUN)
        estimates.append(sigma_gran)
        if dominant == "UNKNOWN":
            dominant = "GRANULATION"

    if teff_k is not None and math.isfinite(teff_k) and teff_k > 0:
        # Kjeldsen & Bedding (1995): v_osc ~ 23.4 * (L/Lsun)^0.7 / (M/Msun)^1.0
        # Simplified: sigma_osc ~ 0.3 * (Tsun/Teff)^4 m/s (approximate)
        sigma_osc = 0.3 * (_TEFF_SUN / teff_k) ** 2.0
        estimates.append(sigma_osc)
        if dominant == "UNKNOWN":
            dominant = "OSCILLATION"

    if not estimates:
        return RvJitterResult(
            jitter_ms=float("nan"), jitter_err_ms=float("nan"),
            activity_level="UNKNOWN", dominant_source="UNKNOWN",
            flag="NO_DIAGNOSTICS",
        )

    jitter = sum(estimates) / len(estimates)

    if jitter < 1.5:
        level = "QUIET"
    elif jitter < 5.0:
        level = "MODERATE"
    else:
        level = "ACTIVE"

    jitter_err = jitter * 0.50

    return RvJitterResult(
        jitter_ms=round(jitter, 3),
        jitter_err_ms=round(jitter_err, 3),
        activity_level=level,
        dominant_source=dominant,
        flag="OK",
    )


def format_rv_jitter_result(r: RvJitterResult) -> str:
    jit = f"{r.jitter_ms:.3f} ± {r.jitter_err_ms:.3f}" if math.isfinite(r.jitter_ms) else "N/A"
    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| RV jitter (m/s) | {jit} |\n"
        f"| Activity level | {r.activity_level} |\n"
        f"| Dominant source | {r.dominant_source} |\n"
        f"| Flag | {r.flag} |\n"
    )


def _cli() -> int:
    p = argparse.ArgumentParser(description="Estimate stellar RV jitter.")
    p.add_argument("--bv-color", type=float, default=None)
    p.add_argument("--prot-days", type=float, default=None)
    p.add_argument("--vsini-kms", type=float, default=None)
    p.add_argument("--teff-k", type=float, default=None)
    args = p.parse_args()
    r = estimate_rv_jitter(args.bv_color, args.prot_days, args.vsini_kms, args.teff_k)
    print(format_rv_jitter_result(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
