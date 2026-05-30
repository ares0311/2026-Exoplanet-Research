"""Remove a sinusoidal stellar activity signal at the rotation period.

Public API:
    ActivityCorrectionResult  -- frozen dataclass
    correct_for_stellar_activity(flux, time_days, rotation_period_days) -> ActivityCorrectionResult
    format_activity_correction(result) -> str
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ActivityCorrectionResult:
    n_points: int
    amplitude: float
    phase_rad: float
    rms_before: float
    rms_after: float
    corrected_flux: list[float]
    flag: str


def _rms(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))


def correct_for_stellar_activity(
    flux: list[float],
    time_days: list[float],
    rotation_period_days: float,
) -> ActivityCorrectionResult:
    n = min(len(flux), len(time_days))
    if n < 4:
        return ActivityCorrectionResult(
            n_points=n, amplitude=0.0, phase_rad=0.0,
            rms_before=0.0, rms_after=0.0, corrected_flux=list(flux),
            flag="INSUFFICIENT_DATA",
        )
    if rotation_period_days <= 0:
        return ActivityCorrectionResult(
            n_points=n, amplitude=0.0, phase_rad=0.0,
            rms_before=0.0, rms_after=0.0, corrected_flux=list(flux),
            flag="INVALID_PERIOD",
        )
    t = time_days[:n]
    f = flux[:n]
    omega = 2.0 * math.pi / rotation_period_days
    sin_t = [math.sin(omega * ti) for ti in t]
    cos_t = [math.cos(omega * ti) for ti in t]
    ss = sum(s * s for s in sin_t)
    sc = sum(s * c for s, c in zip(sin_t, cos_t, strict=False))
    cc = sum(c * c for c in cos_t)
    sf = sum(s * fi for s, fi in zip(sin_t, f, strict=False))
    cf = sum(c * fi for c, fi in zip(cos_t, f, strict=False))
    det = ss * cc - sc * sc
    if abs(det) < 1e-12:
        return ActivityCorrectionResult(
            n_points=n, amplitude=0.0, phase_rad=0.0,
            rms_before=_rms(f), rms_after=_rms(f), corrected_flux=list(f),
            flag="DEGENERATE",
        )
    a = (cc * sf - sc * cf) / det
    b = (ss * cf - sc * sf) / det
    amplitude = math.sqrt(a * a + b * b)
    phase_rad = math.atan2(b, a)
    rms_before = _rms(f)
    corrected = [fi - (a * si + b * ci) for fi, si, ci in zip(f, sin_t, cos_t, strict=False)]
    rms_after = _rms(corrected)
    return ActivityCorrectionResult(
        n_points=n,
        amplitude=amplitude,
        phase_rad=phase_rad,
        rms_before=rms_before,
        rms_after=rms_after,
        corrected_flux=corrected,
        flag="OK",
    )


def format_activity_correction(result: ActivityCorrectionResult) -> str:
    lines = [
        "## Stellar Activity Correction",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| N Points | {result.n_points} |",
        f"| Amplitude | {result.amplitude:.6f} |",
        f"| Phase (rad) | {result.phase_rad:.4f} |",
        f"| RMS Before | {result.rms_before:.6f} |",
        f"| RMS After | {result.rms_after:.6f} |",
        f"| Flag | {result.flag} |",
    ]
    return "\n".join(lines)


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Correct for stellar activity signal.")
    parser.add_argument("flux_file", help="JSON file of flux values.")
    parser.add_argument("time_file", help="JSON file of time values.")
    parser.add_argument("rotation_period_days", type=float)
    args = parser.parse_args()
    with open(args.flux_file) as fh:
        flux = json.load(fh)
    with open(args.time_file) as fh:
        time_days = json.load(fh)
    result = correct_for_stellar_activity(flux, time_days, args.rotation_period_days)
    print(format_activity_correction(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
