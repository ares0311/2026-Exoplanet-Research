"""Estimate stellar activity level from light curve scatter metrics.

Computes a composite activity index from RMS, MAD, outlier fraction, and
peak-to-peak amplitude. Used for FP vetting: highly active stars mimic
transit-like signals.

Public API
----------
ActivityIndexResult(rms_ppm, mad_ppm, peak_to_peak_ppm, n_sigma_outliers,
                    activity_index, activity_level, flag)
compute_activity_index(flux, *, sigma_threshold, quiet_threshold,
                       active_threshold) -> ActivityIndexResult
format_activity_index(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ActivityIndexResult:
    rms_ppm: float
    mad_ppm: float
    peak_to_peak_ppm: float
    n_sigma_outliers: int
    activity_index: float
    activity_level: str  # "quiet" | "moderate" | "active" | "very_active"
    flag: str            # "OK" | "INSUFFICIENT" | "INVALID"


def _median(values: list[float]) -> float:
    n = len(values)
    s = sorted(values)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


def _rms(values: list[float], mean: float) -> float:
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def compute_activity_index(
    flux: list[float],
    *,
    sigma_threshold: float = 3.0,
    quiet_threshold: float = 0.25,
    active_threshold: float = 0.65,
) -> ActivityIndexResult:
    """Compute stellar activity index from a flux array.

    Args:
        flux: Relative flux values (e.g. normalised to ~1.0 or in ppm).
        sigma_threshold: Number of sigma to define outliers.
        quiet_threshold: Activity index below this → "quiet".
        active_threshold: Activity index above this → "very_active"; between → "active".

    Returns:
        :class:`ActivityIndexResult`.
    """
    if not isinstance(flux, (list, tuple)):
        return ActivityIndexResult(0.0, 0.0, 0.0, 0, 0.0, "quiet", "INVALID")
    clean = []
    for v in flux:
        try:
            f = float(v)
            if math.isfinite(f):
                clean.append(f)
        except (TypeError, ValueError):
            pass

    if len(clean) < 3:
        return ActivityIndexResult(0.0, 0.0, 0.0, 0, 0.0, "quiet", "INSUFFICIENT")

    mean_flux = sum(clean) / len(clean)
    rms_raw = _rms(clean, mean_flux)
    med = _median(clean)
    abs_devs = sorted(abs(v - med) for v in clean)
    mad_raw = _median(abs_devs)
    p2p_raw = max(clean) - min(clean)

    # Convert to ppm (assume flux is normalised to ~1.0)
    rms_ppm = rms_raw * 1e6
    mad_ppm = mad_raw * 1e6
    p2p_ppm = p2p_raw * 1e6

    # Sigma for outlier detection
    sigma = rms_raw if rms_raw > 0 else 1e-12
    n_outliers = sum(1 for v in clean if abs(v - mean_flux) > sigma_threshold * sigma)

    # Composite sub-scores
    rms_sub = _clip(rms_ppm / 5000.0)
    mad_sub = _clip(mad_ppm / 3000.0)
    outlier_sub = _clip(n_outliers / 20.0)
    p2p_sub = _clip(p2p_ppm / 50000.0)

    activity_index = (
        0.40 * rms_sub
        + 0.30 * mad_sub
        + 0.20 * outlier_sub
        + 0.10 * p2p_sub
    )
    activity_index = round(_clip(activity_index), 6)

    if activity_index < quiet_threshold:
        level = "quiet"
    elif activity_index < 0.50:
        level = "moderate"
    elif activity_index < active_threshold:
        level = "active"
    else:
        level = "very_active"

    return ActivityIndexResult(
        rms_ppm=round(rms_ppm, 3),
        mad_ppm=round(mad_ppm, 3),
        peak_to_peak_ppm=round(p2p_ppm, 3),
        n_sigma_outliers=n_outliers,
        activity_index=activity_index,
        activity_level=level,
        flag="OK",
    )


def format_activity_index(result: ActivityIndexResult) -> str:
    """Format activity index result as Markdown."""
    lines = [
        "## Stellar Activity Index",
        "",
        f"- RMS: {result.rms_ppm:.1f} ppm",
        f"- MAD: {result.mad_ppm:.1f} ppm",
        f"- Peak-to-peak: {result.peak_to_peak_ppm:.1f} ppm",
        f"- Sigma outliers ({result.n_sigma_outliers})",
        f"- **Activity index: {result.activity_index:.4f}**",
        f"- **Activity level: {result.activity_level}**",
        f"- **Flag: {result.flag}**",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="stellar_activity_index",
        description="Estimate stellar activity from flux scatter.",
    )
    parser.add_argument(
        "flux",
        nargs="+",
        type=float,
        help="Flux values (normalised to ~1.0)",
    )
    parser.add_argument("--sigma-threshold", type=float, default=3.0)
    args = parser.parse_args(argv)

    result = compute_activity_index(
        args.flux,
        sigma_threshold=args.sigma_threshold,
    )
    print(format_activity_index(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
