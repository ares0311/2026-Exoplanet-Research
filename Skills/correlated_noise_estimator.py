"""Estimate correlated (red) noise in a light curve via the beta method.

Bins the light curve into progressively larger time bins and checks whether
the RMS falls as ``1/sqrt(N)`` (white noise) or more slowly (red noise).
Derives a beta factor where ``RMS(N) ∝ N^(-beta/2)``; beta = 1 is pure
white noise, beta < 1 indicates red/correlated noise.

Public API
----------
CorrelatedNoiseResult(rms_white_ppm, rms_red_ppm, beta_factor,
                      timescale_hours, n_bins_used, flag)
estimate_correlated_noise(time, flux, *, bin_sizes_hours,
                          flux_is_normalised) -> CorrelatedNoiseResult
format_correlated_noise_result(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CorrelatedNoiseResult:
    rms_white_ppm: float | None    # RMS at single-cadence binning
    rms_red_ppm: float | None      # RMS extrapolated to 1-hour bins
    beta_factor: float | None      # exponent: 1.0 = pure white, <1 = red noise
    timescale_hours: float | None  # bin width where beta plateaus (hours)
    n_bins_used: int
    flag: str  # "OK" | "INSUFFICIENT" | "INVALID"


def _bin_rms(time: list[float], flux: list[float], bin_width_days: float) -> float | None:
    """Compute RMS of binned flux.  Returns None if fewer than 2 bins."""
    if bin_width_days <= 0 or not time:
        return None
    t0 = time[0]
    bins: dict[int, list[float]] = {}
    for t, f in zip(time, flux, strict=False):
        bi = int((t - t0) / bin_width_days)
        bins.setdefault(bi, []).append(f)

    if len(bins) < 2:
        return None

    means = [sum(v) / len(v) for v in bins.values()]
    grand_mean = sum(means) / len(means)
    variance = sum((m - grand_mean) ** 2 for m in means) / len(means)
    return math.sqrt(variance)


def estimate_correlated_noise(
    time: list[float],
    flux: list[float],
    *,
    bin_sizes_hours: list[float] | None = None,
    flux_is_normalised: bool = True,
) -> CorrelatedNoiseResult:
    """Estimate red noise via the RMS-vs-bin-width (beta) method.

    Args:
        time: Time array (days), sorted ascending.
        flux: Flux array (normalised to ~1 if ``flux_is_normalised``).
        bin_sizes_hours: List of bin widths to test (hours).
            Defaults to ``[0.5, 1.0, 2.0, 4.0]``.
        flux_is_normalised: If True, flux values near 1; RMS reported in ppm.

    Returns:
        :class:`CorrelatedNoiseResult`.
    """
    if len(time) != len(flux):
        return CorrelatedNoiseResult(None, None, None, None, 0, "INVALID")
    if len(time) < 10:
        return CorrelatedNoiseResult(None, None, None, None, 0, "INSUFFICIENT")

    bins_h = bin_sizes_hours if bin_sizes_hours is not None else [0.5, 1.0, 2.0, 4.0]
    scale = 1e6 if flux_is_normalised else 1.0

    rms_vals: list[float] = []
    bin_widths_used: list[float] = []

    for bh in sorted(bins_h):
        bw_days = bh / 24.0
        rms = _bin_rms(time, flux, bw_days)
        if rms is not None:
            rms_vals.append(rms * scale)
            bin_widths_used.append(bh)

    if len(rms_vals) < 2:
        return CorrelatedNoiseResult(None, None, None, None, 0, "INSUFFICIENT")

    # Fit log(RMS) = log(A) - (beta/2)*log(N_bins)
    # Equivalently: log(RMS) = log(A) + (beta/2)*log(bin_width)
    # (wider bins → fewer data points per bin → RMS should be higher if red noise)
    log_bw = [math.log(bh) for bh in bin_widths_used]
    log_rms = [math.log(r) if r > 0 else float("-inf") for r in rms_vals]

    # Simple two-point or least-squares fit
    n = len(log_bw)
    sx = sum(log_bw)
    sy = sum(log_rms)
    sxx = sum(x ** 2 for x in log_bw)
    sxy = sum(log_bw[i] * log_rms[i] for i in range(n))
    denom = n * sxx - sx ** 2
    if abs(denom) < 1e-15:
        beta = 1.0
    else:
        slope = (n * sxy - sx * sy) / denom
        beta = max(0.0, min(2.0, 2.0 * slope + 1.0))

    rms_white = rms_vals[0]  # smallest bin ≈ single cadence
    rms_red = rms_vals[-1]   # largest bin
    timescale_h = bin_widths_used[-1]

    return CorrelatedNoiseResult(
        rms_white_ppm=round(rms_white, 4),
        rms_red_ppm=round(rms_red, 4),
        beta_factor=round(beta, 4),
        timescale_hours=round(timescale_h, 3),
        n_bins_used=len(rms_vals),
        flag="OK",
    )


def format_correlated_noise_result(result: CorrelatedNoiseResult) -> str:
    """Format correlated noise estimate as Markdown."""
    lines = [
        "## Correlated Noise Estimator",
        "",
        f"- White noise RMS: {result.rms_white_ppm} ppm",
        f"- Red noise RMS (at {result.timescale_hours} h): {result.rms_red_ppm} ppm",
        f"- **Beta factor: {result.beta_factor}** (1.0 = pure white, <1 = red noise)",
        f"- Timescale: {result.timescale_hours} hours",
        f"- Bin sizes tested: {result.n_bins_used}",
        f"- **Flag: {result.flag}**",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="correlated_noise_estimator",
        description="Estimate red/correlated noise via RMS-vs-bin-width method.",
    )
    parser.add_argument("--n-points", type=int, default=100)
    parser.parse_args(argv)

    result = estimate_correlated_noise([], [])
    print(format_correlated_noise_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
