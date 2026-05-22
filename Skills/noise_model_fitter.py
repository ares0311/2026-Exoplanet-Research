"""Fit a simple noise model to out-of-transit flux residuals.

Decomposes the photometric noise into a white-noise component (photon + read)
and a red-noise component (correlated systematics) by comparing the RMS on
different time-averaging scales.

Public API
----------
NoiseModelResult(white_noise_ppm, red_noise_ppm, combined_noise_ppm,
                 beta_factor, n_bins_used, flag)
fit_noise_model(flux_oot, cadence_minutes, *,
                bin_durations_minutes) -> NoiseModelResult
format_noise_model_result(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class NoiseModelResult:
    white_noise_ppm: float | None   # estimated white-noise floor in ppm
    red_noise_ppm: float | None     # estimated correlated noise in ppm
    combined_noise_ppm: float | None  # sqrt(white^2 + red^2)
    beta_factor: float | None       # ratio of observed to theoretical binned noise
    n_bins_used: int
    flag: str  # "OK" | "INSUFFICIENT" | "INVALID"


def _rms(values: list[float]) -> float:
    n = len(values)
    if n == 0:
        return 0.0
    m = sum(values) / n
    return math.sqrt(sum((v - m) ** 2 for v in values) / n)


def _bin_rms(flux: list[float], bin_size: int) -> float | None:
    n = len(flux)
    if n < bin_size or bin_size < 1:
        return None
    n_bins = n // bin_size
    if n_bins < 2:
        return None
    bins = []
    for k in range(n_bins):
        chunk = flux[k * bin_size: (k + 1) * bin_size]
        bins.append(sum(chunk) / len(chunk))
    return _rms(bins)


def fit_noise_model(
    flux_oot: list[float],
    cadence_minutes: float = 2.0,
    *,
    bin_durations_minutes: list[float] | None = None,
) -> NoiseModelResult:
    """Fit white + red noise model to out-of-transit flux.

    Args:
        flux_oot: Out-of-transit normalised flux array.
        cadence_minutes: Cadence of individual observations in minutes.
        bin_durations_minutes: Bin durations to test (minutes).
            Defaults to [cadence, 2×, 5×, 10×, 30×].

    Returns:
        :class:`NoiseModelResult`.
    """
    n = len(flux_oot)
    if n < 10:
        return NoiseModelResult(None, None, None, None, 0, "INSUFFICIENT")
    if cadence_minutes <= 0:
        return NoiseModelResult(None, None, None, None, 0, "INVALID")

    if bin_durations_minutes is None:
        bin_durations_minutes = [cadence_minutes, 2 * cadence_minutes, 5 * cadence_minutes,
                                  10 * cadence_minutes, 30 * cadence_minutes]

    # Unbinned RMS = white noise estimate (in fractional units)
    white_frac = _rms(flux_oot)
    if white_frac < 1e-12:
        return NoiseModelResult(None, None, None, None, 0, "INSUFFICIENT")
    white_ppm = white_frac * 1e6

    # Bin at multiple scales to estimate beta (Pont et al. 2006)
    betas: list[float] = []
    n_bins_used = 0
    for dur in bin_durations_minutes:
        bin_size = max(1, round(dur / cadence_minutes))
        binned = _bin_rms(flux_oot, bin_size)
        if binned is None:
            continue
        theoretical = white_frac / math.sqrt(bin_size)
        if theoretical > 1e-20:
            betas.append(binned / theoretical)
            n_bins_used += 1

    if not betas:
        return NoiseModelResult(
            white_noise_ppm=round(white_ppm, 2),
            red_noise_ppm=None,
            combined_noise_ppm=round(white_ppm, 2),
            beta_factor=None,
            n_bins_used=0,
            flag="INSUFFICIENT",
        )

    beta = sum(betas) / len(betas)
    # Red noise: sigma_red = white * sqrt(max(beta^2 - 1, 0))
    red_frac = white_frac * math.sqrt(max(beta ** 2 - 1.0, 0.0))
    red_ppm = red_frac * 1e6
    combined_ppm = math.sqrt(white_ppm ** 2 + red_ppm ** 2)

    return NoiseModelResult(
        white_noise_ppm=round(white_ppm, 2),
        red_noise_ppm=round(red_ppm, 2),
        combined_noise_ppm=round(combined_ppm, 2),
        beta_factor=round(beta, 4),
        n_bins_used=n_bins_used,
        flag="OK",
    )


def format_noise_model_result(result: NoiseModelResult) -> str:
    """Format noise model result as Markdown."""
    lines = [
        "## Noise Model Fit",
        "",
        f"- White noise: {result.white_noise_ppm} ppm",
        f"- Red noise: {result.red_noise_ppm} ppm",
        f"- Combined noise: {result.combined_noise_ppm} ppm",
        f"- Beta factor: {result.beta_factor}",
        f"- Bin scales used: {result.n_bins_used}",
        f"- **Flag: {result.flag}**",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="noise_model_fitter",
        description="Fit white + red noise model to out-of-transit flux.",
    )
    parser.add_argument("--cadence-minutes", type=float, default=2.0)
    args = parser.parse_args(argv)

    result = fit_noise_model([], cadence_minutes=args.cadence_minutes)
    print(format_noise_model_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
