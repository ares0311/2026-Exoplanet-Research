"""Check a light curve for photometric binary signatures.

Ellipsoidal variations (period = P_orb/2) and Doppler boosting (period = P_orb)
are hallmarks of an eclipsing binary or hot Jupiter with significant tidal
deformation.  This module phase-folds the light curve at P/2 and P and
measures the peak-to-peak amplitude at each period.

Public API
----------
PhotometricBinaryResult(half_period_amplitude_ppm, full_period_amplitude_ppm,
                        ellipsoidal_snr, is_binary_candidate, flag)
check_photometric_binary(time, flux, period_days, epoch_bjd, *,
                         flux_err, n_bins,
                         snr_threshold) -> PhotometricBinaryResult
format_binary_check_result(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PhotometricBinaryResult:
    half_period_amplitude_ppm: float | None   # peak-to-peak at P/2
    full_period_amplitude_ppm: float | None   # peak-to-peak at P
    ellipsoidal_snr: float | None             # amplitude / noise per bin
    is_binary_candidate: bool
    flag: str  # "OK" | "INSUFFICIENT" | "INVALID"


def _phase_fold_bin(
    time: list[float], flux: list[float], period: float,
    epoch: float, n_bins: int
) -> tuple[list[float], list[float]]:
    """Phase-fold and bin a light curve."""
    bins_sum = [0.0] * n_bins
    bins_cnt = [0] * n_bins
    for t, f in zip(time, flux, strict=False):
        ph = ((t - epoch) % period) / period
        idx = min(int(ph * n_bins), n_bins - 1)
        bins_sum[idx] += f
        bins_cnt[idx] += 1
    phases = [(i + 0.5) / n_bins for i in range(n_bins)]
    means = [bins_sum[i] / bins_cnt[i] if bins_cnt[i] > 0 else float("nan")
             for i in range(n_bins)]
    valid = [(ph, m) for ph, m in zip(phases, means, strict=False) if not math.isnan(m)]
    if not valid:
        return [], []
    ph_v, m_v = zip(*valid, strict=False)
    return list(ph_v), list(m_v)


def check_photometric_binary(
    time: list[float],
    flux: list[float],
    period_days: float,
    epoch_bjd: float,
    *,
    flux_err: list[float] | None = None,
    n_bins: int = 20,
    snr_threshold: float = 3.0,
) -> PhotometricBinaryResult:
    """Check for ellipsoidal variation at P/2.

    Args:
        time: Time array (days).
        flux: Normalised flux array.
        period_days: Orbital period in days.
        epoch_bjd: Reference epoch.
        flux_err: Per-point uncertainties.
        n_bins: Number of phase bins.
        snr_threshold: SNR threshold for binary flag.

    Returns:
        :class:`PhotometricBinaryResult`.
    """
    n = len(time)
    if n < 10 or len(flux) != n or period_days <= 0:
        return PhotometricBinaryResult(None, None, None, False, "INVALID")

    # Noise estimate
    if flux_err and len(flux_err) == n:
        noise = sum(flux_err) / n
    else:
        m = sum(flux) / n
        vals = sorted(abs(f - m) for f in flux)
        noise = vals[n // 2] * 1.4826 / math.sqrt(max(n // n_bins, 1))
    noise = max(noise, 1e-9)

    # Phase-fold at P (full period)
    _, full_means = _phase_fold_bin(time, flux, period_days, epoch_bjd, n_bins)
    if not full_means:
        return PhotometricBinaryResult(None, None, None, False, "INSUFFICIENT")

    full_amp = (max(full_means) - min(full_means)) * 1e6

    # Phase-fold at P/2 (half period — ellipsoidal)
    _, half_means = _phase_fold_bin(time, flux, period_days / 2.0, epoch_bjd, n_bins)
    if not half_means:
        return PhotometricBinaryResult(None, None, full_amp, False, "INSUFFICIENT")

    half_amp = (max(half_means) - min(half_means)) * 1e6

    noise_ppm = noise * 1e6
    ellipsoidal_snr = half_amp / noise_ppm if noise_ppm > 0 else None

    is_binary = ellipsoidal_snr is not None and ellipsoidal_snr >= snr_threshold

    return PhotometricBinaryResult(
        half_period_amplitude_ppm=round(half_amp, 2),
        full_period_amplitude_ppm=round(full_amp, 2),
        ellipsoidal_snr=round(ellipsoidal_snr, 3) if ellipsoidal_snr is not None else None,
        is_binary_candidate=is_binary,
        flag="OK",
    )


def format_binary_check_result(result: PhotometricBinaryResult) -> str:
    """Format photometric binary check as Markdown."""
    lines = [
        "## Photometric Binary Check",
        "",
        f"- Half-period amplitude: {result.half_period_amplitude_ppm} ppm",
        f"- Full-period amplitude: {result.full_period_amplitude_ppm} ppm",
        f"- Ellipsoidal SNR: {result.ellipsoidal_snr}",
        f"- Binary candidate: {'Yes' if result.is_binary_candidate else 'No'}",
        f"- **Flag: {result.flag}**",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="photometric_binary_checker",
        description="Check for ellipsoidal variation at P/2.",
    )
    parser.add_argument("period_days", type=float)
    parser.add_argument("epoch_bjd", type=float)
    args = parser.parse_args(argv)

    result = check_photometric_binary([], [], args.period_days, args.epoch_bjd)
    print(format_binary_check_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
