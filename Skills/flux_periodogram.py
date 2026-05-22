"""Compute a DFT-based power spectrum from an unevenly sampled light curve.

Uses direct evaluation of the discrete Fourier transform on a user-defined
frequency grid — no numpy or astropy required.  Suitable for detecting
stellar rotation periods and activity cycles.

Public API
----------
PeriodogramPeak(frequency, period_days, power)
PeriodogramResult(freq_grid, power, peaks, flag)
compute_dft_periodogram(time, flux, *, freq_min, freq_max,
                        n_freqs, flux_err) -> PeriodogramResult
find_periodogram_peaks(result, *, n_peaks, min_power) -> list[PeriodogramPeak]
format_periodogram_result(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PeriodogramPeak:
    frequency: float    # cycles/day
    period_days: float
    power: float        # normalised [0, 1]


@dataclass(frozen=True)
class PeriodogramResult:
    freq_grid: tuple[float, ...]    # cycles/day
    power: tuple[float, ...]        # normalised [0, 1]
    peaks: tuple[PeriodogramPeak, ...]
    flag: str  # "OK" | "INSUFFICIENT" | "INVALID"


def compute_dft_periodogram(
    time: list[float],
    flux: list[float],
    *,
    freq_min: float = 0.02,
    freq_max: float = 1.0,
    n_freqs: int = 300,
    flux_err: list[float] | None = None,
) -> PeriodogramResult:
    """Compute DFT-based periodogram (Lomb-Scargle approximation).

    Evaluates the normalised squared DFT amplitude at each frequency in the
    grid.  Weights are uniform unless ``flux_err`` is provided (inverse-variance
    weighting).

    Args:
        time: Time array (days).
        flux: Flux array, same length as time.
        freq_min: Minimum frequency (cycles/day).
        freq_max: Maximum frequency (cycles/day).
        n_freqs: Number of frequency grid points.
        flux_err: Optional per-point flux uncertainties.

    Returns:
        :class:`PeriodogramResult`.
    """
    n = len(time)
    if len(flux) != n:
        return PeriodogramResult((), (), (), "INVALID")
    if n < 5:
        return PeriodogramResult((), (), (), "INVALID")
    if freq_min <= 0 or freq_max <= freq_min or n_freqs < 2:
        return PeriodogramResult((), (), (), "INVALID")

    # Weights
    if flux_err and len(flux_err) == n:
        w = [1.0 / (e ** 2) if e > 0 else 0.0 for e in flux_err]
        wsum = sum(w)
        if wsum <= 0:
            w = [1.0] * n
            wsum = float(n)
    else:
        w = [1.0] * n
        wsum = float(n)

    # Weighted mean-subtracted flux
    mean_f = sum(w[i] * flux[i] for i in range(n)) / wsum
    fz = [flux[i] - mean_f for i in range(n)]

    # Weighted variance (for normalisation)
    var = sum(w[i] * fz[i] ** 2 for i in range(n)) / wsum
    if var < 1e-30:
        return PeriodogramResult((), (), (), "INSUFFICIENT")

    df = (freq_max - freq_min) / (n_freqs - 1)
    freqs: list[float] = [freq_min + k * df for k in range(n_freqs)]
    powers: list[float] = []

    for freq in freqs:
        omega = 2.0 * math.pi * freq
        # Lomb-Scargle style: power = (C² + S²) / var / N_eff
        tau_num = sum(w[i] * math.sin(2 * omega * time[i]) for i in range(n))
        tau_den = sum(w[i] * math.cos(2 * omega * time[i]) for i in range(n))
        _nonzero = abs(tau_den) > 1e-30 or abs(tau_num) > 1e-30
        tau = math.atan2(tau_num, tau_den) / (2 * omega) if _nonzero else 0.0

        cos_t = [math.cos(omega * (time[i] - tau)) for i in range(n)]
        sin_t = [math.sin(omega * (time[i] - tau)) for i in range(n)]

        c_sum = sum(w[i] * fz[i] * cos_t[i] for i in range(n))
        s_sum = sum(w[i] * fz[i] * sin_t[i] for i in range(n))
        c_norm = sum(w[i] * cos_t[i] ** 2 for i in range(n))
        s_norm = sum(w[i] * sin_t[i] ** 2 for i in range(n))

        p = 0.0
        if c_norm > 1e-30:
            p += c_sum ** 2 / c_norm
        if s_norm > 1e-30:
            p += s_sum ** 2 / s_norm
        powers.append(p / (2.0 * var * wsum))

    # Normalise to [0, 1]
    max_p = max(powers) if powers else 1.0
    if max_p < 1e-30:
        max_p = 1.0
    norm_powers = [p / max_p for p in powers]

    # Find peaks (local maxima)
    peaks: list[PeriodogramPeak] = []
    for i in range(1, len(freqs) - 1):
        if norm_powers[i] >= norm_powers[i - 1] and norm_powers[i] >= norm_powers[i + 1]:
            peaks.append(PeriodogramPeak(
                frequency=round(freqs[i], 6),
                period_days=round(1.0 / freqs[i], 4),
                power=round(norm_powers[i], 6),
            ))
    peaks.sort(key=lambda p: p.power, reverse=True)

    return PeriodogramResult(
        freq_grid=tuple(round(f, 6) for f in freqs),
        power=tuple(round(p, 6) for p in norm_powers),
        peaks=tuple(peaks[:20]),
        flag="OK",
    )


def find_periodogram_peaks(
    result: PeriodogramResult,
    *,
    n_peaks: int = 5,
    min_power: float = 0.1,
) -> list[PeriodogramPeak]:
    """Return the top N peaks above a minimum power threshold.

    Args:
        result: Output of :func:`compute_dft_periodogram`.
        n_peaks: Maximum number of peaks to return.
        min_power: Minimum normalised power to include.

    Returns:
        List of :class:`PeriodogramPeak` sorted by power descending.
    """
    if result.flag != "OK":
        return []
    return [p for p in result.peaks if p.power >= min_power][:n_peaks]


def format_periodogram_result(result: PeriodogramResult) -> str:
    """Format periodogram result as Markdown."""
    if result.flag != "OK":
        return f"## DFT Periodogram\n\n_Flag: {result.flag}_\n"

    peaks = find_periodogram_peaks(result, n_peaks=5, min_power=0.1)
    lines = [
        "## DFT Periodogram",
        "",
        f"- Frequencies evaluated: {len(result.freq_grid)}",
        f"- Significant peaks found: {len(peaks)}",
        f"- **Flag: {result.flag}**",
    ]
    if peaks:
        lines += ["", "| Period (d) | Frequency (c/d) | Normalised power |",
                  "|---|---|---|"]
        for pk in peaks:
            lines.append(f"| {pk.period_days:.4f} | {pk.frequency:.6f} | {pk.power:.4f} |")
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="flux_periodogram",
        description="Compute DFT-based power spectrum from light curve.",
    )
    parser.add_argument("--freq-min", type=float, default=0.02)
    parser.add_argument("--freq-max", type=float, default=1.0)
    parser.add_argument("--n-freqs", type=int, default=300)
    args = parser.parse_args(argv)

    result = compute_dft_periodogram(
        [], [], freq_min=args.freq_min, freq_max=args.freq_max, n_freqs=args.n_freqs
    )
    print(format_periodogram_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
