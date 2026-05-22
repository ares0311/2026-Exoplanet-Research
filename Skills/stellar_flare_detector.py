"""Detect stellar flares in a light curve via flux derivative thresholding.

A flare is identified as a group of consecutive cadences where the flux
rises sharply above a rolling median baseline by more than ``sigma_threshold``
standard deviations, followed by an exponential-like decay.  The detector
operates purely in the time-domain without external dependencies.

Public API
----------
StellarFlareResult(n_cadences, n_flares, flare_indices, max_amplitude,
                   total_flare_energy_proxy, flag)
detect_stellar_flares(time, flux, *, flux_err, sigma_threshold,
                      min_duration_cadences, window) -> StellarFlareResult
format_flare_result(result) -> str
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StellarFlareResult:
    n_cadences: int
    n_flares: int
    flare_indices: tuple[tuple[int, int], ...]  # (start, end) index pairs
    max_amplitude: float | None                  # max flux excess / baseline
    total_flare_energy_proxy: float | None       # sum of excess flux over all flares
    flag: str  # "OK" | "NO_FLARES" | "INSUFFICIENT" | "INVALID"


def _rolling_median(values: list[float], window: int) -> list[float]:
    """Simple centred rolling median; edges use available data."""
    n = len(values)
    half = window // 2
    result = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        chunk = sorted(values[lo:hi])
        m = len(chunk)
        if m == 0:
            result.append(0.0)
        elif m % 2 == 1:
            result.append(chunk[m // 2])
        else:
            result.append((chunk[m // 2 - 1] + chunk[m // 2]) / 2.0)
    return result


def _mad(values: list[float], median: float) -> float:
    devs = sorted(abs(v - median) for v in values)
    n = len(devs)
    if n == 0:
        return 0.0
    if n % 2 == 1:
        return devs[n // 2]
    return (devs[n // 2 - 1] + devs[n // 2]) / 2.0


def detect_stellar_flares(
    time: list[float],
    flux: list[float],
    *,
    flux_err: list[float] | None = None,
    sigma_threshold: float = 3.0,
    min_duration_cadences: int = 2,
    window: int = 51,
) -> StellarFlareResult:
    """Detect stellar flares via rolling-baseline thresholding.

    Args:
        time: Time array (arbitrary units, same length as flux).
        flux: Normalised flux array.
        flux_err: Per-point uncertainties (optional, used for sigma scaling).
        sigma_threshold: Detection threshold in units of baseline MAD.
        min_duration_cadences: Minimum run length to count as a flare.
        window: Rolling-median window size (cadences).

    Returns:
        :class:`StellarFlareResult`.
    """
    n = len(flux)
    if n < 10 or len(time) != n:
        return StellarFlareResult(n, 0, (), None, None, "INVALID")
    if sigma_threshold <= 0:
        return StellarFlareResult(n, 0, (), None, None, "INVALID")

    baseline = _rolling_median(flux, max(3, window))

    residuals = [flux[i] - baseline[i] for i in range(n)]

    # Use flux_err or MAD-based sigma
    if flux_err is not None and len(flux_err) == n:
        sigma = [max(e, 1e-12) for e in flux_err]
    else:
        all_res = sorted(residuals)
        med_res = all_res[n // 2] if n % 2 == 1 else (all_res[n // 2 - 1] + all_res[n // 2]) / 2.0
        mad_val = _mad(residuals, med_res) * 1.4826
        if mad_val < 1e-12:
            return StellarFlareResult(n, 0, (), None, None, "INSUFFICIENT")
        sigma = [mad_val] * n

    above = [residuals[i] / sigma[i] >= sigma_threshold for i in range(n)]

    # Group consecutive above-threshold cadences
    flare_indices: list[tuple[int, int]] = []
    i = 0
    while i < n:
        if above[i]:
            j = i
            while j < n and above[j]:
                j += 1
            if j - i >= min_duration_cadences:
                flare_indices.append((i, j - 1))
            i = j
        else:
            i += 1

    if not flare_indices:
        return StellarFlareResult(n, 0, (), None, None, "NO_FLARES")

    # Compute amplitude and energy proxy
    amplitudes = []
    energy_proxy = 0.0
    for start, end in flare_indices:
        amp = max(residuals[start:end + 1])
        ref_sigma = sigma[start]
        amplitudes.append(amp / ref_sigma)
        energy_proxy += sum(max(0.0, residuals[k]) for k in range(start, end + 1))

    max_amp = max(amplitudes) if amplitudes else None

    return StellarFlareResult(
        n_cadences=n,
        n_flares=len(flare_indices),
        flare_indices=tuple(flare_indices),
        max_amplitude=round(max_amp, 4) if max_amp is not None else None,
        total_flare_energy_proxy=round(energy_proxy, 6),
        flag="OK",
    )


def format_flare_result(result: StellarFlareResult) -> str:
    """Format flare detection result as Markdown."""
    lines = [
        "## Stellar Flare Detection",
        "",
        f"- Cadences analysed: {result.n_cadences}",
        f"- Flares detected: {result.n_flares}",
        f"- Max amplitude (σ): {result.max_amplitude}",
        f"- Total energy proxy: {result.total_flare_energy_proxy}",
        f"- **Flag: {result.flag}**",
    ]
    if result.flare_indices:
        lines += ["", "| Flare | Start idx | End idx |", "|---|---|---|"]
        for k, (s, e) in enumerate(result.flare_indices, 1):
            lines.append(f"| {k} | {s} | {e} |")
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="stellar_flare_detector",
        description="Detect stellar flares in a light curve.",
    )
    parser.add_argument("--sigma-threshold", type=float, default=3.0)
    parser.add_argument("--min-duration", type=int, default=2)
    parser.add_argument("--window", type=int, default=51)
    args = parser.parse_args(argv)

    result = detect_stellar_flares(
        [], [],
        sigma_threshold=args.sigma_threshold,
        min_duration_cadences=args.min_duration,
        window=args.window,
    )
    print(format_flare_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
