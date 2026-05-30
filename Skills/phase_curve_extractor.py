"""Extract a phase curve from time/flux arrays and fit a sinusoidal model.

Phase-folds the data using period and epoch, bins into n_bins bins, then
computes the amplitude and phase offset of the sinusoidal component at
frequency 1/period via a DFT (Goertzel-style direct computation).

Public API
----------
PhaseCurveResult(amplitude_ppm, phase_offset_rad, baseline_flux, n_bins, flag)
extract_phase_curve(time, flux, period_days, epoch_bjd, n_bins) -> PhaseCurveResult
format_phase_curve_result(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PhaseCurveResult:
    amplitude_ppm: float
    phase_offset_rad: float
    baseline_flux: float
    n_bins: int
    flag: str = "OK"


def _phase_fold(time: list[float], period: float, epoch: float) -> list[float]:
    """Return phases in [-0.5, 0.5)."""
    phases = []
    for t in time:
        ph = ((t - epoch) / period) % 1.0
        if ph >= 0.5:
            ph -= 1.0
        phases.append(ph)
    return phases


def extract_phase_curve(
    time: list[float],
    flux: list[float],
    period_days: float,
    epoch_bjd: float,
    n_bins: int = 50,
) -> PhaseCurveResult:
    """Extract phase curve and fit sinusoidal model.

    Args:
        time: Time array (BJD or similar).
        flux: Flux array (normalised, median ~1).
        period_days: Orbital period in days.
        epoch_bjd: Reference epoch.
        n_bins: Number of phase bins.

    Returns:
        :class:`PhaseCurveResult`.
    """
    if len(time) < 2 or len(flux) != len(time) or period_days <= 0:
        return PhaseCurveResult(
            amplitude_ppm=0.0, phase_offset_rad=0.0, baseline_flux=1.0,
            n_bins=n_bins, flag="ERROR",
        )

    phases = _phase_fold(time, period_days, epoch_bjd)

    # Bin the data
    bin_sums = [0.0] * n_bins
    bin_counts = [0] * n_bins
    for ph, f in zip(phases, flux, strict=False):
        idx = int((ph + 0.5) * n_bins) % n_bins
        bin_sums[idx] += f
        bin_counts[idx] += 1

    binned: list[float] = []
    bin_phases: list[float] = []
    for i in range(n_bins):
        if bin_counts[i] > 0:
            binned.append(bin_sums[i] / bin_counts[i])
            bin_phases.append((i + 0.5) / n_bins - 0.5)

    if not binned:
        return PhaseCurveResult(
            amplitude_ppm=0.0, phase_offset_rad=0.0, baseline_flux=1.0,
            n_bins=n_bins, flag="WARNING",
        )

    baseline = sum(binned) / len(binned)
    residuals = [b - baseline for b in binned]

    # DFT at frequency 1 (one full cycle across phase range)
    n = len(bin_phases)
    cos_sum = sum(
        r * math.cos(2.0 * math.pi * ph) for r, ph in zip(residuals, bin_phases, strict=False)
    )
    sin_sum = sum(
        r * math.sin(2.0 * math.pi * ph) for r, ph in zip(residuals, bin_phases, strict=False)
    )
    a = 2.0 * cos_sum / n
    b = 2.0 * sin_sum / n
    amplitude = math.sqrt(a**2 + b**2)
    phase_offset = math.atan2(b, a)

    amplitude_ppm = amplitude * 1e6

    return PhaseCurveResult(
        amplitude_ppm=round(amplitude_ppm, 3),
        phase_offset_rad=round(phase_offset, 6),
        baseline_flux=round(baseline, 8),
        n_bins=len(binned),
        flag="OK",
    )


def format_phase_curve_result(result: PhaseCurveResult) -> str:
    """Format phase curve result as Markdown."""
    lines = [
        "## Phase Curve",
        "",
        f"- Amplitude: **{result.amplitude_ppm:.2f} ppm**",
        f"- Phase offset: **{result.phase_offset_rad:.4f} rad**",
        f"- Baseline flux: {result.baseline_flux:.6f}",
        f"- Bins used: {result.n_bins}",
        f"- Flag: {result.flag}",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(
        prog="phase_curve_extractor",
        description="Extract phase curve from time/flux arrays.",
    )
    parser.add_argument("period_days", type=float)
    parser.add_argument("epoch_bjd", type=float)
    parser.add_argument("--time", type=str, default="[]", help="JSON list of times")
    parser.add_argument("--flux", type=str, default="[]", help="JSON list of fluxes")
    parser.add_argument("--n-bins", type=int, default=50)
    args = parser.parse_args(argv)

    time = json.loads(args.time)
    flux = json.loads(args.flux)
    result = extract_phase_curve(time, flux, args.period_days, args.epoch_bjd, args.n_bins)
    print(format_phase_curve_result(result))
    return 0 if result.flag == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
