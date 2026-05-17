"""Compare detrending window lengths and recommend the best one.

Runs clean_lightcurve with multiple Savitzky-Golay window lengths and
evaluates scatter preservation and injected signal recovery.

Public API
----------
compare_detrending(time, flux, flux_err, *, windows, period_days, depth_ppm,
                   duration_days, clean_fn) -> DetrendingReport
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class DetrendingReport:
    recommended_window: int
    scatter_by_window: dict[int, float]
    signal_snr_by_window: dict[int, float]
    reason: str


def _inject_box_transit(
    time: np.ndarray,
    flux: np.ndarray,
    period_days: float,
    epoch_bjd: float,
    depth_ppm: float,
    duration_days: float,
) -> np.ndarray:
    """Add a synthetic box transit to flux."""
    flux_out = flux.copy()
    phase = ((time - epoch_bjd) % period_days) / period_days
    in_transit = (phase < duration_days / period_days / 2.0) | (
        phase > 1.0 - duration_days / period_days / 2.0
    )
    flux_out[in_transit] -= depth_ppm * 1e-6
    return flux_out


def _measure_snr(
    time: np.ndarray,
    flux: np.ndarray,
    period_days: float,
    epoch_bjd: float,
    duration_days: float,
) -> float:
    """Estimate per-transit SNR via phase folding."""
    phase = ((time - epoch_bjd) % period_days) / period_days
    half_dur = duration_days / period_days / 2.0
    in_t = (phase < half_dur) | (phase > 1.0 - half_dur)
    out_t = ~in_t
    if np.sum(in_t) < 3 or np.sum(out_t) < 3:
        return 0.0
    delta = float(np.median(flux[out_t])) - float(np.median(flux[in_t]))
    sigma = float(np.std(flux[out_t])) / np.sqrt(float(np.sum(in_t)))
    return delta / max(sigma, 1e-12)


def compare_detrending(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray | None = None,
    *,
    windows: tuple[int, ...] = (51, 101, 201),
    period_days: float = 10.0,
    depth_ppm: float = 1000.0,
    duration_days: float = 0.0833,
    epoch_bjd: float | None = None,
    clean_fn: Callable[..., Any] | None = None,
) -> DetrendingReport:
    """Compare detrending windows and pick the one with the best SNR.

    Injects a synthetic box transit, then detrends with each window length.
    The window that preserves the most transit signal (highest SNR) is
    recommended.

    Args:
        time: Time array (BJD days).
        flux: Relative flux array (mean ≈ 1.0).
        flux_err: Per-point uncertainties (optional).
        windows: Savitzky-Golay window lengths to evaluate (odd integers).
        period_days: Period of the injected test signal.
        depth_ppm: Depth of the injected test signal.
        duration_days: Duration of the injected test signal.
        epoch_bjd: Epoch for injection (defaults to time[0]).
        clean_fn: Injectable clean function ``(mock_lc, provenance) -> CleanResult``.
            If ``None``, uses direct numpy SG filtering for isolation.

    Returns:
        :class:`DetrendingReport` with recommended window and per-window metrics.
    """
    from scipy.signal import savgol_filter  # noqa: PLC0415

    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    epoch = epoch_bjd if epoch_bjd is not None else float(time[0])

    flux_injected = _inject_box_transit(
        time, flux, period_days, epoch, depth_ppm, duration_days
    )

    scatter_by: dict[int, float] = {}
    snr_by: dict[int, float] = {}

    for w in windows:
        # Clamp window to odd number and valid range
        actual_w = max(5, min(int(w) | 1, len(time) - 1 if len(time) % 2 == 0 else len(time)))
        if actual_w % 2 == 0:
            actual_w += 1
        if actual_w >= len(time):
            actual_w = max(5, len(time) - 2)
            if actual_w % 2 == 0:
                actual_w -= 1

        try:
            trend = savgol_filter(flux_injected, actual_w, 2)
            detrended = flux_injected / np.where(np.abs(trend) > 1e-12, trend, 1.0)
        except Exception:
            scatter_by[w] = float("inf")
            snr_by[w] = 0.0
            continue

        # Out-of-transit scatter
        phase = ((time - epoch) % period_days) / period_days
        half_dur = duration_days / period_days / 2.0
        oot = ~((phase < half_dur) | (phase > 1.0 - half_dur))
        scatter = float(np.std(detrended[oot])) if np.sum(oot) > 3 else float("inf")
        scatter_by[w] = scatter

        snr = _measure_snr(time, detrended, period_days, epoch, duration_days)
        snr_by[w] = snr

    # Recommend window with best SNR (ties broken by smaller scatter)
    best_w = max(windows, key=lambda w: (snr_by.get(w, 0.0), -scatter_by.get(w, 1e9)))
    reason = (
        f"Window {best_w} gives SNR {snr_by.get(best_w, 0.0):.1f} "
        f"(scatter {scatter_by.get(best_w, 0.0):.5f})"
    )

    return DetrendingReport(
        recommended_window=best_w,
        scatter_by_window=scatter_by,
        signal_snr_by_window=snr_by,
        reason=reason,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="detrending_comparator",
        description="Compare detrending window lengths for a light curve.",
    )
    parser.add_argument("--period", type=float, required=True, metavar="DAYS")
    parser.add_argument("--depth", type=float, default=1000.0, metavar="PPM")
    parser.add_argument("--duration", type=float, default=0.0833, metavar="DAYS")
    args = parser.parse_args(argv)

    print("Detrending comparator requires a loaded numpy light curve.")
    print(f"Use the library API: compare_detrending(time, flux, period_days={args.period})")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
