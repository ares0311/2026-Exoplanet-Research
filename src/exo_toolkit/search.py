"""BLS transit search on a cleaned light curve.

Extracts time/flux arrays from the lightkurve LightCurve object, then
runs astropy's BoxLeastSquares directly.  Multiple signals can be found
via iterative transit masking: after each peak is recorded, its transit
windows are excluded from the next BLS run.

Only three attributes are accessed on the incoming LightCurve:
    lc.time.jd        — array of BJD time values
    lc.flux.value     — normalised flux array
    lc.flux_err.value — per-cadence flux uncertainty (optional)

This makes the module testable with lightweight mocks while keeping the
BLS computation real and calibrated.

Public API
----------
search_lightcurve(lc, target_id, mission, *, ...) → list[CandidateSignal]
"""
from __future__ import annotations

import math
from typing import Any

import astropy.units as u
import numpy as np
from astropy.timeseries import BoxLeastSquares

from exo_toolkit.schemas import CandidateSignal, Mission

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def search_lightcurve(
    lc: Any,
    target_id: str,
    mission: Mission,
    *,
    period_min: float = 0.5,
    period_max: float | None = None,
    duration_min_hours: float = 0.5,
    duration_max_hours: float = 12.0,
    n_durations: int = 20,
    min_snr: float = 5.0,
    max_peaks: int = 5,
) -> list[CandidateSignal]:
    """Search a cleaned light curve for periodic transit signals via BLS.

    Args:
        lc: A lightkurve.LightCurve object (from clean_lightcurve()).
            Must expose lc.time.jd, lc.flux.value, and optionally
            lc.flux_err.value.
        target_id: Identifier string for the target (e.g. "TIC 123456789").
        mission: "TESS", "Kepler", or "K2".
        period_min: Minimum trial period in days.
        period_max: Maximum trial period in days.  When None, defaults to
            half the light curve baseline (requires at least 2 transits).
        duration_min_hours: Shortest transit duration to search (hours).
        duration_max_hours: Longest transit duration to search (hours).
        n_durations: Number of duration steps in the BLS grid.
        min_snr: Minimum depth SNR to accept a peak as a candidate signal.
            Peaks below this threshold terminate the search.
        max_peaks: Maximum number of signals to extract (iterative masking
            stops after this many peaks regardless of SNR).

    Returns:
        List of CandidateSignal objects ordered by descending BLS power.
        An empty list is returned when no peak exceeds min_snr.
    """
    time_bjd = np.asarray(lc.time.jd, dtype=float)
    flux = np.asarray(lc.flux.value, dtype=float)
    flux_err = _extract_flux_err(lc, flux)

    if len(time_bjd) < 3:
        return []

    resolved_period_max = (
        period_max
        if period_max is not None
        else max(period_min + 0.1, (time_bjd[-1] - time_bjd[0]) / 2.0)
    )

    # astropy BLS requires max(duration) < min(period); cap to 90% of period_min
    eff_duration_max_hours = min(duration_max_hours, period_min * 24.0 * 0.9)
    duration_grid = np.linspace(
        duration_min_hours / 24.0,
        eff_duration_max_hours / 24.0,
        n_durations,
    )

    signals: list[CandidateSignal] = []
    include = np.ones(len(time_bjd), dtype=bool)

    for peak_idx in range(max_peaks):
        if int(include.sum()) < 3:
            break

        t = time_bjd[include]
        f = flux[include]
        e = flux_err[include]

        bls = BoxLeastSquares(t * u.day, f, dy=e)
        period_grid = bls.autoperiod(
            duration_grid * u.day,
            minimum_period=period_min * u.day,
            maximum_period=resolved_period_max * u.day,
            minimum_n_transit=2,
        )

        if len(period_grid) == 0:
            break

        result = bls.power(period=period_grid, duration=duration_grid * u.day)
        best = int(np.argmax(result.power))

        period_days = float(result.period[best].to(u.day).value)
        epoch_bjd = float(result.transit_time[best].to(u.day).value)
        duration_hours = float(result.duration[best].to(u.hour).value)
        depth = float(abs(result.depth[best]))
        depth_err = float(result.depth_err[best])
        snr = depth / depth_err if depth_err > 0.0 else 0.0

        if snr < min_snr:
            break

        n_transits = _count_transits(
            t_start=time_bjd[0],
            t_end=time_bjd[-1],
            period_days=period_days,
            epoch_bjd=epoch_bjd,
            duration_days=duration_hours / 24.0,
        )

        signals.append(
            CandidateSignal(
                candidate_id=_make_candidate_id(target_id, peak_idx + 1),
                mission=mission,
                target_id=target_id,
                period_days=period_days,
                epoch_bjd=epoch_bjd,
                duration_hours=duration_hours,
                depth_ppm=depth * 1_000_000.0,
                transit_count=n_transits,
                snr=snr,
            )
        )

        # Mask this signal's transit windows before the next BLS run
        half_win = duration_hours / 24.0 * 0.75
        phase = (time_bjd - epoch_bjd) % period_days
        in_transit = (phase < half_win) | (phase > period_days - half_win)
        include &= ~in_transit

    return signals


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_flux_err(lc: Any, flux: np.ndarray) -> np.ndarray:
    """Return per-cadence flux errors; falls back to MAD if unavailable."""
    try:
        err = np.asarray(lc.flux_err.value, dtype=float)
        if err.shape == flux.shape and np.all(np.isfinite(err)) and np.all(err > 0.0):
            return err
    except (AttributeError, TypeError, ValueError):
        pass
    mad = float(np.median(np.abs(flux - np.median(flux))))
    sigma = 1.4826 * mad if mad > 0.0 else 1e-4
    return np.full_like(flux, sigma)


def _count_transits(
    t_start: float,
    t_end: float,
    *,
    period_days: float,
    epoch_bjd: float,
    duration_days: float,
) -> int:
    """Count transit windows that fall within [t_start, t_end]."""
    if t_end <= t_start or period_days <= 0.0:
        return 0
    half_dur = duration_days / 2.0
    n_first = math.ceil((t_start - epoch_bjd - half_dur) / period_days)
    n_last = math.floor((t_end - epoch_bjd + half_dur) / period_days)
    return max(1, n_last - n_first + 1)


def _make_candidate_id(target_id: str, peak_number: int) -> str:
    return f"{target_id.replace(' ', '_')}_s{peak_number:02d}"
