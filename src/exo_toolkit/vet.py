"""Vetting: compute RawDiagnostics from a cleaned light curve + CandidateSignal.

Only three attributes are accessed on the incoming LightCurve object:
    lc.time.jd        — array of BJD time values
    lc.flux.value     — normalised flux array
    lc.flux_err.value — per-cadence flux uncertainty (optional)

Diagnostics that require external catalog data (stellar parameters, Gaia
crowding, quality flags, etc.) must be supplied as keyword arguments; they
default to None and propagate as "not available" into feature scores.

Public API
----------
vet_signal(lc, signal, *, catalog_kwargs...) → VetResult
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from exo_toolkit.features import RawDiagnostics, extract_features
from exo_toolkit.schemas import CandidateFeatures, CandidateSignal

# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VetResult:
    """Computed diagnostics and extracted feature scores for one candidate signal."""

    diagnostics: RawDiagnostics
    features: CandidateFeatures


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def vet_signal(
    lc: Any,
    signal: CandidateSignal,
    *,
    # Stellar parameters (from TIC/GAIA catalog)
    stellar_radius_rsun: float | None = None,
    stellar_mass_msun: float | None = None,
    # Aperture photometry diagnostics (from pipeline headers)
    contamination_ratio: float | None = None,
    centroid_offset_sigma: float | None = None,
    nearby_bright_source_count: int | None = None,
    nearby_source_magnitude_diff: float | None = None,
    aperture_edge_proximity: float | None = None,
    # Instrumental quality diagnostics
    quality_flag_fraction: float | None = None,
    sector_boundary_fraction: float | None = None,
    background_excursion_sigma: float | None = None,
    nearby_targets_common_signal: float | None = None,
    # Stellar variability diagnostics
    ls_power_at_period: float | None = None,
    ls_power_at_harmonics: float | None = None,
    flare_rate_per_day: float | None = None,
    quasi_periodic_strength: float | None = None,
    # Known-object catalog matching
    target_id_matched: bool | None = None,
    period_match_sigma: float | None = None,
    epoch_match_sigma: float | None = None,
    coordinate_match_arcsec: float | None = None,
) -> VetResult:
    """Vet a candidate transit signal against its cleaned light curve.

    Computes what diagnostics can be derived from the light curve arrays alone
    (individual transit depths, odd/even comparison, secondary eclipse search,
    transit shape, data-gap fraction).  All other diagnostics require catalog
    inputs supplied as keyword arguments.

    Args:
        lc: A lightkurve.LightCurve object (from clean_lightcurve()).
            Must expose lc.time.jd, lc.flux.value, and optionally
            lc.flux_err.value.
        signal: The candidate signal to vet (from search_lightcurve()).
        stellar_radius_rsun: Host-star radius in solar radii (from TIC/Gaia).
        stellar_mass_msun: Host-star mass in solar masses.
        contamination_ratio: Flux contamination fraction from nearby sources.
        centroid_offset_sigma: Significance of in-transit centroid shift.
        nearby_bright_source_count: Number of Gaia sources inside aperture.
        nearby_source_magnitude_diff: Delta-mag of nearest contaminating source.
        aperture_edge_proximity: Proximity of target to aperture edge [0, 1].
        quality_flag_fraction: Fraction of transits with pipeline quality flags.
        sector_boundary_fraction: Fraction of events near sector boundaries.
        background_excursion_sigma: Peak background during transit (in sigma).
        nearby_targets_common_signal: Cross-target flux correlation coefficient.
        ls_power_at_period: Lomb-Scargle power at the transit period.
        ls_power_at_harmonics: LS power at harmonic/sub-harmonic periods.
        flare_rate_per_day: Stellar flare rate.
        quasi_periodic_strength: Autocorrelation amplitude of stellar variability.
        target_id_matched: Whether target matches a known catalog entry.
        period_match_sigma: Sigma deviation from a catalog period.
        epoch_match_sigma: Sigma deviation from a catalog epoch.
        coordinate_match_arcsec: Angular distance to nearest catalog match.

    Returns:
        VetResult containing the RawDiagnostics and CandidateFeatures.
    """
    time_bjd, flux, flux_err = _extract_arrays(lc, flux_shape=None)

    depths, depth_errs, n_transits_observed = _measure_individual_transits(
        time_bjd, flux, flux_err, signal=signal
    )

    depth_odd, err_odd, depth_even, err_even = _measure_odd_even(
        depths, depth_errs
    )

    secondary_snr = _measure_secondary_eclipse(time_bjd, flux, flux_err, signal=signal)
    ingress_egress_frac = _measure_transit_shape(time_bjd, flux, signal=signal)
    data_gap_frac = _measure_data_gap_fraction(time_bjd, signal=signal)

    diagnostics = RawDiagnostics(
        individual_depths=depths,
        individual_depth_errors=depth_errs,
        individual_durations=None,       # per-transit duration fitting not in v0
        individual_duration_errors=None,
        depth_odd_ppm=depth_odd,
        err_odd_ppm=err_odd,
        depth_even_ppm=depth_even,
        err_even_ppm=err_even,
        secondary_snr=secondary_snr,
        ingress_egress_fraction=ingress_egress_frac,
        stellar_radius_rsun=stellar_radius_rsun,
        stellar_mass_msun=stellar_mass_msun,
        contamination_ratio=contamination_ratio,
        centroid_offset_sigma=centroid_offset_sigma,
        nearby_bright_source_count=nearby_bright_source_count,
        nearby_source_magnitude_diff=nearby_source_magnitude_diff,
        aperture_edge_proximity=aperture_edge_proximity,
        quality_flag_fraction=quality_flag_fraction,
        sector_boundary_fraction=sector_boundary_fraction,
        background_excursion_sigma=background_excursion_sigma,
        data_gap_fraction=data_gap_frac,
        nearby_targets_common_signal=nearby_targets_common_signal,
        ls_power_at_period=ls_power_at_period,
        ls_power_at_harmonics=ls_power_at_harmonics,
        flare_rate_per_day=flare_rate_per_day,
        quasi_periodic_strength=quasi_periodic_strength,
        target_id_matched=target_id_matched,
        period_match_sigma=period_match_sigma,
        epoch_match_sigma=epoch_match_sigma,
        coordinate_match_arcsec=coordinate_match_arcsec,
    )

    features = extract_features(signal, diagnostics)
    return VetResult(diagnostics=diagnostics, features=features)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_arrays(
    lc: Any,
    *,
    flux_shape: tuple[int, ...] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract (time_bjd, flux, flux_err) numpy arrays from a LightCurve object."""
    time_bjd = np.asarray(lc.time.jd, dtype=float)
    flux = np.asarray(lc.flux.value, dtype=float)
    try:
        err = np.asarray(lc.flux_err.value, dtype=float)
        if err.shape != flux.shape or not np.all(np.isfinite(err)) or not np.all(err > 0.0):
            raise ValueError
    except (AttributeError, TypeError, ValueError):
        mad = float(np.median(np.abs(flux - np.median(flux))))
        sigma = 1.4826 * mad if mad > 0.0 else 1e-4
        err = np.full_like(flux, sigma)
    return time_bjd, flux, err


def _measure_individual_transits(
    time_bjd: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    *,
    signal: CandidateSignal,
) -> tuple[
    tuple[float, ...] | None,
    tuple[float, ...] | None,
    int,
]:
    """Measure depth and depth uncertainty for each observed transit window.

    Returns (individual_depths, individual_depth_errors, n_observed).
    Depths are fractional (same units as flux, normalized around 1.0).
    Both tuples are None when fewer than two transits have sufficient coverage.
    """
    period = signal.period_days
    epoch = signal.epoch_bjd
    half_dur = signal.duration_hours / 24.0 / 2.0

    if period <= 0.0 or half_dur <= 0.0 or len(time_bjd) < 3:
        return None, None, 0

    # Use overall median as baseline (flux is normalized to ~1.0)
    baseline = float(np.median(flux))

    # Enumerate transit numbers that overlap the observation window
    t_start = float(time_bjd[0])
    t_end = float(time_bjd[-1])
    n_first = int(np.floor((t_start - epoch) / period))
    n_last = int(np.ceil((t_end - epoch) / period))

    depths: list[float] = []
    errs: list[float] = []

    for n in range(n_first - 1, n_last + 2):
        t_center = epoch + n * period
        if t_center < t_start - half_dur or t_center > t_end + half_dur:
            continue
        in_transit = np.abs(time_bjd - t_center) <= half_dur
        n_pts = int(in_transit.sum())
        if n_pts < 3:
            continue
        mean_flux = float(np.mean(flux[in_transit]))
        depth = baseline - mean_flux
        # Propagated uncertainty on mean depth
        depth_err = float(np.sqrt(np.sum(flux_err[in_transit] ** 2))) / n_pts
        depths.append(depth)
        errs.append(max(depth_err, 1e-10))

    if len(depths) < 2:
        return None, None, len(depths)

    return tuple(depths), tuple(errs), len(depths)


def _measure_odd_even(
    individual_depths: tuple[float, ...] | None,
    individual_depth_errors: tuple[float, ...] | None,
) -> tuple[float | None, float | None, float | None, float | None]:
    """Compute weighted-mean depth for odd and even transit groups (in ppm).

    Odd = transits 1, 3, 5, … (1-indexed); even = 2, 4, 6, …
    Returns (depth_odd_ppm, err_odd_ppm, depth_even_ppm, err_even_ppm).
    All four are None when fewer than 2 transits are available per group.
    """
    if individual_depths is None or individual_depth_errors is None:
        return None, None, None, None

    n = len(individual_depths)
    if n < 4:
        return None, None, None, None

    depths = np.array(individual_depths, dtype=float)
    errs = np.array(individual_depth_errors, dtype=float)

    odd_idx = np.arange(0, n, 2)   # 0-based: 0, 2, 4, … → transit 1, 3, 5
    even_idx = np.arange(1, n, 2)  # 0-based: 1, 3, 5, … → transit 2, 4, 6

    def _weighted_mean(d: np.ndarray, e: np.ndarray) -> tuple[float, float]:
        w = 1.0 / e ** 2
        mean = float(np.sum(w * d) / np.sum(w))
        err = float(1.0 / np.sqrt(np.sum(w)))
        return mean, err

    d_odd, e_odd = _weighted_mean(depths[odd_idx], errs[odd_idx])
    d_even, e_even = _weighted_mean(depths[even_idx], errs[even_idx])

    return (
        d_odd * 1_000_000.0,
        e_odd * 1_000_000.0,
        d_even * 1_000_000.0,
        e_even * 1_000_000.0,
    )


def _measure_secondary_eclipse(
    time_bjd: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    *,
    signal: CandidateSignal,
) -> float | None:
    """Estimate SNR of a potential secondary eclipse at phase 0.5.

    Returns None when too few points fall in the secondary window.
    A large positive return value indicates a significant secondary dip.
    """
    period = signal.period_days
    epoch = signal.epoch_bjd
    half_dur = signal.duration_hours / 24.0 / 2.0

    if period <= 0.0 or half_dur <= 0.0:
        return None

    # Phase relative to epoch, folded to [0, period)
    phase = (time_bjd - epoch) % period

    # Secondary eclipse window centred at phase = period/2
    sec_center = period / 2.0
    near_sec = np.abs(phase - sec_center) <= half_dur

    if near_sec.sum() < 3:
        return None

    # Baseline: out-of-transit points (primary + secondary windows both excluded)
    near_primary = (phase <= half_dur) | (phase >= period - half_dur)
    oot = ~near_primary & ~near_sec
    if oot.sum() < 3:
        return None

    baseline = float(np.median(flux[oot]))
    sec_flux = flux[near_sec]
    sec_errs = flux_err[near_sec]

    mean_sec = float(np.mean(sec_flux))
    depth_sec = baseline - mean_sec
    n_pts = int(near_sec.sum())
    depth_err = float(np.sqrt(np.sum(sec_errs ** 2))) / n_pts

    if depth_err <= 0.0:
        return None

    return float(depth_sec / depth_err)


def _measure_transit_shape(
    time_bjd: np.ndarray,
    flux: np.ndarray,
    *,
    signal: CandidateSignal,
) -> float | None:
    """Estimate ingress/egress fraction as a transit-shape proxy.

    Compares the mean depth in the outer half of the transit window to the
    mean depth in the inner half.  Returns a value in [0, 1]:
        ~1.0 → flat-bottomed box (transit is box-shaped)
        ~0.0 → pure V-shape (flux falls linearly to centre)

    Returns None when coverage is insufficient in either region.
    """
    period = signal.period_days
    epoch = signal.epoch_bjd
    half_dur = signal.duration_hours / 24.0 / 2.0

    if period <= 0.0 or half_dur <= 0.0:
        return None

    # Centre-fold phase: distance to nearest transit centre
    phase_raw = (time_bjd - epoch) % period
    # Map to [-period/2, period/2]
    phase = np.where(phase_raw > period / 2.0, phase_raw - period, phase_raw)

    abs_phase = np.abs(phase)
    in_transit = abs_phase <= half_dur
    oot = abs_phase > half_dur

    if in_transit.sum() < 6 or oot.sum() < 3:
        return None

    baseline = float(np.median(flux[oot]))

    inner_mask = abs_phase <= half_dur / 2.0
    outer_mask = (abs_phase > half_dur / 2.0) & (abs_phase <= half_dur)

    n_inner = int(inner_mask.sum())
    n_outer = int(outer_mask.sum())
    if n_inner < 2 or n_outer < 2:
        return None

    inner_depth = baseline - float(np.mean(flux[inner_mask]))
    outer_depth = baseline - float(np.mean(flux[outer_mask]))

    if inner_depth <= 0.0:
        return None

    return float(np.clip(outer_depth / inner_depth, 0.0, 1.0))


def _measure_data_gap_fraction(
    time_bjd: np.ndarray,
    *,
    signal: CandidateSignal,
    min_points_per_transit: int = 3,
) -> float | None:
    """Fraction of expected transit windows that have insufficient data coverage.

    A transit window is considered a "gap" if fewer than min_points_per_transit
    cadences fall within ±half_duration of its centre.

    Returns None when period/duration information is unusable.
    """
    period = signal.period_days
    epoch = signal.epoch_bjd
    half_dur = signal.duration_hours / 24.0 / 2.0

    if period <= 0.0 or half_dur <= 0.0 or len(time_bjd) < 3:
        return None

    t_start = float(time_bjd[0])
    t_end = float(time_bjd[-1])

    n_first = int(np.floor((t_start - epoch) / period))
    n_last = int(np.ceil((t_end - epoch) / period))

    total = 0
    gaps = 0
    for n in range(n_first - 1, n_last + 2):
        t_center = epoch + n * period
        if t_center < t_start - half_dur or t_center > t_end + half_dur:
            continue
        total += 1
        n_pts = int(np.sum(np.abs(time_bjd - t_center) <= half_dur))
        if n_pts < min_points_per_transit:
            gaps += 1

    if total == 0:
        return None

    return float(gaps) / float(total)
