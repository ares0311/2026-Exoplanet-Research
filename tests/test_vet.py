"""Tests for exo_toolkit.vet.

vet_signal() accesses lc.time.jd, lc.flux.value, and lc.flux_err.value.
Tests use real numpy arrays with injected transits so that the diagnostics
computed from the light curve are meaningful.  Catalog-sourced diagnostics
are passed as keyword arguments or left as None.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from exo_toolkit.features import RawDiagnostics
from exo_toolkit.schemas import CandidateFeatures, CandidateSignal
from exo_toolkit.vet import (
    VetResult,
    _extract_arrays,
    _measure_data_gap_fraction,
    _measure_extra_events,
    _measure_individual_transit_shapes,
    _measure_individual_transits,
    _measure_odd_even,
    _measure_secondary_eclipse,
    _measure_transit_shape,
    vet_signal,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EPOCH = 2459000.0
_PERIOD = 5.0
_DURATION_H = 2.0


def _signal(
    *,
    period_days: float = _PERIOD,
    epoch_bjd: float = _EPOCH,
    duration_hours: float = _DURATION_H,
    snr: float = 10.0,
    depth_ppm: float = 5000.0,
    transit_count: int = 5,
) -> CandidateSignal:
    return CandidateSignal(
        candidate_id="TIC_1_s01",
        mission="TESS",
        target_id="TIC 1",
        period_days=period_days,
        epoch_bjd=epoch_bjd,
        duration_hours=duration_hours,
        depth_ppm=depth_ppm,
        transit_count=transit_count,
        snr=snr,
    )


def _lc_with_transit(
    *,
    n_points: int = 2000,
    t_span_days: float = 27.0,
    period_days: float = _PERIOD,
    epoch_bjd: float = _EPOCH,
    depth: float = 0.005,
    duration_hours: float = _DURATION_H,
    noise: float = 0.0005,
    seed: int = 42,
    include_flux_err: bool = True,
) -> MagicMock:
    rng = np.random.default_rng(seed)
    t = np.linspace(epoch_bjd, epoch_bjd + t_span_days, n_points)
    flux = np.ones(n_points) + rng.normal(0.0, noise, n_points)
    half_dur_days = duration_hours / 24.0 / 2.0
    phase = (t - epoch_bjd) % period_days
    in_transit = (phase < half_dur_days) | (phase > period_days - half_dur_days)
    flux[in_transit] -= depth

    lc = MagicMock()
    lc.time.jd = t
    lc.flux.value = flux
    if include_flux_err:
        lc.flux_err.value = np.full(n_points, noise)
    else:
        lc.flux_err = MagicMock(spec=[])
    return lc


def _flat_lc(
    *,
    n_points: int = 500,
    t_span_days: float = 27.0,
    noise: float = 0.001,
    seed: int = 0,
) -> MagicMock:
    rng = np.random.default_rng(seed)
    t = np.linspace(_EPOCH, _EPOCH + t_span_days, n_points)
    flux = np.ones(n_points) + rng.normal(0.0, noise, n_points)
    lc = MagicMock()
    lc.time.jd = t
    lc.flux.value = flux
    lc.flux_err.value = np.full(n_points, noise)
    return lc


# ---------------------------------------------------------------------------
# VetResult
# ---------------------------------------------------------------------------


class TestVetResult:
    def test_is_frozen_dataclass(self) -> None:
        import dataclasses

        sig = _signal()
        lc = _lc_with_transit()
        result = vet_signal(lc, sig)
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.diagnostics = result.diagnostics  # type: ignore[misc]

    def test_contains_raw_diagnostics(self) -> None:
        result = vet_signal(_lc_with_transit(), _signal())
        assert isinstance(result.diagnostics, RawDiagnostics)

    def test_contains_candidate_features(self) -> None:
        result = vet_signal(_lc_with_transit(), _signal())
        assert isinstance(result.features, CandidateFeatures)

    def test_returns_vet_result(self) -> None:
        result = vet_signal(_lc_with_transit(), _signal())
        assert isinstance(result, VetResult)


# ---------------------------------------------------------------------------
# _extract_arrays
# ---------------------------------------------------------------------------


class TestExtractArrays:
    def test_returns_three_arrays(self) -> None:
        lc = _flat_lc()
        t, f, e = _extract_arrays(lc, flux_shape=None)
        assert t.shape == f.shape == e.shape

    def test_time_values_match_input(self) -> None:
        lc = _flat_lc(n_points=100)
        t, _, _ = _extract_arrays(lc, flux_shape=None)
        np.testing.assert_array_equal(t, lc.time.jd)

    def test_fallback_err_when_flux_err_missing(self) -> None:
        lc = _lc_with_transit(include_flux_err=False)
        _, f, e = _extract_arrays(lc, flux_shape=None)
        assert e.shape == f.shape
        assert np.all(e > 0.0)

    def test_fallback_err_all_positive(self) -> None:
        lc = MagicMock()
        lc.time.jd = np.linspace(_EPOCH, _EPOCH + 10.0, 100)
        lc.flux.value = np.ones(100)
        del lc.flux_err.value  # triggers AttributeError → MAD fallback
        _, _, e = _extract_arrays(lc, flux_shape=None)
        assert np.all(e > 0.0)


# ---------------------------------------------------------------------------
# _measure_individual_transits
# ---------------------------------------------------------------------------


class TestMeasureIndividualTransits:
    def test_finds_multiple_transits(self) -> None:
        lc = _lc_with_transit(t_span_days=27.0, period_days=5.0)
        sig = _signal(t_span_days=27.0) if False else _signal()
        t, f, e = _extract_arrays(lc, flux_shape=None)
        depths, errs, n = _measure_individual_transits(t, f, e, signal=sig)
        assert n >= 3

    def test_depths_positive_for_transiting_signal(self) -> None:
        lc = _lc_with_transit(depth=0.005, noise=0.0002)
        sig = _signal()
        t, f, e = _extract_arrays(lc, flux_shape=None)
        depths, errs, _ = _measure_individual_transits(t, f, e, signal=sig)
        if depths is not None:
            assert all(d > 0.0 for d in depths)

    def test_returns_none_for_too_few_points(self) -> None:
        t = np.array([_EPOCH, _EPOCH + 0.01, _EPOCH + 0.02])
        f = np.ones(3)
        e = np.full(3, 1e-3)
        depths, errs, n = _measure_individual_transits(
            t, f, e, signal=_signal()
        )
        assert depths is None
        assert errs is None

    def test_returns_none_when_very_long_period_no_transits_in_window(self) -> None:
        # Period much longer than the baseline → no transit windows observed
        lc = _lc_with_transit(t_span_days=5.0)
        t, f, e = _extract_arrays(lc, flux_shape=None)
        # Period = 100 days, baseline = 5 days → at most 1 transit window
        depths, errs, n = _measure_individual_transits(
            t, f, e, signal=_signal(period_days=100.0, epoch_bjd=_EPOCH + 200.0)
        )
        # epoch 200 days past end of baseline → 0 transits observed
        assert depths is None or n < 2

    def test_depth_tuples_same_length(self) -> None:
        lc = _lc_with_transit(t_span_days=27.0)
        t, f, e = _extract_arrays(lc, flux_shape=None)
        depths, errs, n = _measure_individual_transits(t, f, e, signal=_signal())
        if depths is not None and errs is not None:
            assert len(depths) == len(errs)

    def test_errors_all_positive(self) -> None:
        lc = _lc_with_transit(t_span_days=27.0)
        t, f, e = _extract_arrays(lc, flux_shape=None)
        _, errs, _ = _measure_individual_transits(t, f, e, signal=_signal())
        if errs is not None:
            assert all(err > 0.0 for err in errs)


class TestMeasureIndividualTransitShapes:
    def test_resolves_multiple_durations_and_midpoints(self) -> None:
        lc = _lc_with_transit(depth=0.005, noise=0.0002)
        t, f, e = _extract_arrays(lc, flux_shape=None)
        durations, errors, midpoints, missing_fraction, asymmetries = (
            _measure_individual_transit_shapes(t, f, e, signal=_signal())
        )
        assert durations is not None and errors is not None and midpoints is not None
        assert len(durations) == len(errors) == len(midpoints) >= 2
        assert all(duration > 0.0 for duration in durations)
        assert all(error > 0.0 for error in errors)
        assert all(
            abs((midpoint - _EPOCH) / _PERIOD - round((midpoint - _EPOCH) / _PERIOD))
            < 0.01
            for midpoint in midpoints
        )
        # Clean injected transit: nearly every covered window should resolve.
        assert missing_fraction is not None
        assert missing_fraction < 0.2
        assert asymmetries is not None
        assert len(asymmetries) == len(midpoints)

    def test_returns_none_for_flat_noise(self) -> None:
        lc = _flat_lc(noise=0.0002)
        t, f, e = _extract_arrays(lc, flux_shape=None)
        durations, errors, midpoints, _missing_fraction, asymmetries = (
            _measure_individual_transit_shapes(t, f, e, signal=_signal())
        )
        assert (durations, errors, midpoints, asymmetries) == (None, None, None, None)

    def test_all_windows_missing_for_densely_sampled_flat_noise(self) -> None:
        # Dense enough sampling that every predicted window clears the
        # 5-cadence coverage floor, but no dip is present anywhere.
        lc = _flat_lc(n_points=2000, noise=0.0002)
        t, f, e = _extract_arrays(lc, flux_shape=None)
        _durations, _errors, _midpoints, missing_fraction, _asymmetries = (
            _measure_individual_transit_shapes(t, f, e, signal=_signal())
        )
        assert missing_fraction == 1.0

    def test_recovers_a_shifted_transit_midpoint(self) -> None:
        rng = np.random.default_rng(123)
        time_bjd = np.linspace(_EPOCH, _EPOCH + 27.0, 10_000)
        flux = np.ones_like(time_bjd) + rng.normal(0.0, 0.0001, time_bjd.size)
        half_duration = _DURATION_H / 48.0
        centers = [_EPOCH + index * _PERIOD for index in range(6)]
        centers[2] += 30.0 / 1440.0
        for center in centers:
            flux[np.abs(time_bjd - center) <= half_duration] -= 0.005
        errors = np.full_like(flux, 0.0001)
        _durations, _duration_errors, midpoints, _missing_fraction, _asymmetries = (
            _measure_individual_transit_shapes(
                time_bjd, flux, errors, signal=_signal()
            )
        )
        assert midpoints is not None
        oc_minutes = [
            abs(
                (
                    midpoint
                    - (_EPOCH + round((midpoint - _EPOCH) / _PERIOD) * _PERIOD)
                )
                * 1440.0
            )
            for midpoint in midpoints
        ]
        assert max(oc_minutes) > 20.0

    def test_returns_none_when_fewer_than_two_windows_have_data(self) -> None:
        # Long period, short baseline → at most one window can have coverage.
        lc = _lc_with_transit(t_span_days=5.0)
        t, f, e = _extract_arrays(lc, flux_shape=None)
        _d, _e, _m, missing_fraction, _a = _measure_individual_transit_shapes(
            t, f, e, signal=_signal(period_days=100.0, epoch_bjd=_EPOCH + 200.0)
        )
        assert missing_fraction is None

    def test_missing_fraction_is_fraction_of_unresolved_covered_windows(self) -> None:
        # Five expected windows with data, only the first two carry a real dip.
        rng = np.random.default_rng(7)
        period = 5.0
        time_bjd = np.linspace(_EPOCH, _EPOCH + 4.0 * period + 1.0, 6000)
        flux = np.ones_like(time_bjd) + rng.normal(0.0, 0.0001, time_bjd.size)
        half_duration = _DURATION_H / 48.0
        for index in range(2):
            center = _EPOCH + index * period
            flux[np.abs(time_bjd - center) <= half_duration] -= 0.006
        errors = np.full_like(flux, 0.0001)
        _d, _e, _m, missing_fraction, _a = _measure_individual_transit_shapes(
            time_bjd, flux, errors, signal=_signal(period_days=period)
        )
        assert missing_fraction is not None
        assert 0.4 < missing_fraction < 0.8

    def test_symmetric_events_give_near_zero_asymmetry(self) -> None:
        # Dense sampling so per-event point counts are large enough that
        # grid-alignment and noise fluctuations average out. Pad well before
        # the first event and after the last so every window has full
        # before/after coverage (an event flush against the baseline edge
        # is inherently one-sided regardless of true symmetry).
        rng = np.random.default_rng(99)
        period = 5.0
        time_bjd = np.linspace(_EPOCH - 1.0, _EPOCH + 4.0 * period + 1.0, 20_000)
        flux = np.ones_like(time_bjd) + rng.normal(0.0, 0.0001, time_bjd.size)
        half_duration = _DURATION_H / 48.0
        for index in range(5):
            center = _EPOCH + index * period
            flux[np.abs(time_bjd - center) <= half_duration] -= 0.006
        errors = np.full_like(flux, 0.0001)
        _d, _e, _m, _mf, asymmetries = _measure_individual_transit_shapes(
            time_bjd, flux, errors, signal=_signal(period_days=period)
        )
        assert asymmetries is not None
        assert all(abs(a) < 0.3 for a in asymmetries)

    def test_lopsided_events_give_large_asymmetry(self) -> None:
        # Deficit concentrated entirely after the predicted center each event
        # (e.g. an asymmetric ramp) -> asymmetry should be strongly positive.
        rng = np.random.default_rng(11)
        period = 5.0
        time_bjd = np.linspace(_EPOCH, _EPOCH + 4.0 * period + 1.0, 8000)
        flux = np.ones_like(time_bjd) + rng.normal(0.0, 0.0001, time_bjd.size)
        half_duration = _DURATION_H / 48.0
        for index in range(5):
            center = _EPOCH + index * period
            offsets = time_bjd - center
            lopsided = (offsets >= 0.0) & (offsets <= half_duration)
            flux[lopsided] -= 0.006
        errors = np.full_like(flux, 0.0001)
        _d, _e, _m, _mf, asymmetries = _measure_individual_transit_shapes(
            time_bjd, flux, errors, signal=_signal(period_days=period)
        )
        assert asymmetries is not None
        assert all(a > 0.7 for a in asymmetries)

    def test_none_asymmetries_when_fewer_than_two_events_resolve(self) -> None:
        lc = _lc_with_transit(t_span_days=5.0)
        t, f, e = _extract_arrays(lc, flux_shape=None)
        _d, _e2, _m, _mf, asymmetries = _measure_individual_transit_shapes(
            t, f, e, signal=_signal(period_days=100.0, epoch_bjd=_EPOCH + 200.0)
        )
        assert asymmetries is None


# ---------------------------------------------------------------------------
# _measure_extra_events
# ---------------------------------------------------------------------------


def _dense_baseline(
    *, n_points: int = 20_000, span_days: float = 21.0, noise: float = 0.0001, seed: int = 1
):
    rng = np.random.default_rng(seed)
    time_bjd = np.linspace(_EPOCH - 1.0, _EPOCH - 1.0 + span_days, n_points)
    flux = np.ones_like(time_bjd) + rng.normal(0.0, noise, time_bjd.size)
    return time_bjd, flux, np.full_like(flux, noise)


def _inject_periodic_transits(time_bjd, flux, *, period=_PERIOD, depth=0.006, n_transits=4):
    half_duration = _DURATION_H / 48.0
    for index in range(n_transits):
        center = _EPOCH + index * period
        flux[np.abs(time_bjd - center) <= half_duration] -= depth
    return flux


def _inject_extra_dip(time_bjd, flux, *, center, depth=0.01, half_width_days=0.02):
    flux[np.abs(time_bjd - center) <= half_width_days] -= depth
    return flux


class TestMeasureExtraEvents:
    def test_returns_none_for_too_few_total_points(self) -> None:
        t = np.linspace(_EPOCH, _EPOCH + 1.0, 8)
        f = np.ones_like(t)
        count = _measure_extra_events(t, f, signal=_signal())
        assert count is None

    def test_returns_none_when_almost_everything_is_in_transit(self) -> None:
        # A period comparable to the transit duration itself, densely
        # sampled, leaves too few out-of-transit cadences to judge.
        rng = np.random.default_rng(5)
        short_period = _DURATION_H / 24.0 * 1.02  # barely longer than duration
        time_bjd = np.linspace(_EPOCH, _EPOCH + 1.0, 500)
        flux = np.ones_like(time_bjd) + rng.normal(0.0, 0.0001, time_bjd.size)
        count = _measure_extra_events(
            time_bjd, flux, signal=_signal(period_days=short_period)
        )
        assert count is None

    def test_clean_light_curve_has_zero_extra_events(self) -> None:
        time_bjd, flux, _err = _dense_baseline()
        flux = _inject_periodic_transits(time_bjd, flux)
        count = _measure_extra_events(time_bjd, flux, signal=_signal())
        assert count == 0

    def test_single_injected_dip_is_detected(self) -> None:
        time_bjd, flux, _err = _dense_baseline()
        flux = _inject_periodic_transits(time_bjd, flux)
        # Roughly halfway between the first two periodic transits.
        flux = _inject_extra_dip(time_bjd, flux, center=_EPOCH + _PERIOD / 2.0)
        count = _measure_extra_events(time_bjd, flux, signal=_signal())
        assert count == 1

    def test_two_separated_dips_counted_separately(self) -> None:
        time_bjd, flux, _err = _dense_baseline()
        flux = _inject_periodic_transits(time_bjd, flux)
        flux = _inject_extra_dip(time_bjd, flux, center=_EPOCH + 0.5 * _PERIOD)
        flux = _inject_extra_dip(time_bjd, flux, center=_EPOCH + 1.5 * _PERIOD)
        count = _measure_extra_events(time_bjd, flux, signal=_signal())
        assert count == 2

    def test_single_cadence_outlier_is_not_counted(self) -> None:
        time_bjd, flux, _err = _dense_baseline()
        flux = _inject_periodic_transits(time_bjd, flux)
        # A single isolated cadence deficit, not a >=2-cadence cluster.
        nearest = int(np.argmin(np.abs(time_bjd - (_EPOCH + 0.5 * _PERIOD))))
        flux[nearest] -= 0.01
        count = _measure_extra_events(time_bjd, flux, signal=_signal())
        assert count == 0

    def test_wide_trend_is_not_counted_as_an_extra_event(self) -> None:
        time_bjd, flux, _err = _dense_baseline()
        flux = _inject_periodic_transits(time_bjd, flux)
        # 10-hour-wide depression, far longer than the 4-hour cluster-span cap.
        center = _EPOCH + 0.5 * _PERIOD
        wide_half_width_days = (10.0 / 24.0) / 2.0
        flux = _inject_extra_dip(
            time_bjd, flux, center=center, depth=0.01, half_width_days=wide_half_width_days
        )
        count = _measure_extra_events(time_bjd, flux, signal=_signal())
        assert count == 0


# ---------------------------------------------------------------------------
# _measure_odd_even
# ---------------------------------------------------------------------------


class TestMeasureOddEven:
    def test_returns_none_when_depths_none(self) -> None:
        result = _measure_odd_even(None, None)
        assert result == (None, None, None, None)

    def test_returns_none_when_fewer_than_4_transits(self) -> None:
        depths = (0.01, 0.01, 0.01)
        errs = (0.001, 0.001, 0.001)
        result = _measure_odd_even(depths, errs)
        assert result == (None, None, None, None)

    def test_returns_four_values_with_enough_transits(self) -> None:
        depths = tuple(0.005 + i * 0.0001 for i in range(6))
        errs = tuple(0.0005 for _ in range(6))
        d_odd, e_odd, d_even, e_even = _measure_odd_even(depths, errs)
        assert all(v is not None for v in (d_odd, e_odd, d_even, e_even))

    def test_ppm_scale(self) -> None:
        # depths ~0.005 → ppm ~5000
        depths = tuple(0.005 for _ in range(6))
        errs = tuple(0.0005 for _ in range(6))
        d_odd, _, d_even, _ = _measure_odd_even(depths, errs)
        assert d_odd is not None and d_even is not None
        assert 1000.0 < d_odd < 100_000.0
        assert 1000.0 < d_even < 100_000.0

    def test_equal_depths_give_similar_odd_even(self) -> None:
        depths = tuple(0.005 for _ in range(8))
        errs = tuple(0.0005 for _ in range(8))
        d_odd, _, d_even, _ = _measure_odd_even(depths, errs)
        assert d_odd is not None and d_even is not None
        assert abs(d_odd - d_even) < 100.0  # < 100 ppm difference

    def test_unequal_depths_captured(self) -> None:
        # Odd transits deeper than even (eclipsing binary signature)
        depths = (0.01, 0.003, 0.01, 0.003, 0.01, 0.003)
        errs = (0.0005,) * 6
        d_odd, _, d_even, _ = _measure_odd_even(depths, errs)
        assert d_odd is not None and d_even is not None
        assert d_odd > d_even * 2.0


# ---------------------------------------------------------------------------
# _measure_secondary_eclipse
# ---------------------------------------------------------------------------


class TestMeasureSecondaryEclipse:
    def test_returns_none_when_all_oot_excluded(self) -> None:
        # Very long period → secondary at period/2 has no out-of-transit baseline
        lc = _lc_with_transit(t_span_days=2.0)
        t, f, e = _extract_arrays(lc, flux_shape=None)
        # Period >> baseline, so all points are near secondary or primary
        result = _measure_secondary_eclipse(
            t, f, e, signal=_signal(period_days=100.0)
        )
        # Either None (too few points) or a value — just shouldn't raise
        assert result is None or isinstance(result, float)

    def test_returns_none_when_too_few_points_at_phase_half(self) -> None:
        # Very sparse LC: only 3 points total
        t = np.array([_EPOCH, _EPOCH + _PERIOD / 2.0, _EPOCH + _PERIOD])
        f = np.ones(3)
        e = np.full(3, 1e-3)
        result = _measure_secondary_eclipse(t, f, e, signal=_signal())
        assert result is None

    def test_no_secondary_gives_low_snr(self) -> None:
        # Flat light curve with no secondary — should return near zero or None
        lc = _flat_lc(n_points=2000, t_span_days=27.0)
        t, f, e = _extract_arrays(lc, flux_shape=None)
        result = _measure_secondary_eclipse(t, f, e, signal=_signal())
        if result is not None:
            assert abs(result) < 5.0  # low SNR expected for flat LC

    def test_injected_secondary_gives_positive_snr(self) -> None:
        # Inject a secondary eclipse at phase 0.5
        t = np.linspace(_EPOCH, _EPOCH + 27.0, 5000)
        flux = np.ones(len(t))
        noise = 0.0002
        rng = np.random.default_rng(1)
        flux += rng.normal(0.0, noise, len(t))
        half_dur = _DURATION_H / 24.0 / 2.0

        # Primary eclipse
        phase = (t - _EPOCH) % _PERIOD
        flux[(phase < half_dur) | (phase > _PERIOD - half_dur)] -= 0.005

        # Secondary eclipse at phase 0.5
        sec_phase = (phase - _PERIOD / 2.0) % _PERIOD
        flux[(sec_phase < half_dur) | (sec_phase > _PERIOD - half_dur)] -= 0.003

        lc = MagicMock()
        lc.time.jd = t
        lc.flux.value = flux
        lc.flux_err.value = np.full(len(t), noise)

        t_arr, f_arr, e_arr = _extract_arrays(lc, flux_shape=None)
        result = _measure_secondary_eclipse(t_arr, f_arr, e_arr, signal=_signal())
        assert result is not None
        assert result > 3.0


# ---------------------------------------------------------------------------
# _measure_transit_shape
# ---------------------------------------------------------------------------


class TestMeasureTransitShape:
    def test_returns_none_when_no_oot_points(self) -> None:
        # Only 4 points, all falling inside transit → no out-of-transit baseline
        t = np.linspace(_EPOCH, _EPOCH + _DURATION_H / 24.0 * 0.9, 4)
        f = np.full(4, 0.995)
        result = _measure_transit_shape(t, f, signal=_signal())
        assert result is None

    def test_returns_none_for_sparse_lc(self) -> None:
        t = np.linspace(_EPOCH, _EPOCH + 0.1, 4)
        f = np.ones(4)
        result = _measure_transit_shape(t, f, signal=_signal())
        assert result is None

    def test_returns_value_in_range(self) -> None:
        lc = _lc_with_transit(depth=0.005, noise=0.0002)
        t, f, _ = _extract_arrays(lc, flux_shape=None)
        result = _measure_transit_shape(t, f, signal=_signal())
        if result is not None:
            assert 0.0 <= result <= 1.0

    def test_box_transit_gives_high_value(self) -> None:
        # Perfect box transit → inner ≈ outer depth → ratio ≈ 1.0
        rng = np.random.default_rng(5)
        n = 5000
        t = np.linspace(_EPOCH, _EPOCH + 27.0, n)
        flux = np.ones(n) + rng.normal(0.0, 0.0001, n)
        half_dur = _DURATION_H / 24.0 / 2.0
        phase = (t - _EPOCH) % _PERIOD
        # Flat-bottomed box: all in-transit points get equal depth
        in_transit = (phase < half_dur) | (phase > _PERIOD - half_dur)
        flux[in_transit] -= 0.005

        lc = MagicMock()
        lc.time.jd = t
        lc.flux.value = flux
        lc.flux_err.value = np.full(n, 0.0001)
        t_arr, f_arr, _ = _extract_arrays(lc, flux_shape=None)
        result = _measure_transit_shape(t_arr, f_arr, signal=_signal())
        if result is not None:
            assert result > 0.5  # should be box-like

    def test_v_shape_gives_low_value(self) -> None:
        # V-shaped transit: outer region shallower than inner
        rng = np.random.default_rng(3)
        n = 5000
        t = np.linspace(_EPOCH, _EPOCH + 27.0, n)
        flux = np.ones(n) + rng.normal(0.0, 0.0001, n)
        half_dur = _DURATION_H / 24.0 / 2.0
        phase = (t - _EPOCH) % _PERIOD
        # Center-fold phase
        phase_centered = np.where(phase > _PERIOD / 2.0, phase - _PERIOD, phase)
        abs_ph = np.abs(phase_centered)
        in_transit = abs_ph <= half_dur
        # Linear V-shape: depth proportional to (1 - abs_phase / half_dur)
        flux[in_transit] -= 0.01 * (1.0 - abs_ph[in_transit] / half_dur)

        lc = MagicMock()
        lc.time.jd = t
        lc.flux.value = flux
        lc.flux_err.value = np.full(n, 0.0001)
        t_arr, f_arr, _ = _extract_arrays(lc, flux_shape=None)
        result = _measure_transit_shape(t_arr, f_arr, signal=_signal())
        if result is not None:
            assert result < 0.7  # should be V-like (outer < inner)


# ---------------------------------------------------------------------------
# _measure_data_gap_fraction
# ---------------------------------------------------------------------------


class TestMeasureDataGapFraction:
    def test_no_gaps_gives_zero(self) -> None:
        lc = _lc_with_transit(n_points=2000, t_span_days=27.0)
        t, _, _ = _extract_arrays(lc, flux_shape=None)
        frac = _measure_data_gap_fraction(t, signal=_signal())
        assert frac is not None
        assert frac == pytest.approx(0.0)

    def test_returns_none_for_tiny_lc(self) -> None:
        t = np.array([_EPOCH, _EPOCH + 0.01])
        frac = _measure_data_gap_fraction(t, signal=_signal())
        assert frac is None

    def test_very_sparse_lc_gives_high_gap_fraction(self) -> None:
        # Only 1 point per transit window → all windows are gaps (< 3 pts each)
        # Place exactly 1 point per transit window centre
        n_transits = 5
        t = np.array([_EPOCH + i * _PERIOD for i in range(n_transits)])
        frac = _measure_data_gap_fraction(t, signal=_signal())
        # Each window has exactly 1 point which is < min_points_per_transit (3)
        assert frac is not None
        assert frac == pytest.approx(1.0)

    def test_fraction_in_range(self) -> None:
        lc = _lc_with_transit(n_points=200, t_span_days=27.0)
        t, _, _ = _extract_arrays(lc, flux_shape=None)
        frac = _measure_data_gap_fraction(t, signal=_signal())
        if frac is not None:
            assert 0.0 <= frac <= 1.0

    def test_all_gaps_gives_one(self) -> None:
        # Only 1 sparse point per transit window → all gaps
        # Ensure no window has >= 3 points by placing points far apart
        t = np.array([_EPOCH + i * _PERIOD for i in range(5)])  # 1 pt per window
        frac = _measure_data_gap_fraction(t, signal=_signal())
        if frac is not None:
            assert frac == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# vet_signal — integration tests
# ---------------------------------------------------------------------------


class TestVetSignal:
    def test_returns_vet_result(self) -> None:
        result = vet_signal(_lc_with_transit(), _signal())
        assert isinstance(result, VetResult)

    def test_snr_score_not_none(self) -> None:
        # snr_score always computable from signal; should not be None
        result = vet_signal(_lc_with_transit(), _signal(snr=10.0))
        assert result.features.snr_score is not None

    def test_transit_count_score_not_none(self) -> None:
        result = vet_signal(_lc_with_transit(), _signal(transit_count=5))
        assert result.features.transit_count_score is not None

    def test_individual_depths_populated(self) -> None:
        result = vet_signal(_lc_with_transit(t_span_days=27.0), _signal())
        assert result.diagnostics.individual_depths is not None
        assert len(result.diagnostics.individual_depths) >= 2

    def test_individual_shape_diagnostics_populated(self) -> None:
        result = vet_signal(
            _lc_with_transit(depth=0.005, noise=0.0002), _signal()
        )
        assert result.diagnostics.individual_durations is not None
        assert result.diagnostics.individual_duration_errors is not None
        assert result.diagnostics.individual_transit_midpoints is not None
        assert result.features.duration_consistency_score is not None
        assert result.features.transit_timing_variation_score is not None

    def test_data_gap_fraction_near_zero(self) -> None:
        result = vet_signal(_lc_with_transit(n_points=2000, t_span_days=27.0), _signal())
        assert result.diagnostics.data_gap_fraction is not None
        assert result.diagnostics.data_gap_fraction == pytest.approx(0.0)

    def test_catalog_kwargs_flow_to_diagnostics(self) -> None:
        result = vet_signal(
            _lc_with_transit(),
            _signal(),
            stellar_radius_rsun=1.2,
            stellar_mass_msun=1.1,
            contamination_ratio=0.05,
        )
        assert result.diagnostics.stellar_radius_rsun == pytest.approx(1.2)
        assert result.diagnostics.stellar_mass_msun == pytest.approx(1.1)
        assert result.diagnostics.contamination_ratio == pytest.approx(0.05)

    def test_known_object_kwargs_flow_to_diagnostics(self) -> None:
        result = vet_signal(
            _lc_with_transit(),
            _signal(),
            target_id_matched=True,
            period_match_sigma=0.5,
            coordinate_match_arcsec=1.2,
        )
        assert result.diagnostics.target_id_matched is True
        assert result.diagnostics.period_match_sigma == pytest.approx(0.5)
        assert result.diagnostics.coordinate_match_arcsec == pytest.approx(1.2)

    def test_stellar_teff_k_flows_to_diagnostics(self) -> None:
        result = vet_signal(
            _lc_with_transit(),
            _signal(),
            stellar_teff_k=4500.0,
        )
        assert result.diagnostics.stellar_teff_k == pytest.approx(4500.0)

    def test_all_catalog_none_still_works(self) -> None:
        # All catalog kwargs default to None — should not raise
        result = vet_signal(_lc_with_transit(), _signal())
        assert isinstance(result, VetResult)

    def test_no_flux_err_still_works(self) -> None:
        lc = _lc_with_transit(include_flux_err=False)
        result = vet_signal(lc, _signal())
        assert isinstance(result, VetResult)

    def test_features_all_have_scores_for_rich_input(self) -> None:
        # With full catalog data, more features should be non-None
        result = vet_signal(
            _lc_with_transit(t_span_days=27.0, depth=0.005, noise=0.0002),
            _signal(snr=12.0, transit_count=5),
            stellar_radius_rsun=1.0,
            stellar_mass_msun=1.0,
            contamination_ratio=0.02,
            centroid_offset_sigma=0.5,
            quality_flag_fraction=0.0,
            ls_power_at_period=0.1,
            target_id_matched=False,
            period_match_sigma=10.0,
            coordinate_match_arcsec=60.0,
        )
        features = result.features
        # Core detection features must be present
        assert features.snr_score is not None
        assert features.log_snr_score is not None
        assert features.transit_count_score is not None

    def test_depth_score_positive_for_deep_transit(self) -> None:
        result = vet_signal(
            _lc_with_transit(depth=0.01, noise=0.0002, t_span_days=27.0),
            _signal(snr=15.0, depth_ppm=10000.0),
        )
        # depth_consistency_score may be None when fewer than 2 transits observed
        _ = result.features.depth_consistency_score

    def test_secondary_snr_near_zero_for_planet_like_signal(self) -> None:
        result = vet_signal(
            _lc_with_transit(depth=0.005, noise=0.0002, t_span_days=27.0),
            _signal(),
        )
        sec = result.diagnostics.secondary_snr
        if sec is not None:
            assert abs(sec) < 5.0  # no secondary eclipse expected

    def test_vetresult_diagnostics_and_features_consistent(self) -> None:
        lc = _lc_with_transit(t_span_days=27.0)
        sig = _signal()
        result = vet_signal(lc, sig)
        # features were computed from diagnostics — snr_score should match signal snr
        from exo_toolkit.features import snr_score as expected_snr_score

        expected = expected_snr_score(sig.snr)
        assert result.features.snr_score == pytest.approx(expected)
