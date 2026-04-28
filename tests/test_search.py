"""Tests for exo_toolkit.search.

search_lightcurve() only accesses three attributes on the LightCurve object:
    lc.time.jd, lc.flux.value, lc.flux_err.value

Tests use real numpy arrays and real astropy BLS computation — no mocking of
BLS internals.  A synthetic transit is injected into white-noise flux so that
the BLS search finds a real periodic signal.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from exo_toolkit.search import (
    _count_transits,
    _extract_flux_err,
    _make_candidate_id,
    search_lightcurve,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _synthetic_lc(
    *,
    n_points: int = 2000,
    t_span_days: float = 27.0,
    period_days: float = 5.0,
    epoch_bjd: float = 2459000.0,
    depth: float = 0.01,
    duration_hours: float = 2.5,
    noise: float = 0.001,
    seed: int = 42,
    include_flux_err: bool = True,
) -> MagicMock:
    """Mock LightCurve with real numpy arrays and an injected box transit."""
    rng = np.random.default_rng(seed)
    t = np.linspace(epoch_bjd, epoch_bjd + t_span_days, n_points)

    flux = np.ones(n_points) + rng.normal(0.0, noise, n_points)
    # Inject transits
    half_dur_days = duration_hours / 24.0 / 2.0
    phase = (t - epoch_bjd) % period_days
    in_transit = (phase < half_dur_days) | (phase > period_days - half_dur_days)
    flux[in_transit] -= depth

    flux_err = np.full(n_points, noise)

    lc = MagicMock()
    lc.time.jd = t
    lc.flux.value = flux
    if include_flux_err:
        lc.flux_err.value = flux_err
    else:
        del lc.flux_err.value  # raise AttributeError on access
    return lc


def _flat_lc(
    *,
    n_points: int = 1000,
    t_span_days: float = 27.0,
    noise: float = 0.001,
    seed: int = 0,
) -> MagicMock:
    """Mock LightCurve with white-noise flux and no transit signal."""
    rng = np.random.default_rng(seed)
    t = np.linspace(2459000.0, 2459000.0 + t_span_days, n_points)
    flux = np.ones(n_points) + rng.normal(0.0, noise, n_points)
    flux_err = np.full(n_points, noise)

    lc = MagicMock()
    lc.time.jd = t
    lc.flux.value = flux
    lc.flux_err.value = flux_err
    return lc


# ---------------------------------------------------------------------------
# TestCountTransits
# ---------------------------------------------------------------------------


class TestCountTransits:
    def test_single_transit(self) -> None:
        n = _count_transits(
            t_start=2459000.0,
            t_end=2459001.0,
            period_days=10.0,
            epoch_bjd=2459000.5,
            duration_days=0.1,
        )
        assert n >= 1

    def test_multiple_transits(self) -> None:
        # 3 transits expected: at epoch, epoch+5, epoch+10
        n = _count_transits(
            t_start=2459000.0,
            t_end=2459011.0,
            period_days=5.0,
            epoch_bjd=2459000.5,
            duration_days=0.1,
        )
        assert n == 3

    def test_returns_int(self) -> None:
        n = _count_transits(
            t_start=2459000.0,
            t_end=2459030.0,
            period_days=5.0,
            epoch_bjd=2459000.0,
            duration_days=0.1,
        )
        assert isinstance(n, int)

    def test_t_end_before_t_start_returns_zero(self) -> None:
        n = _count_transits(
            t_start=2459010.0,
            t_end=2459000.0,
            period_days=5.0,
            epoch_bjd=2459000.0,
            duration_days=0.1,
        )
        assert n == 0

    def test_zero_period_returns_zero(self) -> None:
        n = _count_transits(
            t_start=2459000.0,
            t_end=2459010.0,
            period_days=0.0,
            epoch_bjd=2459000.0,
            duration_days=0.1,
        )
        assert n == 0

    def test_negative_period_returns_zero(self) -> None:
        n = _count_transits(
            t_start=2459000.0,
            t_end=2459010.0,
            period_days=-1.0,
            epoch_bjd=2459000.0,
            duration_days=0.1,
        )
        assert n == 0

    def test_at_least_one_even_if_epoch_outside_baseline(self) -> None:
        # epoch is before the baseline — transit at epoch+2*period should be inside
        n = _count_transits(
            t_start=2459010.0,
            t_end=2459020.0,
            period_days=5.0,
            epoch_bjd=2459000.0,
            duration_days=0.1,
        )
        assert n >= 1

    def test_long_baseline_many_transits(self) -> None:
        n = _count_transits(
            t_start=2459000.0,
            t_end=2459365.0,
            period_days=1.0,
            epoch_bjd=2459000.5,
            duration_days=0.05,
        )
        assert n > 300

    def test_minimum_return_is_one(self) -> None:
        # max(1, ...) ensures we always return at least 1 when called
        n = _count_transits(
            t_start=2459000.0,
            t_end=2459000.2,
            period_days=100.0,
            epoch_bjd=2459000.1,
            duration_days=0.5,
        )
        assert n >= 1


# ---------------------------------------------------------------------------
# TestExtractFluxErr
# ---------------------------------------------------------------------------


class TestExtractFluxErr:
    def test_uses_lc_flux_err_when_valid(self) -> None:
        flux = np.ones(100)
        err = np.full(100, 0.001)
        lc = MagicMock()
        lc.flux_err.value = err
        result = _extract_flux_err(lc, flux)
        np.testing.assert_array_equal(result, err)

    def test_fallback_on_attribute_error(self) -> None:
        flux = np.ones(100) + np.random.default_rng(0).normal(0, 0.002, 100)
        lc = MagicMock(spec=[])  # no attributes
        result = _extract_flux_err(lc, flux)
        assert result.shape == flux.shape
        assert np.all(result > 0.0)

    def test_fallback_on_nan_errors(self) -> None:
        flux = np.ones(10)
        err = np.full(10, np.nan)
        lc = MagicMock()
        lc.flux_err.value = err
        result = _extract_flux_err(lc, flux)
        assert np.all(np.isfinite(result))
        assert np.all(result > 0.0)

    def test_fallback_on_zero_errors(self) -> None:
        flux = np.ones(10)
        err = np.zeros(10)
        lc = MagicMock()
        lc.flux_err.value = err
        result = _extract_flux_err(lc, flux)
        assert np.all(result > 0.0)

    def test_fallback_on_negative_errors(self) -> None:
        flux = np.ones(10)
        err = np.full(10, -0.001)
        lc = MagicMock()
        lc.flux_err.value = err
        result = _extract_flux_err(lc, flux)
        assert np.all(result > 0.0)

    def test_fallback_shape_matches_flux(self) -> None:
        flux = np.ones(77)
        lc = MagicMock(spec=[])
        result = _extract_flux_err(lc, flux)
        assert result.shape == (77,)

    def test_fallback_uses_mad(self) -> None:
        rng = np.random.default_rng(7)
        sigma = 0.005
        flux = np.ones(500) + rng.normal(0.0, sigma, 500)
        lc = MagicMock(spec=[])
        result = _extract_flux_err(lc, flux)
        # MAD-based estimate should be within a factor of 3 of true sigma
        assert result[0] == pytest.approx(result[1])  # uniform array
        assert 0.3 * sigma < result[0] < 3.0 * sigma

    def test_fallback_uniform_constant_flux(self) -> None:
        flux = np.ones(50)
        lc = MagicMock(spec=[])
        result = _extract_flux_err(lc, flux)
        # MAD of constant = 0, triggers the 1e-4 floor
        assert np.all(result == pytest.approx(1e-4))


# ---------------------------------------------------------------------------
# TestMakeCandidateId
# ---------------------------------------------------------------------------


class TestMakeCandidateId:
    def test_basic_formatting(self) -> None:
        assert _make_candidate_id("TIC 123456789", 1) == "TIC_123456789_s01"

    def test_peak_number_zero_padded(self) -> None:
        assert _make_candidate_id("TIC 1", 3) == "TIC_1_s03"

    def test_double_digit_peak_number(self) -> None:
        assert _make_candidate_id("KIC 999", 12) == "KIC_999_s12"

    def test_no_spaces_in_id(self) -> None:
        cid = _make_candidate_id("EPIC 201 002 003", 1)
        assert " " not in cid

    def test_replaces_all_spaces(self) -> None:
        cid = _make_candidate_id("A B C", 1)
        assert cid == "A_B_C_s01"


# ---------------------------------------------------------------------------
# TestSearchLightcurve — edge cases and short-circuit returns
# ---------------------------------------------------------------------------


class TestSearchLightcurveEdgeCases:
    def test_too_few_points_returns_empty(self) -> None:
        lc = MagicMock()
        lc.time.jd = np.array([2459000.0, 2459001.0])
        lc.flux.value = np.array([1.0, 1.0])
        lc.flux_err.value = np.array([0.001, 0.001])
        result = search_lightcurve(lc, "TIC 1", "TESS")
        assert result == []

    def test_returns_list(self) -> None:
        result = search_lightcurve(_flat_lc(n_points=500), "TIC 1", "TESS")
        assert isinstance(result, list)

    def test_max_peaks_zero_returns_empty(self) -> None:
        lc = _synthetic_lc()
        result = search_lightcurve(lc, "TIC 1", "TESS", max_peaks=0)
        assert result == []

    def test_very_high_snr_threshold_returns_empty(self) -> None:
        # min_snr=9999 should reject every peak
        result = search_lightcurve(
            _flat_lc(), "TIC 1", "TESS", min_snr=9999.0
        )
        assert result == []

    def test_period_max_less_than_period_min_returns_empty_gracefully(self) -> None:
        lc = _synthetic_lc(t_span_days=5.0, period_days=2.0)
        # period_max < period_min → autoperiod returns empty grid → breaks loop
        result = search_lightcurve(
            lc, "TIC 1", "TESS", period_min=10.0, period_max=5.0
        )
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# TestSearchLightcurve — signal recovery
# ---------------------------------------------------------------------------


class TestSearchLightcurveSignalRecovery:
    """Use real astropy BLS on synthetic data to verify end-to-end behaviour."""

    def test_finds_injected_transit(self) -> None:
        lc = _synthetic_lc(
            period_days=5.0,
            depth=0.01,
            duration_hours=2.5,
            noise=0.0005,
        )
        results = search_lightcurve(
            lc,
            "TIC 123456789",
            "TESS",
            period_min=1.0,
            period_max=13.0,
            min_snr=3.0,
        )
        assert len(results) >= 1

    def test_recovered_period_close_to_injected(self) -> None:
        injected_period = 5.0
        lc = _synthetic_lc(period_days=injected_period, depth=0.015, noise=0.0005)
        results = search_lightcurve(
            lc,
            "TIC 1",
            "TESS",
            period_min=1.0,
            period_max=13.0,
            min_snr=3.0,
        )
        assert len(results) >= 1
        recovered = results[0].period_days
        # Accept period or harmonic within 10%
        ratio = recovered / injected_period
        assert any(
            abs(ratio - r) < 0.1 for r in [0.5, 1.0, 2.0]
        ), f"Recovered period {recovered:.3f} d not close to injected {injected_period} d"

    def test_returns_candidate_signal_objects(self) -> None:
        from exo_toolkit.schemas import CandidateSignal

        lc = _synthetic_lc(depth=0.015, noise=0.0005)
        results = search_lightcurve(lc, "TIC 1", "TESS", min_snr=3.0)
        for sig in results:
            assert isinstance(sig, CandidateSignal)

    def test_candidate_id_uses_target_id(self) -> None:
        lc = _synthetic_lc(depth=0.015, noise=0.0005)
        results = search_lightcurve(lc, "TIC 999", "TESS", min_snr=3.0)
        if results:
            assert results[0].candidate_id.startswith("TIC_999")

    def test_mission_stored_correctly(self) -> None:
        lc = _synthetic_lc(depth=0.015, noise=0.0005)
        for mission in ("TESS", "Kepler", "K2"):
            results = search_lightcurve(lc, "TIC 1", mission, min_snr=3.0)  # type: ignore[arg-type]
            for sig in results:
                assert sig.mission == mission

    def test_snr_above_threshold(self) -> None:
        threshold = 5.0
        lc = _synthetic_lc(depth=0.015, noise=0.0005)
        results = search_lightcurve(lc, "TIC 1", "TESS", min_snr=threshold)
        for sig in results:
            assert sig.snr >= threshold

    def test_depth_ppm_positive(self) -> None:
        lc = _synthetic_lc(depth=0.01, noise=0.0005)
        results = search_lightcurve(lc, "TIC 1", "TESS", min_snr=3.0)
        for sig in results:
            assert sig.depth_ppm > 0.0

    def test_duration_hours_positive(self) -> None:
        lc = _synthetic_lc(depth=0.01, noise=0.0005)
        results = search_lightcurve(lc, "TIC 1", "TESS", min_snr=3.0)
        for sig in results:
            assert sig.duration_hours > 0.0

    def test_transit_count_positive_integer(self) -> None:
        lc = _synthetic_lc(depth=0.01, noise=0.0005)
        results = search_lightcurve(lc, "TIC 1", "TESS", min_snr=3.0)
        for sig in results:
            assert isinstance(sig.transit_count, int)
            assert sig.transit_count >= 1

    def test_max_peaks_limits_results(self) -> None:
        lc = _synthetic_lc(depth=0.015, noise=0.0005)
        results = search_lightcurve(lc, "TIC 1", "TESS", max_peaks=1, min_snr=3.0)
        assert len(results) <= 1

    def test_candidate_ids_are_unique(self) -> None:
        lc = _synthetic_lc(depth=0.015, noise=0.0005)
        results = search_lightcurve(lc, "TIC 1", "TESS", min_snr=3.0, max_peaks=3)
        ids = [s.candidate_id for s in results]
        assert len(ids) == len(set(ids))

    def test_no_flux_err_attribute_still_works(self) -> None:
        # Falls back to MAD-based error estimation
        lc = _synthetic_lc(depth=0.015, noise=0.0005, include_flux_err=False)
        lc.flux_err = MagicMock(spec=[])  # accessing .value raises AttributeError
        results = search_lightcurve(lc, "TIC 1", "TESS", min_snr=3.0)
        assert isinstance(results, list)

    def test_period_max_none_uses_half_baseline(self) -> None:
        # period_max=None → half the baseline; should not raise
        lc = _synthetic_lc(t_span_days=27.0, period_days=5.0, depth=0.015, noise=0.0005)
        results = search_lightcurve(lc, "TIC 1", "TESS", period_max=None, min_snr=3.0)
        assert isinstance(results, list)

    def test_depth_ppm_magnitude(self) -> None:
        injected_depth = 0.01  # 1 % = 10_000 ppm
        lc = _synthetic_lc(depth=injected_depth, noise=0.0005)
        results = search_lightcurve(lc, "TIC 1", "TESS", min_snr=3.0)
        if results:
            # Within order of magnitude: 1000–100000 ppm
            assert 1_000.0 < results[0].depth_ppm < 100_000.0

    def test_kepler_mission_accepted(self) -> None:
        lc = _synthetic_lc(depth=0.015, noise=0.0005)
        results = search_lightcurve(lc, "KIC 123", "Kepler", min_snr=3.0)
        assert isinstance(results, list)

    def test_k2_mission_accepted(self) -> None:
        lc = _synthetic_lc(depth=0.015, noise=0.0005)
        results = search_lightcurve(lc, "EPIC 123", "K2", min_snr=3.0)
        assert isinstance(results, list)
