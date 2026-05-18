"""Tests for Skills.ephemeris_uncertainty_growth."""
from __future__ import annotations

import pytest
from Skills.ephemeris_uncertainty_growth import (
    EphemerisUncertaintyResult,
    format_ephemeris_uncertainty_result,
    project_ephemeris_uncertainty,
)


class TestProjectEphemerisUncertainty:
    def test_returns_result(self) -> None:
        r = project_ephemeris_uncertainty(
            2458000.0, 5.0, epoch_err_days=0.001, period_err_days=0.0001
        )
        assert isinstance(r, EphemerisUncertaintyResult)

    def test_zero_period_returns_poorly_constrained(self) -> None:
        r = project_ephemeris_uncertainty(
            2458000.0, 0.0, epoch_err_days=0.001, period_err_days=0.0001
        )
        assert r.flag == "POORLY_CONSTRAINED"

    def test_n_predictions_correct(self) -> None:
        r = project_ephemeris_uncertainty(
            2458000.0, 5.0, epoch_err_days=0.001, period_err_days=0.0001, n_cycles=7
        )
        assert len(r.predictions) == 7

    def test_sigma_increases_with_cycle(self) -> None:
        r = project_ephemeris_uncertainty(
            2458000.0, 5.0, epoch_err_days=0.001, period_err_days=0.01, n_cycles=5
        )
        sigmas = [p.sigma_minutes for p in r.predictions]
        assert sigmas[-1] >= sigmas[0]

    def test_window_hours_3_sigma(self) -> None:
        r = project_ephemeris_uncertainty(
            2458000.0, 5.0, epoch_err_days=0.001, period_err_days=0.0001
        )
        for p in r.predictions:
            assert abs(p.window_hours - 3.0 * p.sigma_minutes / 60.0) < 0.001

    def test_well_constrained_flag(self) -> None:
        r = project_ephemeris_uncertainty(
            2458000.0, 5.0,
            epoch_err_days=1e-5, period_err_days=1e-6,
            n_cycles=5,
            well_constrained_minutes=30.0,
        )
        assert r.flag == "WELL_CONSTRAINED"

    def test_poorly_constrained_flag(self) -> None:
        r = project_ephemeris_uncertainty(
            2458000.0, 5.0,
            epoch_err_days=0.1, period_err_days=0.01,
            n_cycles=10,
            poorly_constrained_minutes=60.0,
        )
        assert r.flag == "POORLY_CONSTRAINED"

    def test_sigma_t0_stored(self) -> None:
        r = project_ephemeris_uncertainty(
            2458000.0, 5.0, epoch_err_days=0.001, period_err_days=0.0001
        )
        assert r.sigma_t0_minutes == pytest.approx(0.001 * 1440, rel=1e-4)

    def test_transit_number_int(self) -> None:
        r = project_ephemeris_uncertainty(
            2458000.0, 5.0, epoch_err_days=0.001, period_err_days=0.0001
        )
        for p in r.predictions:
            assert isinstance(p.transit_number, int)


class TestFormatEphemerisUncertainty:
    def test_returns_string(self) -> None:
        r = project_ephemeris_uncertainty(
            2458000.0, 5.0, epoch_err_days=0.001, period_err_days=0.0001
        )
        assert isinstance(format_ephemeris_uncertainty_result(r), str)

    def test_contains_flag(self) -> None:
        r = project_ephemeris_uncertainty(
            2458000.0, 5.0, epoch_err_days=0.001, period_err_days=0.0001
        )
        assert r.flag in format_ephemeris_uncertainty_result(r)
