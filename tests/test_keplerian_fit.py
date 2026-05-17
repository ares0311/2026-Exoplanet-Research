"""Tests for Skills.keplerian_fit."""
from __future__ import annotations

import numpy as np
import pytest
from Skills.keplerian_fit import TrapezoidFit, fit_trapezoid, trapezoid_model


class TestTrapezoidModel:
    def test_out_of_transit_is_one(self) -> None:
        phase = np.array([-0.4, 0.4])
        model = trapezoid_model(phase, 1000.0, 0.05, 0.2)
        assert np.allclose(model, 1.0, atol=1e-9)

    def test_in_transit_center_is_depressed(self) -> None:
        phase = np.array([0.0])
        model = trapezoid_model(phase, 1000.0, 0.10, 0.1)
        assert model[0] < 1.0

    def test_depth_scales_transit_bottom(self) -> None:
        phase = np.array([0.0])
        m1 = trapezoid_model(phase, 500.0, 0.10, 0.01)
        m2 = trapezoid_model(phase, 1000.0, 0.10, 0.01)
        assert m2[0] < m1[0]

    def test_ingress_frac_clipped_at_minimum(self) -> None:
        phase = np.linspace(-0.3, 0.3, 100)
        model = trapezoid_model(phase, 1000.0, 0.10, 0.0)
        assert np.all(model <= 1.0 + 1e-9)

    def test_returns_array_same_shape_as_input(self) -> None:
        phase = np.linspace(-0.5, 0.5, 200)
        model = trapezoid_model(phase, 1000.0, 0.10, 0.2)
        assert model.shape == phase.shape


class TestFitTrapezoid:
    def _injected_transit(
        self,
        period: float = 10.0,
        depth_ppm: float = 5000.0,
        dur_phase: float = 0.05,
    ) -> tuple[np.ndarray, np.ndarray]:
        phase = np.linspace(-0.5, 0.5, 500)
        flux = trapezoid_model(phase, depth_ppm, dur_phase, 0.2)
        return phase, flux

    def test_returns_trapezoid_fit_instance(self) -> None:
        phase, flux = self._injected_transit()
        result = fit_trapezoid(phase, flux, period_days=10.0)
        assert isinstance(result, TrapezoidFit)

    def test_recovers_depth_approximately(self) -> None:
        phase, flux = self._injected_transit(depth_ppm=5000.0)
        result = fit_trapezoid(phase, flux, period_days=10.0)
        assert abs(result.depth_ppm - 5000.0) < 2000.0

    def test_duration_hours_positive(self) -> None:
        phase, flux = self._injected_transit()
        result = fit_trapezoid(phase, flux, period_days=10.0)
        assert result.duration_hours > 0.0

    def test_period_days_stored_in_result(self) -> None:
        phase, flux = self._injected_transit()
        result = fit_trapezoid(phase, flux, period_days=7.0)
        assert result.period_days == pytest.approx(7.0)

    def test_chi2_reduced_non_negative(self) -> None:
        phase, flux = self._injected_transit()
        result = fit_trapezoid(phase, flux, period_days=10.0)
        assert result.chi2_reduced >= 0.0

    def test_flat_flux_gives_near_zero_depth(self) -> None:
        phase = np.linspace(-0.5, 0.5, 200)
        flux = np.ones(200)
        result = fit_trapezoid(phase, flux, period_days=10.0)
        assert result.depth_ppm < 500.0

    def test_mismatched_arrays_raise_value_error(self) -> None:
        phase = np.linspace(-0.5, 0.5, 100)
        flux = np.ones(50)
        with pytest.raises(ValueError):
            fit_trapezoid(phase, flux, period_days=10.0)

    def test_with_flux_err_does_not_crash(self) -> None:
        phase, flux = self._injected_transit()
        flux_err = np.full_like(flux, 1e-3)
        result = fit_trapezoid(phase, flux, period_days=10.0, flux_err=flux_err)
        assert isinstance(result, TrapezoidFit)

    def test_ingress_fraction_in_valid_range(self) -> None:
        phase, flux = self._injected_transit()
        result = fit_trapezoid(phase, flux, period_days=10.0)
        assert 0.0 < result.ingress_fraction < 0.5

    def test_duration_hours_scales_with_period(self) -> None:
        phase, flux = self._injected_transit(dur_phase=0.05)
        r1 = fit_trapezoid(phase, flux, period_days=5.0)
        r2 = fit_trapezoid(phase, flux, period_days=10.0)
        assert r2.duration_hours > r1.duration_hours

    def test_converged_field_is_bool(self) -> None:
        phase, flux = self._injected_transit()
        result = fit_trapezoid(phase, flux, period_days=10.0)
        assert isinstance(result.converged, bool)
