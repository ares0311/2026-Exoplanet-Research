"""Tests for Skills.transit_modeler."""
from __future__ import annotations

import numpy as np
import pytest
from Skills.transit_modeler import (
    TransitModelResult,
    fit_transit_model,
    format_model_result,
    transit_model,
)


def _box_lc(
    n: int = 300,
    period: float = 5.0,
    epoch: float = 2458002.0,
    depth: float = 0.01,
    duration: float = 0.1,
) -> tuple[list[float], list[float]]:
    rng = np.random.default_rng(42)
    t = list(np.linspace(2458000.0, 2458027.0, n))
    f = []
    for ti in t:
        ph = (ti - epoch) % period
        if ph > period / 2:
            ph -= period
        f.append(1.0 - depth if abs(ph) <= duration / 2 else 1.0)
    noise = rng.normal(0, 5e-4, n)
    return t, [fi + ni for fi, ni in zip(f, noise, strict=False)]


class TestTransitModel:
    def test_out_of_transit_is_one(self) -> None:
        vals = transit_model([0.5, 1.0], 0.01, 0.1, 5.0)
        assert all(v == pytest.approx(1.0) for v in vals)

    def test_in_transit_depressed(self) -> None:
        vals = transit_model([0.0], 0.01, 0.1, 5.0)
        assert vals[0] < 1.0

    def test_depth_scales_linearly(self) -> None:
        v1 = transit_model([0.0], 0.01, 0.1, 5.0)
        v2 = transit_model([0.0], 0.02, 0.1, 5.0)
        assert (1.0 - v2[0]) > (1.0 - v1[0])

    def test_ingress_partial_depth(self) -> None:
        # At exactly half the duration from centre → ingress region
        vals = transit_model([0.049], 0.01, 0.1, 5.0)
        assert 0.95 < vals[0] < 1.0

    def test_list_input_returns_list(self) -> None:
        result = transit_model([0.0, 0.5, 1.0], 0.01, 0.1, 5.0)
        assert isinstance(result, list)
        assert len(result) == 3


class TestFitTransitModel:
    def test_returns_transit_model_result(self) -> None:
        t, f = _box_lc()
        result = fit_transit_model(t, f, 5.0, 2458002.0)
        assert isinstance(result, TransitModelResult)

    def test_depth_positive(self) -> None:
        t, f = _box_lc(depth=0.01)
        result = fit_transit_model(t, f, 5.0, 2458002.0, duration_days=0.1)
        assert result.depth_ppm > 0

    def test_duration_positive(self) -> None:
        t, f = _box_lc()
        result = fit_transit_model(t, f, 5.0, 2458002.0)
        assert result.duration_hours > 0

    def test_period_stored(self) -> None:
        t, f = _box_lc(period=7.0)
        result = fit_transit_model(t, f, 7.0, 2458002.0)
        assert result.period_days == pytest.approx(7.0)

    def test_chi2_reduced_non_negative(self) -> None:
        t, f = _box_lc()
        result = fit_transit_model(t, f, 5.0, 2458002.0)
        assert result.chi2_reduced >= 0.0

    def test_converged_is_bool(self) -> None:
        t, f = _box_lc()
        result = fit_transit_model(t, f, 5.0, 2458002.0)
        assert isinstance(result.converged, bool)

    def test_rms_residual_non_negative(self) -> None:
        t, f = _box_lc()
        result = fit_transit_model(t, f, 5.0, 2458002.0)
        assert result.rms_residual >= 0.0

    def test_flat_flux_gives_small_depth(self) -> None:
        t = list(np.linspace(2458000.0, 2458027.0, 200))
        f = [1.0] * 200
        result = fit_transit_model(t, f, 5.0, 2458002.0)
        assert result.depth_ppm < 1000.0


class TestFormatModelResult:
    def test_format_contains_depth(self) -> None:
        t, f = _box_lc()
        result = fit_transit_model(t, f, 5.0, 2458002.0)
        text = format_model_result(result)
        assert "Depth" in text
        assert "ppm" in text
