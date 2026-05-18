"""Tests for Skills.scatter_metric_calculator."""
from __future__ import annotations

from Skills.scatter_metric_calculator import (
    ScatterMetricResult,
    compute_scatter_metrics,
    format_scatter_metrics,
)


def _make_flux(n=1000, noise=1e-4):
    import random
    random.seed(42)
    return [1.0 + random.gauss(0, noise) for _ in range(n)]


class TestComputeScatterMetrics:
    def test_returns_result(self) -> None:
        f = _make_flux()
        r = compute_scatter_metrics(list(range(len(f))), f)
        assert isinstance(r, ScatterMetricResult)

    def test_insufficient_for_empty(self) -> None:
        r = compute_scatter_metrics([], [])
        assert r.flag == "INSUFFICIENT"

    def test_insufficient_for_single_point(self) -> None:
        r = compute_scatter_metrics([0.0], [1.0])
        assert r.flag == "INSUFFICIENT"

    def test_ok_flag(self) -> None:
        f = _make_flux()
        r = compute_scatter_metrics(list(range(len(f))), f)
        assert r.flag == "OK"

    def test_rms_positive(self) -> None:
        f = _make_flux()
        r = compute_scatter_metrics(list(range(len(f))), f)
        assert r.rms_ppm > 0

    def test_mad_positive(self) -> None:
        f = _make_flux()
        r = compute_scatter_metrics(list(range(len(f))), f)
        assert r.mad_ppm > 0

    def test_cdpp_leq_rms(self) -> None:
        f = _make_flux()
        r = compute_scatter_metrics(list(range(len(f))), f)
        # CDPP averages over multiple cadences
        assert r.cdpp_6hr_ppm <= r.rms_ppm + 1.0  # allow floating-point slack

    def test_point_to_point_positive(self) -> None:
        f = _make_flux()
        r = compute_scatter_metrics(list(range(len(f))), f)
        assert r.point_to_point_ppm >= 0

    def test_n_points_correct(self) -> None:
        f = _make_flux(100)
        r = compute_scatter_metrics(list(range(100)), f)
        assert r.n_points == 100

    def test_higher_noise_higher_rms(self) -> None:
        f1 = _make_flux(noise=1e-4)
        f2 = _make_flux(noise=1e-3)
        r1 = compute_scatter_metrics(list(range(len(f1))), f1)
        r2 = compute_scatter_metrics(list(range(len(f2))), f2)
        assert r2.rms_ppm > r1.rms_ppm


class TestFormatScatterMetrics:
    def test_returns_string(self) -> None:
        f = _make_flux()
        r = compute_scatter_metrics(list(range(len(f))), f)
        assert isinstance(format_scatter_metrics(r), str)

    def test_insufficient_handled(self) -> None:
        r = compute_scatter_metrics([], [])
        out = format_scatter_metrics(r)
        assert "INSUFFICIENT" in out

    def test_contains_rms(self) -> None:
        f = _make_flux()
        r = compute_scatter_metrics(list(range(len(f))), f)
        assert "RMS" in format_scatter_metrics(r)
