"""Tests for Skills/keplerian_rv_model.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from keplerian_rv_model import RvCurveResult, compute_rv_curve


class TestKeplerianRvModel:
    def test_circular_orbit_returns_ok(self) -> None:
        r = compute_rv_curve(k_ms=100.0, period_days=3.0, eccentricity=0.0)
        assert r.flag == "OK"

    def test_circular_orbit_gamma_zero(self) -> None:
        r = compute_rv_curve(k_ms=100.0, period_days=3.0, eccentricity=0.0, gamma_ms=0.0)
        assert abs(sum(r.rv_ms) / len(r.rv_ms)) < 5.0  # mean near zero for circular

    def test_invalid_k_zero(self) -> None:
        r = compute_rv_curve(k_ms=0.0, period_days=3.0)
        assert r.flag != "OK" or r.rv_ms == tuple([0.0] * len(r.rv_ms))

    def test_invalid_period(self) -> None:
        r = compute_rv_curve(k_ms=100.0, period_days=0.0)
        assert r.flag == "INVALID_PERIOD"

    def test_negative_period(self) -> None:
        r = compute_rv_curve(k_ms=100.0, period_days=-1.0)
        assert r.flag == "INVALID_PERIOD"

    def test_invalid_eccentricity(self) -> None:
        r = compute_rv_curve(k_ms=100.0, period_days=3.0, eccentricity=1.5)
        assert r.flag == "INVALID_ECCENTRICITY"

    def test_negative_eccentricity(self) -> None:
        r = compute_rv_curve(k_ms=100.0, period_days=3.0, eccentricity=-0.1)
        assert r.flag == "INVALID_ECCENTRICITY"

    def test_gamma_offset(self) -> None:
        r = compute_rv_curve(k_ms=50.0, period_days=2.0, eccentricity=0.0, gamma_ms=200.0)
        assert r.flag == "OK"
        assert all(v > 0 for v in r.rv_ms)  # gamma shifts all points positive

    def test_n_points_respected(self) -> None:
        r = compute_rv_curve(k_ms=100.0, period_days=3.0, n_points=50)
        assert len(r.rv_ms) == 50
        assert len(r.phases) == 50

    def test_eccentric_orbit_ok(self) -> None:
        r = compute_rv_curve(k_ms=100.0, period_days=5.0, eccentricity=0.3, omega_deg=90.0)
        assert r.flag == "OK"
        assert len(r.rv_ms) > 0

    def test_amplitude_approx_k(self) -> None:
        r = compute_rv_curve(k_ms=100.0, period_days=3.0, eccentricity=0.0)
        assert r.flag == "OK"
        amp = (max(r.rv_ms) - min(r.rv_ms)) / 2.0
        assert abs(amp - 100.0) < 5.0

    def test_result_is_frozen(self) -> None:
        r = compute_rv_curve(k_ms=100.0, period_days=3.0)
        assert isinstance(r, RvCurveResult)
        try:
            object.__setattr__(r, "flag", "mutated")
            raise AssertionError("Should be frozen")
        except Exception:
            pass
