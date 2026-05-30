"""Tests for Skills/rv_curve_simulator.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from rv_curve_simulator import format_rv_curve, simulate_rv_curve


class TestSimulateRvCurve:
    def test_basic_circular(self) -> None:
        r = simulate_rv_curve(10.0, 5.0)
        assert r.flag == "OK"
        assert len(r.rv_ms) == 100

    def test_n_points_stored(self) -> None:
        r = simulate_rv_curve(10.0, 5.0, n_points=50)
        assert len(r.phases) == 50
        assert len(r.rv_ms) == 50

    def test_invalid_k_amplitude(self) -> None:
        r = simulate_rv_curve(-1.0, 5.0)
        assert r.flag == "INVALID_K_AMPLITUDE"

    def test_invalid_period(self) -> None:
        r = simulate_rv_curve(10.0, 0.0)
        assert r.flag == "INVALID_PERIOD"

    def test_invalid_eccentricity(self) -> None:
        r = simulate_rv_curve(10.0, 5.0, eccentricity=1.0)
        assert r.flag == "INVALID_ECCENTRICITY"

    def test_circular_amplitude(self) -> None:
        r = simulate_rv_curve(10.0, 5.0, eccentricity=0.0, omega_deg=90.0)
        assert abs(max(r.rv_ms) - 10.0) < 0.5
        assert abs(min(r.rv_ms) + 10.0) < 0.5

    def test_phases_range(self) -> None:
        r = simulate_rv_curve(10.0, 5.0)
        assert r.phases[0] == 0.0
        assert r.phases[-1] < 1.0

    def test_eccentricity_stored(self) -> None:
        r = simulate_rv_curve(10.0, 5.0, eccentricity=0.3)
        assert r.eccentricity == 0.3

    def test_period_stored(self) -> None:
        r = simulate_rv_curve(10.0, 7.3)
        assert r.period_days == 7.3

    def test_k_amplitude_stored(self) -> None:
        r = simulate_rv_curve(15.0, 5.0)
        assert r.k_amplitude_ms == 15.0

    def test_format_returns_string(self) -> None:
        r = simulate_rv_curve(10.0, 5.0)
        s = format_rv_curve(r)
        assert isinstance(s, str)
        assert "RV" in s

    def test_zero_k_all_zero_rv(self) -> None:
        r = simulate_rv_curve(0.0, 5.0, eccentricity=0.0, omega_deg=90.0)
        assert r.flag == "OK"
        assert all(abs(v) < 1e-9 for v in r.rv_ms)
