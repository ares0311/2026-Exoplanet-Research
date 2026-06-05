"""Tests for Skills/geometric_albedo_from_phase.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from geometric_albedo_from_phase import (
    compute_geometric_albedo_from_phase,
    format_geometric_albedo_result,
)


class TestComputeGeometricAlbedoFromPhase:
    def test_ok_flag(self) -> None:
        r = compute_geometric_albedo_from_phase(100.0, 0.05, 1.0)
        assert r.flag == "OK"

    def test_geometric_albedo_positive(self) -> None:
        r = compute_geometric_albedo_from_phase(100.0, 0.05, 1.0)
        assert r.geometric_albedo > 0.0

    def test_geometric_albedo_le_one(self) -> None:
        r = compute_geometric_albedo_from_phase(100.0, 0.05, 1.0)
        assert r.geometric_albedo <= 1.0

    def test_spherical_albedo_le_one(self) -> None:
        r = compute_geometric_albedo_from_phase(100.0, 0.05, 1.0)
        assert r.spherical_albedo <= 1.0

    def test_larger_amplitude_larger_albedo(self) -> None:
        r1 = compute_geometric_albedo_from_phase(50.0, 0.05, 1.0)
        r2 = compute_geometric_albedo_from_phase(200.0, 0.05, 1.0)
        assert r2.geometric_albedo > r1.geometric_albedo

    def test_farther_orbit_larger_albedo(self) -> None:
        r_close = compute_geometric_albedo_from_phase(100.0, 0.03, 1.0)
        r_far = compute_geometric_albedo_from_phase(100.0, 0.10, 1.0)
        assert r_far.geometric_albedo > r_close.geometric_albedo

    def test_phase_integral_preserved(self) -> None:
        r = compute_geometric_albedo_from_phase(100.0, 0.05, 1.0, phase_integral=1.2)
        assert r.phase_integral == 1.2

    def test_invalid_amplitude(self) -> None:
        r = compute_geometric_albedo_from_phase(0.0, 0.05, 1.0)
        assert r.flag == "INVALID_AMPLITUDE"

    def test_invalid_distance(self) -> None:
        r = compute_geometric_albedo_from_phase(100.0, 0.0, 1.0)
        assert r.flag == "INVALID_DISTANCE"

    def test_invalid_radius(self) -> None:
        r = compute_geometric_albedo_from_phase(100.0, 0.05, 0.0)
        assert r.flag == "INVALID_RADIUS"

    def test_result_frozen(self) -> None:
        r = compute_geometric_albedo_from_phase(100.0, 0.05, 1.0)
        try:
            r.geometric_albedo = 0.0  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except (AttributeError, TypeError):
            pass

    def test_format_returns_string(self) -> None:
        r = compute_geometric_albedo_from_phase(100.0, 0.05, 1.0)
        s = format_geometric_albedo_result(r)
        assert isinstance(s, str)
        assert r.flag in s
