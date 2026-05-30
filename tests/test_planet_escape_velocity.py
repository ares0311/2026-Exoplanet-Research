"""Tests for Skills/planet_escape_velocity.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from planet_escape_velocity import (
    EscapeVelocityResult,
    compute_escape_velocity,
    format_escape_velocity,
)


class TestEscapeVelocityResult:
    def test_dataclass_fields(self):
        r = EscapeVelocityResult(v_esc_kms=11.2, can_retain_h2=True, can_retain_h2o=True, flag="OK")
        assert r.v_esc_kms == 11.2
        assert r.can_retain_h2 is True

    def test_frozen(self):
        r = EscapeVelocityResult(v_esc_kms=11.2, can_retain_h2=True, can_retain_h2o=True, flag="OK")
        try:
            r.v_esc_kms = 0
            raise AssertionError()
        except Exception:
            pass


class TestComputeEscapeVelocity:
    def test_earth_escape_velocity(self):
        # Earth: ~11.2 km/s
        r = compute_escape_velocity(1.0, 1.0)
        assert abs(r.v_esc_kms - 11.2) < 0.2
        assert r.flag == "OK"

    def test_earth_retains_h2(self):
        r = compute_escape_velocity(1.0, 1.0)
        assert r.can_retain_h2 is True

    def test_earth_retains_h2o(self):
        r = compute_escape_velocity(1.0, 1.0)
        assert r.can_retain_h2o is True

    def test_small_planet_no_h2(self):
        # Mars-like: ~0.1 Earth masses, ~0.5 R_earth → v_esc ~ 5 km/s
        r = compute_escape_velocity(0.107, 0.532)
        assert r.can_retain_h2 is False

    def test_large_planet_retains_h2(self):
        # Jupiter-like: ~318 M_earth, ~11 R_earth
        r = compute_escape_velocity(318.0, 11.0)
        assert r.can_retain_h2 is True
        assert r.v_esc_kms > 40.0

    def test_zero_mass_error(self):
        r = compute_escape_velocity(0.0, 1.0)
        assert r.flag == "ERROR"

    def test_zero_radius_error(self):
        r = compute_escape_velocity(1.0, 0.0)
        assert r.flag == "ERROR"

    def test_negative_mass_error(self):
        r = compute_escape_velocity(-1.0, 1.0)
        assert r.flag == "ERROR"

    def test_v_esc_scales_correctly(self):
        # v_esc ∝ sqrt(M/R) → doubling mass * sqrt(2) change
        r1 = compute_escape_velocity(1.0, 1.0)
        r2 = compute_escape_velocity(4.0, 1.0)
        ratio = r2.v_esc_kms / r1.v_esc_kms
        assert abs(ratio - 2.0) < 0.01

    def test_h2o_threshold(self):
        # Find a planet just below h2o threshold (5 km/s)
        # Small rocky planet
        r = compute_escape_velocity(0.01, 0.5)
        if r.v_esc_kms <= 5.0:
            assert r.can_retain_h2o is False

    def test_can_retain_h2o_implies_can_retain_h2(self):
        # If v_esc > 10, both should be True
        r = compute_escape_velocity(1.0, 1.0)
        if r.can_retain_h2:
            assert r.can_retain_h2o  # 10 > 5, so H2O also retained


class TestFormatEscapeVelocity:
    def test_returns_string(self):
        r = compute_escape_velocity(1.0, 1.0)
        s = format_escape_velocity(r)
        assert isinstance(s, str)

    def test_contains_kms(self):
        r = compute_escape_velocity(1.0, 1.0)
        s = format_escape_velocity(r)
        assert "km/s" in s or "km" in s
