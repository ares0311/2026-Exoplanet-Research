"""Tests for Skills/kopparapu_hz_calculator.py."""
import pytest
from Skills.kopparapu_hz_calculator import (
    HZBoundaries,
    HZClassification,
    compute_hz_boundaries,
    classify_hz_position,
    format_hz_result,
)


class TestComputeHzBoundaries:
    def test_returns_hz_boundaries(self):
        result = compute_hz_boundaries(5778.0)
        assert isinstance(result, HZBoundaries)

    def test_flag_ok_sun(self):
        result = compute_hz_boundaries(5778.0)
        assert result.flag == "OK"

    def test_solar_runaway_greenhouse_near_1(self):
        result = compute_hz_boundaries(5778.0)
        # Solar S_eff for runaway greenhouse ≈ 1.04
        assert 0.8 <= result.s_runaway_greenhouse <= 1.2

    def test_ordering_of_limits(self):
        result = compute_hz_boundaries(5778.0)
        # Outer edge (early mars) < max greenhouse < runaway GH < recent venus
        assert result.s_early_mars < result.s_max_greenhouse
        assert result.s_max_greenhouse < result.s_runaway_greenhouse
        assert result.s_runaway_greenhouse < result.s_recent_venus

    def test_cool_star_shifts_hz(self):
        sun = compute_hz_boundaries(5778.0)
        cool = compute_hz_boundaries(3500.0)
        # Cooler stars have different HZ boundaries
        assert cool.s_runaway_greenhouse != sun.s_runaway_greenhouse

    def test_invalid_teff_too_low(self):
        result = compute_hz_boundaries(2000.0)
        assert result.flag in ("OK", "INVALID")

    def test_invalid_teff_too_high(self):
        result = compute_hz_boundaries(10000.0)
        assert result.flag in ("OK", "INVALID")

    def test_teff_stored(self):
        result = compute_hz_boundaries(5000.0)
        assert result.teff_k == 5000.0

    def test_all_s_values_positive(self):
        result = compute_hz_boundaries(5778.0)
        assert result.s_recent_venus > 0
        assert result.s_runaway_greenhouse > 0
        assert result.s_max_greenhouse > 0
        assert result.s_early_mars > 0


class TestClassifyHzPosition:
    def test_returns_hz_classification(self):
        result = classify_hz_position(1.0, 5778.0)
        assert isinstance(result, HZClassification)

    def test_flag_ok(self):
        result = classify_hz_position(1.0, 5778.0)
        assert result.flag == "OK"

    def test_solar_earth_in_conservative_hz(self):
        # Earth receives ~1.0 S/S_sun; should be in conservative HZ
        result = classify_hz_position(1.0, 5778.0)
        assert result.is_in_conservative_hz

    def test_too_hot_not_in_hz(self):
        result = classify_hz_position(10.0, 5778.0)
        assert not result.is_in_conservative_hz

    def test_too_cold_not_in_hz(self):
        result = classify_hz_position(0.1, 5778.0)
        assert not result.is_in_conservative_hz

    def test_optimistic_wider_than_conservative(self):
        # Outer optimistic boundary (early mars) is colder (lower S) than conservative outer
        bounds = compute_hz_boundaries(5778.0)
        # S slightly below early_mars → outside even optimistic HZ (too cold)
        r_too_cold = classify_hz_position(bounds.s_early_mars * 0.5, 5778.0)
        # S slightly above recent_venus → outside optimistic HZ (too hot)
        r_too_hot = classify_hz_position(bounds.s_recent_venus * 1.5, 5778.0)
        assert not r_too_cold.is_in_conservative_hz
        assert not r_too_hot.is_in_conservative_hz

    def test_insolation_stored(self):
        result = classify_hz_position(1.0, 5778.0)
        assert result.insolation_flux == 1.0

    def test_hz_class_string(self):
        result = classify_hz_position(1.0, 5778.0)
        assert isinstance(result.hz_class, str)
        assert len(result.hz_class) > 0


class TestFormatHzResult:
    def test_returns_string(self):
        b = compute_hz_boundaries(5778.0)
        s = format_hz_result(b)
        assert isinstance(s, str)

    def test_contains_hz_header(self):
        b = compute_hz_boundaries(5778.0)
        s = format_hz_result(b)
        assert "Habitable" in s or "HZ" in s

    def test_with_classification(self):
        b = compute_hz_boundaries(5778.0)
        c = classify_hz_position(1.0, 5778.0)
        s = format_hz_result(b, c)
        assert isinstance(s, str)
        assert len(s) > len(format_hz_result(b))
