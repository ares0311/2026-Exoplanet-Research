"""Tests for Skills/insolation_flux_calculator.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from insolation_flux_calculator import (
    InsolationResult,
    compute_insolation,
    format_insolation_result,
)


class TestInsolationResult:
    def test_dataclass_fields(self):
        r = InsolationResult(insolation_earth_units=1.0, hz_class="inner_hz", flag="OK")
        assert r.insolation_earth_units == 1.0
        assert r.hz_class == "inner_hz"
        assert r.flag == "OK"

    def test_frozen(self):
        r = InsolationResult(insolation_earth_units=1.0, hz_class="inner_hz", flag="OK")
        try:
            r.flag = "FAIL"
            raise AssertionError("Should be frozen")
        except Exception:
            pass


class TestComputeInsolation:
    def test_earth_like(self):
        r = compute_insolation(1.0, 1.0)
        assert abs(r.insolation_earth_units - 1.0) < 1e-9
        assert r.flag == "OK"

    def test_hot_zone(self):
        r = compute_insolation(1.0, 0.5)
        assert r.hz_class == "hot_zone"
        assert r.insolation_earth_units > 1.1

    def test_inner_hz(self):
        r = compute_insolation(1.0, 1.0)
        # S=1.0 which is > 0.36 and <= 1.1 → inner_hz
        assert r.hz_class == "inner_hz"

    def test_outer_hz(self):
        r = compute_insolation(0.1, 0.55)
        # S = 0.1/0.55^2 ~ 0.33 → outer_hz
        assert r.hz_class in ("outer_hz", "cold_zone")

    def test_cold_zone(self):
        r = compute_insolation(1.0, 10.0)
        assert r.hz_class == "cold_zone"
        assert r.insolation_earth_units < 0.20

    def test_zero_luminosity_returns_error(self):
        r = compute_insolation(0.0, 1.0)
        assert r.flag == "ERROR"

    def test_zero_distance_returns_error(self):
        r = compute_insolation(1.0, 0.0)
        assert r.flag == "ERROR"

    def test_negative_inputs_return_error(self):
        r = compute_insolation(-1.0, 1.0)
        assert r.flag == "ERROR"

    def test_high_luminosity(self):
        r = compute_insolation(10.0, 1.0)
        assert r.insolation_earth_units == 10.0

    def test_distance_squared_scaling(self):
        r1 = compute_insolation(1.0, 1.0)
        r2 = compute_insolation(1.0, 2.0)
        assert abs(r1.insolation_earth_units / r2.insolation_earth_units - 4.0) < 1e-6


class TestFormatInsolation:
    def test_returns_string(self):
        r = compute_insolation(1.0, 1.0)
        s = format_insolation_result(r)
        assert isinstance(s, str)

    def test_contains_hz_class(self):
        r = compute_insolation(1.0, 1.0)
        s = format_insolation_result(r)
        assert r.hz_class in s
