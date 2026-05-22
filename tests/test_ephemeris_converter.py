"""Tests for ephemeris_converter.py"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from ephemeris_converter import (
    convert_epoch,
    convert_time_array,
    format_conversion_result,
)

_BTJD_OFFSET = 2457000.0
_BKJD_OFFSET = 2454833.0


class TestConvertEpoch:
    def test_btjd_to_bjd(self):
        r = convert_epoch(1234.5, "BTJD", "BJD")
        assert r.flag == "OK"
        assert abs(r.output_value - (1234.5 + _BTJD_OFFSET)) < 1e-6

    def test_bjd_to_btjd(self):
        r = convert_epoch(2458000.0, "BJD", "BTJD")
        assert abs(r.output_value - (2458000.0 - _BTJD_OFFSET)) < 1e-6

    def test_bkjd_to_bjd(self):
        r = convert_epoch(500.0, "BKJD", "BJD")
        assert abs(r.output_value - (500.0 + _BKJD_OFFSET)) < 1e-6

    def test_btjd_to_bkjd(self):
        r = convert_epoch(1000.0, "BTJD", "BKJD")
        expected = (1000.0 + _BTJD_OFFSET) - _BKJD_OFFSET
        assert abs(r.output_value - expected) < 1e-5

    def test_identity_bjd_to_bjd(self):
        r = convert_epoch(2457500.0, "BJD", "BJD")
        assert r.output_value == 2457500.0

    def test_invalid_from_system(self):
        r = convert_epoch(1000.0, "INVALID", "BJD")
        assert r.flag == "INVALID"
        assert math.isnan(r.output_value)

    def test_invalid_to_system(self):
        r = convert_epoch(1000.0, "BJD", "BOGUS")
        assert r.flag == "INVALID"

    def test_uncertainty_preserved(self):
        r = convert_epoch(1234.5, "BTJD", "BJD", uncertainty=0.001)
        assert r.uncertainty_out == 0.001

    def test_case_insensitive(self):
        r = convert_epoch(1234.5, "btjd", "bjd")
        assert r.flag == "OK"

    def test_result_frozen(self):
        r = convert_epoch(1234.5, "BTJD", "BJD")
        try:
            r.output_value = 0.0  # type: ignore[misc]
            raise AssertionError()
        except (AttributeError, TypeError):
            pass


class TestConvertTimeArray:
    def test_basic_shift(self):
        times = [1000.0, 1001.0, 1002.0]
        out = convert_time_array(times, "BTJD", "BJD")
        assert len(out) == 3
        assert all(abs(out[i] - (times[i] + _BTJD_OFFSET)) < 1e-6 for i in range(3))

    def test_empty_array(self):
        out = convert_time_array([], "BTJD", "BJD")
        assert out == []

    def test_invalid_system_returns_empty(self):
        out = convert_time_array([1.0, 2.0], "BADTIME", "BJD")
        assert out == []


class TestFormatConversionResult:
    def test_returns_string(self):
        r = convert_epoch(1234.5, "BTJD", "BJD")
        assert isinstance(format_conversion_result(r), str)

    def test_contains_flag(self):
        r = convert_epoch(1234.5, "BTJD", "BJD")
        assert r.flag in format_conversion_result(r)
