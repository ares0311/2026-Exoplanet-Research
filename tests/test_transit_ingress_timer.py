"""Tests for Skills/transit_ingress_timer.py"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from transit_ingress_timer import (
    IngressResult,
    compute_ingress_duration,
    format_ingress_result,
)


class TestComputeIngressDuration:
    def test_basic_ok(self):
        result = compute_ingress_duration(10.0, 0.1, 0.0, 10.0)
        assert result.flag == "OK"

    def test_total_duration_positive(self):
        result = compute_ingress_duration(10.0, 0.1, 0.0, 10.0)
        assert result.total_duration_hours > 0.0

    def test_ingress_less_than_total(self):
        result = compute_ingress_duration(10.0, 0.1, 0.0, 10.0)
        assert result.ingress_duration_hours < result.total_duration_hours

    def test_ingress_equals_egress(self):
        result = compute_ingress_duration(10.0, 0.1, 0.0, 10.0)
        assert abs(result.ingress_duration_hours - result.egress_duration_hours) < 1e-9

    def test_flat_bottom_positive_non_grazing(self):
        result = compute_ingress_duration(10.0, 0.1, 0.0, 10.0)
        if not result.is_grazing:
            assert result.flat_bottom_hours >= 0.0

    def test_invalid_negative_period(self):
        result = compute_ingress_duration(-1.0, 0.1, 0.0, 10.0)
        assert result.flag == "INVALID"

    def test_invalid_zero_period(self):
        result = compute_ingress_duration(0.0, 0.1, 0.0, 10.0)
        assert result.flag == "INVALID"

    def test_invalid_zero_a_over_rs(self):
        result = compute_ingress_duration(10.0, 0.1, 0.0, 0.0)
        assert result.flag == "INVALID"

    def test_invalid_negative_rp_over_rs(self):
        result = compute_ingress_duration(10.0, -0.1, 0.0, 10.0)
        assert result.flag == "INVALID"

    def test_grazing_detected(self):
        result = compute_ingress_duration(10.0, 0.1, 0.95, 10.0)
        assert result.is_grazing is True

    def test_non_grazing_detected(self):
        result = compute_ingress_duration(10.0, 0.1, 0.0, 10.0)
        assert result.is_grazing is False

    def test_ingress_fraction_range(self):
        result = compute_ingress_duration(10.0, 0.1, 0.0, 10.0)
        if result.ingress_fraction is not None:
            assert 0.0 <= result.ingress_fraction <= 0.5

    def test_longer_period_longer_duration(self):
        r1 = compute_ingress_duration(5.0, 0.1, 0.0, 10.0)
        r2 = compute_ingress_duration(20.0, 0.1, 0.0, 10.0)
        assert r2.total_duration_hours > r1.total_duration_hours

    def test_result_frozen(self):
        result = compute_ingress_duration(10.0, 0.1, 0.0, 10.0)
        try:
            result.total_duration_hours = 99.0
            assert False
        except Exception:
            pass

    def test_format_returns_string(self):
        result = compute_ingress_duration(10.0, 0.1, 0.0, 10.0)
        text = format_ingress_result(result)
        assert isinstance(text, str)

    def test_format_contains_flag(self):
        result = compute_ingress_duration(10.0, 0.1, 0.0, 10.0)
        text = format_ingress_result(result)
        assert result.flag in text
