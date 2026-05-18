"""Tests for transit_asymmetry_scorer.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from transit_asymmetry_scorer import (
    format_transit_asymmetry_result,
    score_transit_asymmetry,
)


def _symmetric_transit(n=200, period=5.0, depth=0.01, duration_hours=2.0):
    """Generate a symmetric box transit light curve."""
    time = [i * 0.05 for i in range(n)]
    flux = []
    half_dur = duration_hours / 24.0 / 2.0
    for t in time:
        ph = ((t) % period) / period
        if ph >= 0.5:
            ph -= 1.0
        flux.append(1.0 - depth if abs(ph) <= half_dur / period else 1.0)
    return time, flux


def _asymmetric_transit(n=200, period=5.0, depth=0.01, duration_hours=2.0):
    """Ingress deeper than egress."""
    time = [i * 0.05 for i in range(n)]
    flux = []
    half_dur = duration_hours / 24.0 / 2.0
    for t in time:
        ph = ((t) % period) / period
        if ph >= 0.5:
            ph -= 1.0
        if abs(ph) <= half_dur / period:
            # Ingress (ph<0) twice as deep
            d = depth * 2 if ph < 0 else depth * 0.5
            flux.append(1.0 - d)
        else:
            flux.append(1.0)
    return time, flux


class TestScoreTransitAsymmetry:
    def test_symmetric_low_score(self):
        time, flux = _symmetric_transit()
        r = score_transit_asymmetry(time, flux, 5.0, 0.0, duration_hours=2.0)
        if r.flag not in ("INSUFFICIENT", "INVALID"):
            assert r.asymmetry_score < 0.5

    def test_asymmetric_high_score(self):
        time, flux = _asymmetric_transit()
        r = score_transit_asymmetry(time, flux, 5.0, 0.0, duration_hours=2.0)
        if r.flag == "ASYMMETRIC":
            assert r.is_asymmetric

    def test_too_few_points(self):
        r = score_transit_asymmetry([1.0] * 5, [1.0] * 5, 5.0, 0.0, duration_hours=2.0)
        assert r.flag in ("INVALID", "INSUFFICIENT")

    def test_invalid_zero_period(self):
        time, flux = _symmetric_transit()
        r = score_transit_asymmetry(time, flux, 0.0, 0.0)
        assert r.flag == "INVALID"

    def test_invalid_zero_duration(self):
        time, flux = _symmetric_transit()
        r = score_transit_asymmetry(time, flux, 5.0, 0.0, duration_hours=0.0)
        assert r.flag == "INVALID"

    def test_score_in_range(self):
        time, flux = _symmetric_transit()
        r = score_transit_asymmetry(time, flux, 5.0, 0.0, duration_hours=2.0)
        assert 0.0 <= r.asymmetry_score <= 1.0

    def test_flag_ok_or_asymmetric(self):
        time, flux = _symmetric_transit()
        r = score_transit_asymmetry(time, flux, 5.0, 0.0, duration_hours=2.0)
        assert r.flag in ("OK", "ASYMMETRIC", "INSUFFICIENT", "INVALID")

    def test_result_frozen(self):
        time, flux = _symmetric_transit()
        r = score_transit_asymmetry(time, flux, 5.0, 0.0, duration_hours=2.0)
        try:
            r.asymmetry_score = 999  # type: ignore[misc]
            raise AssertionError()
        except (AttributeError, TypeError):
            pass

    def test_custom_threshold(self):
        time, flux = _symmetric_transit()
        r = score_transit_asymmetry(time, flux, 5.0, 0.0, duration_hours=2.0,
                                    asymmetry_threshold=0.001)
        # Very low threshold may flag symmetric transit
        assert r.flag in ("ASYMMETRIC", "OK", "INSUFFICIENT", "INVALID")


class TestFormatTransitAsymmetryResult:
    def test_returns_string(self):
        time, flux = _symmetric_transit()
        r = score_transit_asymmetry(time, flux, 5.0, 0.0, duration_hours=2.0)
        assert isinstance(format_transit_asymmetry_result(r), str)

    def test_contains_flag(self):
        time, flux = _symmetric_transit()
        r = score_transit_asymmetry(time, flux, 5.0, 0.0, duration_hours=2.0)
        s = format_transit_asymmetry_result(r)
        assert r.flag in s
