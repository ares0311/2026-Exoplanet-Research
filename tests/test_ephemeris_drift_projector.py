"""Tests for Skills/ephemeris_drift_projector.py"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from ephemeris_drift_projector import (
    format_ephemeris_drift,
    project_ephemeris_drift,
)


def test_zero_cycles_at_epoch():
    """With reference_bjd = epoch, n_cycles = 0 → σ_T(0) = σ_epoch."""
    r = project_ephemeris_drift(3.0, 2460000.0, 0.001, 0.0001)
    assert r.flag == "OK"
    assert r.n_cycles == 0
    assert abs(r.sigma_tn_days - 0.001) < 1e-9


def test_sigma_grows_with_cycles():
    epoch = 2460000.0
    period = 3.0
    future_bjd = epoch + 100 * period  # 100 cycles
    r = project_ephemeris_drift(
        period, epoch, 0.001, 0.0001, reference_bjd=future_bjd
    )
    assert r.n_cycles == 100
    expected = math.sqrt(0.001 ** 2 + (100 ** 2) * (0.0001 ** 2))
    assert abs(r.sigma_tn_days - expected) < 1e-7


def test_exceeds_threshold():
    epoch = 2460000.0
    period = 3.0
    future_bjd = epoch + 5000 * period
    r = project_ephemeris_drift(
        period, epoch, 0.001, 0.001, reference_bjd=future_bjd,
        transit_duration_days=0.1,
    )
    assert r.exceeds_threshold is True
    assert r.flag == "EXCEEDS_THRESHOLD"


def test_ok_when_within_threshold():
    r = project_ephemeris_drift(
        10.0, 2460000.0, 0.0001, 0.00001,
        transit_duration_days=1.0,
    )
    assert r.flag == "OK"
    assert r.exceeds_threshold is False


def test_invalid_negative_period():
    r = project_ephemeris_drift(-1.0, 2460000.0, 0.001, 0.0001)
    assert r.flag == "INVALID"


def test_invalid_negative_sigma():
    r = project_ephemeris_drift(3.0, 2460000.0, -0.001, 0.0001)
    assert r.flag == "INVALID"


def test_invalid_nan_period():
    r = project_ephemeris_drift(float("nan"), 2460000.0, 0.001, 0.0001)
    assert r.flag == "INVALID"


def test_default_transit_duration_is_period_over_10():
    """Default transit_duration = period/10; threshold = 0.5 * period/10."""
    period = 10.0
    epoch = 2460000.0
    sigma_period = 0.0
    # σ_T(0) = σ_epoch = 0.0; threshold = 0.5 * 1.0 = 0.5 → should not exceed
    r = project_ephemeris_drift(period, epoch, 0.001, sigma_period)
    assert r.drift_threshold_days == pytest_approx(0.5, rel=1e-6)
    assert r.flag == "OK"


def pytest_approx(val, rel=1e-6):
    class _A:
        def __eq__(self, other):
            return abs(other - val) / abs(val + 1e-30) < rel
    return _A()


def test_next_transit_bjd():
    epoch = 2460000.0
    period = 5.0
    ref = epoch + 10 * period
    r = project_ephemeris_drift(period, epoch, 0.001, 0.0001, reference_bjd=ref)
    assert abs(r.next_transit_bjd - (epoch + 10 * period)) < 1e-5


def test_sigma_minutes_conversion():
    r = project_ephemeris_drift(3.0, 2460000.0, 1.0 / 1440.0, 0.0)
    assert abs(r.sigma_tn_minutes - 1.0) < 1e-6


def test_format_contains_keywords():
    r = project_ephemeris_drift(3.0, 2460000.0, 0.001, 0.0001)
    text = format_ephemeris_drift(r)
    assert "Ephemeris Drift" in text
    assert "σ_T(n)" in text
    assert "OK" in text


def test_format_exceeds_threshold():
    epoch = 2460000.0
    r = project_ephemeris_drift(
        3.0, epoch, 0.001, 0.001,
        reference_bjd=epoch + 5000 * 3.0,
        transit_duration_days=0.1,
    )
    text = format_ephemeris_drift(r)
    assert "EXCEEDS_THRESHOLD" in text
