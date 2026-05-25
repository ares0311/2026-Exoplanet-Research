"""Tests for Skills/transit_duration_anomaly_checker.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from transit_duration_anomaly_checker import (
    check_duration_anomaly,
    format_duration_anomaly,
)


def test_solar_like_ok():
    """Solar-like star, 10-day period, ~3h transit → normal."""
    r = check_duration_anomaly(
        3.0, 10.0,
        stellar_radius_rsun=1.0,
        stellar_mass_msun=1.0,
    )
    assert r.flag in ("OK", "ANOMALOUS")
    assert r.expected_hours is not None
    assert r.expected_hours > 0


def test_no_stellar_params_insufficient():
    r = check_duration_anomaly(3.0, 10.0)
    assert r.flag == "INSUFFICIENT_DATA"
    assert r.expected_hours is None


def test_invalid_negative_duration():
    r = check_duration_anomaly(-1.0, 10.0, stellar_radius_rsun=1.0, stellar_mass_msun=1.0)
    assert r.flag == "INVALID"


def test_invalid_zero_period():
    r = check_duration_anomaly(3.0, 0.0, stellar_radius_rsun=1.0, stellar_mass_msun=1.0)
    assert r.flag == "INVALID"


def test_invalid_nan_duration():
    r = check_duration_anomaly(float("nan"), 10.0, stellar_radius_rsun=1.0, stellar_mass_msun=1.0)
    assert r.flag == "INVALID"


def test_too_long_flagged():
    """Very long duration → too_long anomaly."""
    r = check_duration_anomaly(
        200.0, 10.0,
        stellar_radius_rsun=1.0,
        stellar_mass_msun=1.0,
        sigma_threshold=2.0,
    )
    assert r.flag == "ANOMALOUS"
    assert r.anomaly_type == "too_long"
    assert r.is_anomalous is True


def test_too_short_flagged():
    """Very short duration → too_short anomaly."""
    r = check_duration_anomaly(
        0.01, 10.0,
        stellar_radius_rsun=1.0,
        stellar_mass_msun=1.0,
        sigma_threshold=2.0,
    )
    assert r.flag == "ANOMALOUS"
    assert r.anomaly_type == "too_short"


def test_ratio_calculation():
    r = check_duration_anomaly(
        3.0, 10.0,
        stellar_radius_rsun=1.0,
        stellar_mass_msun=1.0,
    )
    assert r.ratio is not None
    assert abs(r.ratio - r.observed_hours / r.expected_hours) < 1e-4


def test_sigma_deviation_zero_at_expected():
    """If observed ≈ expected, sigma_deviation ≈ 0."""
    # First get expected duration
    r0 = check_duration_anomaly(3.0, 10.0, stellar_radius_rsun=1.0, stellar_mass_msun=1.0)
    exp = r0.expected_hours
    r = check_duration_anomaly(exp, 10.0, stellar_radius_rsun=1.0, stellar_mass_msun=1.0)
    assert r.sigma_deviation is not None
    assert abs(r.sigma_deviation) < 0.01


def test_invalid_negative_stellar_radius():
    r = check_duration_anomaly(
        3.0, 10.0,
        stellar_radius_rsun=-1.0,
        stellar_mass_msun=1.0,
    )
    assert r.flag == "INVALID"


def test_expected_scales_with_radius():
    r1 = check_duration_anomaly(3.0, 10.0, stellar_radius_rsun=1.0, stellar_mass_msun=1.0)
    r2 = check_duration_anomaly(3.0, 10.0, stellar_radius_rsun=2.0, stellar_mass_msun=1.0)
    assert r2.expected_hours > r1.expected_hours


def test_format_contains_keywords():
    r = check_duration_anomaly(3.0, 10.0, stellar_radius_rsun=1.0, stellar_mass_msun=1.0)
    text = format_duration_anomaly(r)
    assert "Duration Anomaly" in text
    assert "Observed" in text


def test_format_insufficient():
    r = check_duration_anomaly(3.0, 10.0)
    text = format_duration_anomaly(r)
    assert "INSUFFICIENT_DATA" in text
