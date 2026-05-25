"""Tests for Skills/planet_radius_gap_classifier.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from planet_radius_gap_classifier import (
    classify_radius_gap,
    format_radius_gap,
)


def test_rocky():
    r = classify_radius_gap(1.2)
    assert r.flag == "OK"
    assert r.category == "rocky"


def test_gap():
    r = classify_radius_gap(1.7)
    assert r.flag == "OK"
    assert r.category == "gap"


def test_sub_neptune():
    r = classify_radius_gap(2.5)
    assert r.flag == "OK"
    assert r.category == "sub_neptune"


def test_neptune():
    r = classify_radius_gap(4.0)
    assert r.flag == "OK"
    assert r.category == "neptune"


def test_giant():
    r = classify_radius_gap(10.0)
    assert r.flag == "OK"
    assert r.category == "giant"


def test_boundary_rocky_gap():
    """1.5 R_Earth is the start of the gap, not rocky."""
    r = classify_radius_gap(1.5)
    assert r.category == "gap"


def test_boundary_gap_sub_neptune():
    r = classify_radius_gap(2.0)
    assert r.category == "sub_neptune"


def test_boundary_sub_neptune_neptune():
    r = classify_radius_gap(3.5)
    assert r.category == "neptune"


def test_boundary_neptune_giant():
    r = classify_radius_gap(6.0)
    assert r.category == "giant"


def test_invalid_zero_radius():
    r = classify_radius_gap(0.0)
    assert r.flag == "INVALID"
    assert r.category == "unknown"


def test_invalid_negative_radius():
    r = classify_radius_gap(-1.5)
    assert r.flag == "INVALID"


def test_invalid_nan():
    r = classify_radius_gap(float("nan"))
    assert r.flag == "INVALID"


def test_period_stored():
    r = classify_radius_gap(1.0, period_days=5.2)
    assert r.period_days == 5.2


def test_format_contains_keywords():
    r = classify_radius_gap(2.5, period_days=10.0)
    text = format_radius_gap(r)
    assert "Radius Gap" in text
    assert "sub_neptune" in text
    assert "OK" in text


def test_format_invalid():
    r = classify_radius_gap(-1.0)
    text = format_radius_gap(r)
    assert "INVALID" in text
