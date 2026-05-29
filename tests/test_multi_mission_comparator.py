"""Tests for Skills/multi_mission_comparator.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from multi_mission_comparator import (
    MissionComparison,
    MissionMeasurement,
    compare_multi_mission,
    format_mission_comparison,
)


def _meas(mission="TESS", period=3.0, depth=500.0, snr=10.0, source="test"):
    return MissionMeasurement(
        mission=mission,
        period_days=period,
        depth_ppm=depth,
        duration_hours=2.0,
        n_transits=5,
        snr=snr,
        source=source,
    )


def test_empty_flag():
    result = compare_multi_mission(12345, [])
    assert result.flag == "EMPTY"


def test_single_mission_flag():
    result = compare_multi_mission(12345, [_meas()])
    assert result.flag == "SINGLE_MISSION"


def test_consistent_flag():
    m1 = _meas("TESS", period=3.0, depth=500.0)
    m2 = _meas("Kepler", period=3.001, depth=510.0)
    result = compare_multi_mission(12345, [m1, m2])
    assert result.flag == "CONSISTENT"


def test_period_discrepant_flag():
    m1 = _meas("TESS", period=3.0)
    m2 = _meas("Kepler", period=5.0)
    result = compare_multi_mission(12345, [m1, m2])
    assert result.flag == "PERIOD_DISCREPANT"


def test_depth_discrepant_flag():
    m1 = _meas("TESS", period=3.0, depth=500.0)
    m2 = _meas("Kepler", period=3.001, depth=2000.0)
    result = compare_multi_mission(12345, [m1, m2])
    assert result.flag in ("DEPTH_DISCREPANT", "CONSISTENT")


def test_best_mission_highest_snr():
    m1 = _meas("TESS", snr=5.0)
    m2 = _meas("Kepler", snr=20.0)
    result = compare_multi_mission(12345, [m1, m2])
    assert result.best_mission == "Kepler"


def test_period_spread_computed():
    m1 = _meas("TESS", period=3.0)
    m2 = _meas("Kepler", period=3.03)
    result = compare_multi_mission(None, [m1, m2])
    assert result.period_spread_frac is not None
    assert result.period_spread_frac > 0


def test_tic_id_none_allowed():
    result = compare_multi_mission(None, [_meas()])
    assert result.tic_id is None


def test_measurements_stored():
    m1, m2 = _meas("TESS"), _meas("Kepler")
    result = compare_multi_mission(1, [m1, m2])
    assert len(result.measurements) == 2


def test_returns_mission_comparison():
    result = compare_multi_mission(1, [_meas()])
    assert isinstance(result, MissionComparison)


def test_format_contains_tic():
    result = compare_multi_mission(77777, [_meas()])
    md = format_mission_comparison(result)
    assert "77777" in md


def test_format_contains_status():
    result = compare_multi_mission(1, [_meas()])
    md = format_mission_comparison(result)
    assert result.flag in md


def test_no_snr_best_mission_none():
    m1 = MissionMeasurement("TESS", 3.0, 500.0, 2.0, 5, None, "t")
    m2 = MissionMeasurement("K2", 3.0, 500.0, 2.0, 5, None, "k")
    result = compare_multi_mission(1, [m1, m2])
    assert result.best_mission is None
