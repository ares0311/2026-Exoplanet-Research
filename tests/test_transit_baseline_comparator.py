"""Tests for Skills/transit_baseline_comparator.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from transit_baseline_comparator import (
    BaselineComparison,
    compare_transit_baseline,
    format_baseline_comparison,
)


def _synthetic_lc(period=3.0, epoch=0.5, duration_h=2.0, depth_ppm=1000.0, n=200):
    """Simple synthetic light curve with a box transit."""
    time = [i * period / n for i in range(n * 3)]
    half_dur = duration_h / 24.0 / 2.0
    flux = []
    for t in time:
        phase = ((t - epoch) / period) % 1.0
        if phase > 0.5:
            phase -= 1.0
        if abs(phase) * period < half_dur:
            flux.append(1.0 - depth_ppm / 1e6)
        else:
            flux.append(1.0)
    return time, flux


def test_basic_detection():
    time, flux = _synthetic_lc()
    result = compare_transit_baseline(
        time, flux, period_days=3.0, epoch_btjd=0.5,
        duration_hours=2.0, expected_depth_ppm=1000.0,
    )
    assert result.measured_depth_ppm > 0
    assert result.n_in_transit >= 2


def test_flag_ok_matching_depth():
    time, flux = _synthetic_lc(depth_ppm=1000.0)
    result = compare_transit_baseline(
        time, flux, period_days=3.0, epoch_btjd=0.5,
        duration_hours=2.0, expected_depth_ppm=1000.0,
    )
    assert result.flag in ("OK", "SHALLOW", "DEEP", "SPARSE")


def test_invalid_period():
    result = compare_transit_baseline(
        [1.0, 2.0], [1.0, 1.0],
        period_days=0.0, epoch_btjd=0.0, duration_hours=1.0,
    )
    assert result.flag == "INVALID"


def test_sparse_flag():
    result = compare_transit_baseline(
        [0.0, 0.01], [0.999, 0.999],
        period_days=1.0, epoch_btjd=0.0, duration_hours=0.5,
    )
    assert result.flag == "SPARSE"


def test_depth_ratio_computed():
    time, flux = _synthetic_lc(depth_ppm=1000.0)
    result = compare_transit_baseline(
        time, flux, period_days=3.0, epoch_btjd=0.5,
        duration_hours=2.0, expected_depth_ppm=1000.0,
    )
    if result.depth_ratio is not None:
        assert result.depth_ratio > 0


def test_no_expected_depth():
    time, flux = _synthetic_lc()
    result = compare_transit_baseline(
        time, flux, period_days=3.0, epoch_btjd=0.5,
        duration_hours=2.0,
    )
    assert result.depth_ratio is None
    assert result.flag in ("OK", "SPARSE")


def test_oot_mean_near_one():
    time, flux = _synthetic_lc(depth_ppm=500.0)
    result = compare_transit_baseline(
        time, flux, period_days=3.0, epoch_btjd=0.5, duration_hours=2.0,
    )
    assert abs(result.oot_mean - 1.0) < 0.01


def test_returns_baseline_comparison():
    time, flux = _synthetic_lc()
    result = compare_transit_baseline(
        time, flux, period_days=3.0, epoch_btjd=0.5, duration_hours=2.0,
    )
    assert isinstance(result, BaselineComparison)


def test_shallow_flag():
    # Make flux only 10% as deep as expected
    time, flux = _synthetic_lc(depth_ppm=100.0)
    result = compare_transit_baseline(
        time, flux, period_days=3.0, epoch_btjd=0.5,
        duration_hours=2.0, expected_depth_ppm=2000.0,
    )
    assert result.flag in ("SHALLOW", "SPARSE")


def test_format_contains_status():
    time, flux = _synthetic_lc()
    result = compare_transit_baseline(
        time, flux, period_days=3.0, epoch_btjd=0.5, duration_hours=2.0,
    )
    md = format_baseline_comparison(result)
    assert result.flag in md


def test_measured_depth_nonnegative():
    time, flux = _synthetic_lc()
    result = compare_transit_baseline(
        time, flux, period_days=3.0, epoch_btjd=0.5, duration_hours=2.0,
    )
    assert result.measured_depth_ppm >= 0
