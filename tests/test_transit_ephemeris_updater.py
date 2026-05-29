"""Tests for Skills/transit_ephemeris_updater.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))


from transit_ephemeris_updater import (
    EphemerisUpdate,
    format_ephemeris_update,
    update_ephemeris,
)


def _exact_midpoints(period=3.0, epoch=1000.0, n=5):
    """Midpoints exactly on the predicted ephemeris."""
    return [epoch + i * period for i in range(n)]


def _shifted_midpoints(period=3.0, epoch=1000.0, n=5, shift_days=0.01):
    """Midpoints with a constant epoch offset."""
    return [epoch + shift_days + i * period for i in range(n)]


def test_exact_midpoints_low_residuals():
    mpts = _exact_midpoints()
    result = update_ephemeris(mpts, period_days=3.0, epoch_btjd=1000.0)
    assert result.rms_oc_min < 0.01


def test_invalid_period():
    result = update_ephemeris([1.0, 2.0], period_days=0.0, epoch_btjd=0.0)
    assert result.flag == "INVALID"


def test_sparse_single_midpoint():
    result = update_ephemeris([1000.0], period_days=3.0, epoch_btjd=1000.0)
    assert result.flag == "SPARSE"


def test_returns_ephemeris_update():
    mpts = _exact_midpoints()
    result = update_ephemeris(mpts, period_days=3.0, epoch_btjd=1000.0)
    assert isinstance(result, EphemerisUpdate)


def test_epoch_correction_for_shift():
    shift = 0.02
    mpts = _shifted_midpoints(shift_days=shift)
    result = update_ephemeris(mpts, period_days=3.0, epoch_btjd=1000.0)
    assert abs(result.epoch_correction - shift) < 0.001


def test_period_preserved_for_exact():
    mpts = _exact_midpoints(period=3.14159)
    result = update_ephemeris(mpts, period_days=3.14159, epoch_btjd=1000.0)
    assert abs(result.fitted_period - 3.14159) < 1e-5


def test_n_transits_count():
    mpts = _exact_midpoints(n=7)
    result = update_ephemeris(mpts, period_days=3.0, epoch_btjd=1000.0)
    assert result.n_transits == 7


def test_original_period_preserved():
    mpts = _exact_midpoints(period=4.0)
    result = update_ephemeris(mpts, period_days=4.0, epoch_btjd=1000.0)
    assert result.original_period == 4.0


def test_improved_flag_for_offset():
    mpts = _shifted_midpoints(shift_days=0.05)
    result = update_ephemeris(mpts, period_days=3.0, epoch_btjd=1000.0)
    assert result.flag in ("IMPROVED", "OK")


def test_format_contains_status():
    mpts = _exact_midpoints()
    result = update_ephemeris(mpts, period_days=3.0, epoch_btjd=1000.0)
    md = format_ephemeris_update(result)
    assert result.flag in md


def test_format_contains_period():
    mpts = _exact_midpoints()
    result = update_ephemeris(mpts, period_days=3.0, epoch_btjd=1000.0)
    md = format_ephemeris_update(result)
    assert "3.0" in md or "3." in md


def test_rms_oc_nonnegative():
    mpts = _exact_midpoints()
    result = update_ephemeris(mpts, period_days=3.0, epoch_btjd=1000.0)
    assert result.rms_oc_min >= 0
