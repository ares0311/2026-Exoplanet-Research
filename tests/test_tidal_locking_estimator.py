"""Tests for Skills/tidal_locking_estimator.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from tidal_locking_estimator import (
    TidalLockResult,
    estimate_tidal_locking,
    format_tidal_lock_result,
)


class TestTidalLockResult:
    def test_dataclass_fields(self):
        r = TidalLockResult(t_lock_yr=1e9, is_likely_locked=False, flag="OK")
        assert r.t_lock_yr == 1e9
        assert r.is_likely_locked is False

    def test_frozen(self):
        r = TidalLockResult(t_lock_yr=1e9, is_likely_locked=False)
        try:
            r.t_lock_yr = 0
            raise AssertionError("Should be frozen")
        except Exception:
            pass


class TestEstimateTidalLocking:
    def test_close_planet_locked(self):
        # Very close-in planet should be locked quickly
        r = estimate_tidal_locking(0.05, 1.0, 1.0)
        assert r.is_likely_locked is True
        assert r.flag == "OK"

    def test_earth_not_locked(self):
        # Earth at 1 AU should not be locked
        r = estimate_tidal_locking(1.0, 1.0, 1.0)
        assert r.is_likely_locked is False

    def test_zero_a_returns_error(self):
        r = estimate_tidal_locking(0.0, 1.0, 1.0)
        assert r.flag == "ERROR"

    def test_zero_mass_returns_error(self):
        r = estimate_tidal_locking(1.0, 0.0, 1.0)
        assert r.flag == "ERROR"

    def test_zero_radius_returns_error(self):
        r = estimate_tidal_locking(1.0, 1.0, 0.0)
        assert r.flag == "ERROR"

    def test_period_scales_with_a6(self):
        r1 = estimate_tidal_locking(1.0, 1.0, 1.0)
        r2 = estimate_tidal_locking(2.0, 1.0, 1.0)
        ratio = r2.t_lock_yr / r1.t_lock_yr
        assert abs(ratio - 64.0) < 1.0  # 2^6 = 64

    def test_q_factor_scales_linearly(self):
        r1 = estimate_tidal_locking(0.1, 1.0, 1.0, q_factor=100.0)
        r2 = estimate_tidal_locking(0.1, 1.0, 1.0, q_factor=200.0)
        assert abs(r2.t_lock_yr / r1.t_lock_yr - 2.0) < 0.01

    def test_positive_t_lock(self):
        r = estimate_tidal_locking(0.5, 5.0, 2.0)
        assert r.t_lock_yr > 0

    def test_hot_jupiter_locked(self):
        # Hot Jupiter at 0.05 AU with large mass - should still lock quickly
        r = estimate_tidal_locking(0.04, 318.0, 11.2, m_star_msun=1.0)
        assert r.t_lock_yr > 0


class TestFormatTidalLock:
    def test_returns_string(self):
        r = estimate_tidal_locking(0.1, 1.0, 1.0)
        s = format_tidal_lock_result(r)
        assert isinstance(s, str)

    def test_contains_lock_status(self):
        r = estimate_tidal_locking(0.05, 1.0, 1.0)
        s = format_tidal_lock_result(r)
        assert "Yes" in s or "No" in s

    def test_contains_timescale(self):
        r = estimate_tidal_locking(1.0, 1.0, 1.0)
        s = format_tidal_lock_result(r)
        assert "yr" in s
