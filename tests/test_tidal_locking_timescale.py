"""Tests for Skills/tidal_locking_timescale.py."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from tidal_locking_timescale import (
    TidalLockingResult,
    compute_tidal_locking_timescale,
    format_tidal_locking_result,
)


class TestComputeTidalLockingTimescale:
    def test_returns_result_type(self):
        r = compute_tidal_locking_timescale(period_days=3.0)
        assert isinstance(r, TidalLockingResult)

    def test_hot_jupiter_likely_locked(self):
        # Very short period → should be tidally locked
        r = compute_tidal_locking_timescale(period_days=1.5, planet_mass_mjup=1.0,
                                             stellar_mass_msun=1.0, planet_radius_rjup=1.0)
        assert r.is_likely_locked
        assert r.flag == "OK"

    def test_wide_orbit_not_locked(self):
        # Long period → should NOT be locked
        r = compute_tidal_locking_timescale(period_days=365.0, planet_mass_mjup=1.0,
                                             stellar_mass_msun=1.0, stellar_age_gyr=5.0)
        assert not r.is_likely_locked

    def test_sync_timescale_positive(self):
        r = compute_tidal_locking_timescale(period_days=3.0)
        assert r.sync_timescale_gyr > 0.0

    def test_lock_ratio_finite(self):
        r = compute_tidal_locking_timescale(period_days=3.0)
        assert math.isfinite(r.lock_ratio)

    def test_dominant_channel_string(self):
        r = compute_tidal_locking_timescale(period_days=3.0)
        assert r.dominant_channel in ("PLANETARY", "STELLAR")

    def test_shorter_period_faster_locking(self):
        r1 = compute_tidal_locking_timescale(period_days=1.0)
        r2 = compute_tidal_locking_timescale(period_days=10.0)
        assert r1.sync_timescale_gyr < r2.sync_timescale_gyr

    def test_invalid_period(self):
        r = compute_tidal_locking_timescale(period_days=0.0)
        assert r.flag != "OK"
        assert math.isnan(r.sync_timescale_gyr)

    def test_negative_period(self):
        r = compute_tidal_locking_timescale(period_days=-1.0)
        assert r.flag != "OK"

    def test_larger_planet_locks_faster(self):
        r1 = compute_tidal_locking_timescale(period_days=5.0, planet_radius_rjup=2.0)
        r2 = compute_tidal_locking_timescale(period_days=5.0, planet_radius_rjup=0.5)
        assert r1.sync_timescale_gyr < r2.sync_timescale_gyr

    def test_frozen_dataclass(self):
        r = compute_tidal_locking_timescale(period_days=3.0)
        try:
            r.sync_timescale_gyr = 0.0  # type: ignore[misc]
            raise AssertionError("frozen dataclass should not allow setattr")
        except (AttributeError, TypeError):
            pass

    def test_flag_ok(self):
        r = compute_tidal_locking_timescale(period_days=3.0)
        assert r.flag == "OK"


class TestFormatTidalLockingResult:
    def test_ok_returns_table(self):
        r = compute_tidal_locking_timescale(period_days=3.0)
        out = format_tidal_locking_result(r)
        assert "Sync timescale" in out
        assert "|" in out

    def test_invalid_returns_flag(self):
        r = compute_tidal_locking_timescale(period_days=0.0)
        out = format_tidal_locking_result(r)
        assert "flag=" in out
