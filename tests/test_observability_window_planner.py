"""Tests for Skills/observability_window_planner.py."""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from observability_window_planner import (
    format_observability_windows,
    plan_observability,
    plan_observability_windows,
)


class TestPlanObservability:
    def test_equatorial_target_observable(self) -> None:
        r = plan_observability(ra_deg=0.0, dec_deg=0.0, site_lat_deg=0.0)
        assert r.flag == "OK"
        assert r.observable

    def test_max_elevation_correct(self) -> None:
        r = plan_observability(ra_deg=0.0, dec_deg=30.0, site_lat_deg=30.0)
        assert abs(r.max_elevation_deg - 90.0) < 1.0

    def test_target_not_observable_high_dec_low_lat(self) -> None:
        # Target at dec=70, site at -30: max elevation = 90 - 100 = -10 → not observable
        # abs(lat)+abs(dec) = 30+70 = 100 ≥ 90 → circumpolar, use different coords
        # Target at dec=60, site at -20: max el = 90 - 80 = 10, HA small → few hours
        # Better: very mismatched (dec=-60, lat=+20 → max el = 90-80=10, not circumpolar)
        r = plan_observability(ra_deg=0.0, dec_deg=-60.0, site_lat_deg=20.0)
        # max_alt = 90 - |20-(-60)| = 90-80 = 10 deg → below airmass=2 limit
        assert r.hours_above_airmass == 0.0 or not r.observable

    def test_hours_above_airmass_nonneg(self) -> None:
        r = plan_observability(ra_deg=0.0, dec_deg=10.0, site_lat_deg=20.0)
        assert r.hours_above_airmass >= 0.0

    def test_transit_ut_hour_stored(self) -> None:
        r = plan_observability(
            ra_deg=0.0, dec_deg=20.0, site_lat_deg=30.0, transit_ut_hour=23.5
        )
        assert abs(r.best_ut_hour - 23.5) < 0.01

    def test_no_transit_ut_is_nan(self) -> None:
        r = plan_observability(ra_deg=0.0, dec_deg=20.0, site_lat_deg=30.0)
        assert not math.isfinite(r.best_ut_hour)

    def test_invalid_ra(self) -> None:
        r = plan_observability(ra_deg=float("nan"), dec_deg=0.0, site_lat_deg=0.0)
        assert r.flag == "INVALID_RA_DEG"

    def test_invalid_dec(self) -> None:
        r = plan_observability(ra_deg=0.0, dec_deg=float("nan"), site_lat_deg=0.0)
        assert r.flag == "INVALID_DEC_DEG"

    def test_invalid_lat(self) -> None:
        r = plan_observability(ra_deg=0.0, dec_deg=0.0, site_lat_deg=float("inf"))
        assert r.flag == "INVALID_SITE_LAT_DEG"

    def test_tic_id_stored(self) -> None:
        r = plan_observability(ra_deg=0.0, dec_deg=20.0, site_lat_deg=30.0, tic_id="TIC123")
        assert r.tic_id == "TIC123"

    def test_batch_planner(self) -> None:
        targets = [
            {"ra_deg": 10.0, "dec_deg": 20.0, "tic_id": "TIC1"},
            {"ra_deg": 50.0, "dec_deg": -10.0, "tic_id": "TIC2"},
        ]
        windows = plan_observability_windows(targets, site_lat_deg=30.0)
        assert len(windows) == 2

    def test_format_output(self) -> None:
        targets = [{"ra_deg": 0.0, "dec_deg": 15.0, "tic_id": "TIC42"}]
        windows = plan_observability_windows(targets, site_lat_deg=20.0)
        s = format_observability_windows(windows)
        assert "|" in s
        assert "TIC42" in s
