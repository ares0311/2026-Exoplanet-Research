"""Tests for Skills/atmospheric_escape_rate_estimator.py"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from atmospheric_escape_rate_estimator import (
    estimate_escape_rate,
    format_escape_rate_result,
)


class TestAtmosphericEscapeRateEstimator:
    def test_basic_ok(self) -> None:
        r = estimate_escape_rate(2.0, 5.0, 0.1)
        assert r.flag == "OK"
        assert r.xuv_flux_wm2 > 0
        assert r.escape_rate_kg_s > 0

    def test_escape_rate_positive(self) -> None:
        r = estimate_escape_rate(2.0, 5.0, 0.05)
        assert r.escape_rate_kg_s > 0
        assert math.isfinite(r.escape_rate_kg_s)

    def test_closer_orbit_higher_flux(self) -> None:
        r_close = estimate_escape_rate(2.0, 5.0, 0.05)
        r_far = estimate_escape_rate(2.0, 5.0, 1.0)
        assert r_close.xuv_flux_wm2 > r_far.xuv_flux_wm2

    def test_closer_orbit_faster_escape(self) -> None:
        r_close = estimate_escape_rate(2.0, 5.0, 0.05)
        r_far = estimate_escape_rate(2.0, 5.0, 1.0)
        assert r_close.escape_rate_kg_s > r_far.escape_rate_kg_s

    def test_escape_class_catastrophic_close(self) -> None:
        r = estimate_escape_rate(3.0, 2.0, 0.01, stellar_xuv_luminosity_lsun=1e-2)
        assert r.escape_class in ("RAPID", "CATASTROPHIC")

    def test_escape_class_safe_far(self) -> None:
        r = estimate_escape_rate(1.0, 10.0, 5.0, stellar_xuv_luminosity_lsun=1e-6)
        assert r.escape_class == "SAFE"

    def test_mass_loss_timescale_positive(self) -> None:
        r = estimate_escape_rate(2.0, 5.0, 0.1)
        assert r.mass_loss_timescale_gyr > 0

    def test_invalid_radius(self) -> None:
        r = estimate_escape_rate(0.0, 5.0, 0.1)
        assert r.flag == "INVALID_RADIUS"
        assert math.isnan(r.escape_rate_kg_s)

    def test_invalid_mass(self) -> None:
        r = estimate_escape_rate(2.0, 0.0, 0.1)
        assert r.flag == "INVALID_MASS"

    def test_invalid_distance(self) -> None:
        r = estimate_escape_rate(2.0, 5.0, 0.0)
        assert r.flag == "INVALID_DISTANCE"

    def test_result_is_frozen(self) -> None:
        r = estimate_escape_rate(2.0, 5.0, 0.1)
        try:
            r.escape_class = "X"  # type: ignore[misc]
            raise AssertionError("frozen dataclass should not allow setattr")
        except Exception:
            pass

    def test_format_ok(self) -> None:
        r = estimate_escape_rate(2.0, 5.0, 0.1)
        s = format_escape_rate_result(r)
        assert "escape" in s.lower() or "Escape" in s
        assert r.escape_class in s

    def test_format_error(self) -> None:
        r = estimate_escape_rate(0.0, 5.0, 0.1)
        s = format_escape_rate_result(r)
        assert "INVALID_RADIUS" in s
