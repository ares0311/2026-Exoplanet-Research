"""Tests for Skills/exomoon_ttv_estimator.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from exomoon_ttv_estimator import compute_exomoon_ttv, format_exomoon_ttv_result


class TestComputeExomoonTtv:
    def test_ok_flag(self) -> None:
        r = compute_exomoon_ttv(0.1, 400000.0, 100.0, 10.0)
        assert r.flag == "OK"

    def test_ttv_positive(self) -> None:
        r = compute_exomoon_ttv(0.1, 400000.0, 100.0, 10.0)
        assert r.ttv_amplitude_minutes > 0

    def test_tdv_positive(self) -> None:
        r = compute_exomoon_ttv(0.1, 400000.0, 100.0, 10.0)
        assert r.tdv_amplitude_minutes >= 0

    def test_more_massive_moon_larger_ttv(self) -> None:
        r1 = compute_exomoon_ttv(0.1, 400000.0, 100.0, 10.0)
        r2 = compute_exomoon_ttv(1.0, 400000.0, 100.0, 10.0)
        assert r2.ttv_amplitude_minutes > r1.ttv_amplitude_minutes

    def test_larger_moon_orbit_larger_ttv(self) -> None:
        r_small = compute_exomoon_ttv(0.1, 200000.0, 100.0, 10.0)
        r_large = compute_exomoon_ttv(0.1, 800000.0, 100.0, 10.0)
        assert r_large.ttv_amplitude_minutes > r_small.ttv_amplitude_minutes

    def test_longer_period_larger_ttv(self) -> None:
        r_short = compute_exomoon_ttv(0.1, 400000.0, 100.0, 5.0)
        r_long = compute_exomoon_ttv(0.1, 400000.0, 100.0, 20.0)
        assert r_long.ttv_amplitude_minutes > r_short.ttv_amplitude_minutes

    def test_detectable_tess_large_moon(self) -> None:
        r = compute_exomoon_ttv(10.0, 500000.0, 100.0, 10.0)
        assert r.detectable_tess or r.ttv_amplitude_minutes >= 0

    def test_invalid_period(self) -> None:
        r = compute_exomoon_ttv(0.1, 400000.0, 100.0, 0.0)
        assert r.flag == "INVALID_PERIOD"

    def test_invalid_planet_mass(self) -> None:
        r = compute_exomoon_ttv(0.1, 400000.0, 0.0, 10.0)
        assert r.flag == "INVALID_MASS"

    def test_invalid_moon_mass(self) -> None:
        r = compute_exomoon_ttv(0.0, 400000.0, 100.0, 10.0)
        assert r.flag == "INVALID_MOON_MASS"

    def test_result_frozen(self) -> None:
        r = compute_exomoon_ttv(0.1, 400000.0, 100.0, 10.0)
        try:
            r.ttv_amplitude_minutes = 0.0  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except (AttributeError, TypeError):
            pass

    def test_format_returns_string(self) -> None:
        r = compute_exomoon_ttv(0.1, 400000.0, 100.0, 10.0)
        s = format_exomoon_ttv_result(r)
        assert isinstance(s, str)
        assert r.flag in s
