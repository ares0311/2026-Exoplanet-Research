"""Tests for Skills/stellar_chromospheric_activity_age.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from stellar_chromospheric_activity_age import (
    compute_chromospheric_age,
    format_chromospheric_activity_result,
)


class TestComputeChromosphericAge:
    def test_ok_flag_solar_value(self) -> None:
        r = compute_chromospheric_age(-4.5)
        assert r.flag == "OK"

    def test_age_positive_for_solar(self) -> None:
        r = compute_chromospheric_age(-4.5)
        assert r.age_gyr > 0.0

    def test_solar_activity_moderate(self) -> None:
        r = compute_chromospheric_age(-4.75)
        assert r.activity_level in ("MODERATE", "QUIET")

    def test_active_star_younger(self) -> None:
        r_active = compute_chromospheric_age(-4.2)
        r_quiet = compute_chromospheric_age(-5.0)
        assert r_active.age_gyr < r_quiet.age_gyr

    def test_rv_jitter_positive(self) -> None:
        r = compute_chromospheric_age(-4.5)
        assert r.rv_jitter_ms > 0.0

    def test_active_star_higher_jitter(self) -> None:
        r_active = compute_chromospheric_age(-4.0)
        r_quiet = compute_chromospheric_age(-5.0)
        assert r_active.rv_jitter_ms > r_quiet.rv_jitter_ms

    def test_very_active_level(self) -> None:
        r = compute_chromospheric_age(-4.0)
        assert r.activity_level == "VERY_ACTIVE"

    def test_very_quiet_level(self) -> None:
        r = compute_chromospheric_age(-5.2)
        assert r.activity_level == "VERY_QUIET"

    def test_age_uncertainty_positive(self) -> None:
        r = compute_chromospheric_age(-4.5)
        assert r.age_uncertainty_gyr > 0.0

    def test_invalid_log_rhk_too_active(self) -> None:
        r = compute_chromospheric_age(-3.0)
        assert r.flag == "INVALID_LOG_RHK"

    def test_invalid_log_rhk_too_quiet(self) -> None:
        r = compute_chromospheric_age(-7.0)
        assert r.flag == "INVALID_LOG_RHK"

    def test_result_frozen(self) -> None:
        r = compute_chromospheric_age(-4.5)
        try:
            r.age_gyr = 0.0  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except (AttributeError, TypeError):
            pass

    def test_format_returns_string(self) -> None:
        r = compute_chromospheric_age(-4.5)
        s = format_chromospheric_activity_result(r)
        assert isinstance(s, str)
        assert r.flag in s
