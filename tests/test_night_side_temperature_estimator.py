"""Tests for Skills/night_side_temperature_estimator.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from night_side_temperature_estimator import (
    compute_night_side_temperature,
    format_night_side_temperature_result,
)


class TestComputeNightSideTemperature:
    def test_ok_flag(self) -> None:
        r = compute_night_side_temperature(1500.0, 500.0, 1.0, 1.0)
        assert r.flag == "OK"

    def test_day_side_warmer_than_equilibrium(self) -> None:
        r = compute_night_side_temperature(1500.0, 500.0, 1.0, 1.0)
        assert r.day_side_temp_k >= 1500.0

    def test_night_side_cooler_than_day(self) -> None:
        r = compute_night_side_temperature(1500.0, 500.0, 1.0, 1.0)
        assert r.night_side_temp_k <= r.day_side_temp_k

    def test_contrast_positive(self) -> None:
        r = compute_night_side_temperature(1500.0, 500.0, 1.0, 1.0)
        assert r.day_night_contrast_k >= 0.0

    def test_efficiency_between_zero_and_one(self) -> None:
        r = compute_night_side_temperature(1500.0, 500.0, 1.0, 1.0)
        assert 0.0 <= r.heat_redistribution_efficiency <= 1.0

    def test_zero_amplitude_efficient_circulation(self) -> None:
        r = compute_night_side_temperature(1500.0, 0.0, 1.0, 1.0)
        assert r.circulation_class == "EFFICIENT"

    def test_large_amplitude_poor_circulation(self) -> None:
        r = compute_night_side_temperature(1500.0, 10000.0, 1.0, 1.0)
        assert r.circulation_class in ("POOR", "VERY_POOR")

    def test_flux_ratio_ge_one(self) -> None:
        r = compute_night_side_temperature(1500.0, 500.0, 1.0, 1.0)
        assert r.day_night_flux_ratio >= 1.0

    def test_invalid_teq(self) -> None:
        r = compute_night_side_temperature(0.0, 500.0, 1.0, 1.0)
        assert r.flag == "INVALID_TEQ"

    def test_invalid_amplitude(self) -> None:
        r = compute_night_side_temperature(1500.0, -1.0, 1.0, 1.0)
        assert r.flag == "INVALID_AMPLITUDE"

    def test_invalid_radii(self) -> None:
        r = compute_night_side_temperature(1500.0, 500.0, 0.0, 1.0)
        assert r.flag == "INVALID_RADII"

    def test_result_frozen(self) -> None:
        r = compute_night_side_temperature(1500.0, 500.0, 1.0, 1.0)
        try:
            r.night_side_temp_k = 0.0  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except (AttributeError, TypeError):
            pass

    def test_format_returns_string(self) -> None:
        r = compute_night_side_temperature(1500.0, 500.0, 1.0, 1.0)
        s = format_night_side_temperature_result(r)
        assert isinstance(s, str)
        assert r.flag in s
