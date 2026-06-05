"""Tests for Skills/binary_eclipse_timing_model.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from binary_eclipse_timing_model import (
    compute_binary_eclipse_timing,
    format_binary_eclipse_timing_result,
)


class TestComputeBinaryEclipseTiming:
    def test_ok_flag(self) -> None:
        r = compute_binary_eclipse_timing(2458000.0, 1.0, 5)
        assert r.flag == "OK"

    def test_n_eclipse_times_correct(self) -> None:
        r = compute_binary_eclipse_timing(2458000.0, 1.0, 10)
        assert len(r.eclipse_times) == 10

    def test_period_preserved(self) -> None:
        r = compute_binary_eclipse_timing(2458000.0, 2.5, 5)
        assert r.period_days == 2.5

    def test_zero_roemer_without_tertiary(self) -> None:
        r = compute_binary_eclipse_timing(2458000.0, 1.0, 5, third_body_mass_mjup=0.0)
        assert r.romer_semi_amplitude_seconds == 0.0

    def test_positive_roemer_with_tertiary(self) -> None:
        r = compute_binary_eclipse_timing(
            2458000.0, 1.0, 5,
            third_body_mass_mjup=300.0, third_body_period_yr=5.0
        )
        assert r.romer_semi_amplitude_seconds > 0.0

    def test_epoch_in_first_eclipse(self) -> None:
        epoch = 2458000.0
        r = compute_binary_eclipse_timing(epoch, 1.0, 5)
        assert abs(r.eclipse_times[0].predicted_time_bjd - epoch) < 1e-3

    def test_ltt_zero_without_tertiary(self) -> None:
        r = compute_binary_eclipse_timing(2458000.0, 1.0, 5, third_body_mass_mjup=0.0)
        for e in r.eclipse_times:
            assert e.ltt_offset_seconds == 0.0

    def test_ltt_nonzero_with_tertiary(self) -> None:
        r = compute_binary_eclipse_timing(
            2458000.0, 1.0, 20,
            third_body_mass_mjup=1000.0, third_body_period_yr=2.0
        )
        assert any(abs(e.ltt_offset_seconds) > 0.0 for e in r.eclipse_times)

    def test_eclipse_spacing_is_period(self) -> None:
        period = 3.7
        r = compute_binary_eclipse_timing(2458000.0, period, 5, third_body_mass_mjup=0.0)
        times = [e.predicted_time_bjd for e in r.eclipse_times]
        for i in range(1, len(times)):
            assert abs(times[i] - times[i - 1] - period) < 1e-6

    def test_invalid_period(self) -> None:
        r = compute_binary_eclipse_timing(2458000.0, 0.0, 5)
        assert r.flag == "INVALID_PERIOD"

    def test_invalid_n_eclipses(self) -> None:
        r = compute_binary_eclipse_timing(2458000.0, 1.0, 0)
        assert r.flag == "INVALID_N_ECLIPSES"

    def test_result_frozen(self) -> None:
        r = compute_binary_eclipse_timing(2458000.0, 1.0, 5)
        try:
            r.period_days = 0.0  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except (AttributeError, TypeError):
            pass

    def test_format_returns_string(self) -> None:
        r = compute_binary_eclipse_timing(2458000.0, 1.0, 5)
        s = format_binary_eclipse_timing_result(r)
        assert isinstance(s, str)
        assert r.flag in s
