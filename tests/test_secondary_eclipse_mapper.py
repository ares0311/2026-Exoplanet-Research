"""Tests for Skills.secondary_eclipse_mapper."""
from __future__ import annotations

from Skills.secondary_eclipse_mapper import (
    SecondaryEclipseResult,
    format_secondary_eclipse_result,
    map_secondary_eclipse,
)


def _make_lc_with_secondary(period=5.0, epoch=2458000.0,
                              secondary_depth=0.003, n_points=2000):
    """LC covering more than one period; secondary eclipse only at phase 0.5."""
    dt = 2.0 / 1440.0
    # Start at epoch so phase starts at 0 and OOT region (ph<0.1) is populated
    time = [epoch + i * dt for i in range(n_points)]
    flux = []
    for t in time:
        ph = ((t - epoch) % period) / period
        if 0.45 < ph < 0.55:
            flux.append(1.0 - secondary_depth)
        else:
            flux.append(1.0)
    return time, flux


class TestMapSecondaryEclipse:
    def test_returns_result(self) -> None:
        t, f = _make_lc_with_secondary()
        r = map_secondary_eclipse(t, f, 5.0, 2458000.0)
        assert isinstance(r, SecondaryEclipseResult)

    def test_empty_returns_insufficient(self) -> None:
        r = map_secondary_eclipse([], [], 5.0, 2458000.0)
        assert r.flag == "INSUFFICIENT"

    def test_zero_period_returns_insufficient(self) -> None:
        t, f = _make_lc_with_secondary()
        r = map_secondary_eclipse(t, f, 0.0, 2458000.0)
        assert r.flag == "INSUFFICIENT"

    def test_secondary_detected(self) -> None:
        t, f = _make_lc_with_secondary(secondary_depth=0.005)
        r = map_secondary_eclipse(
            t, f, 5.0, 2458000.0, duration_days=0.3, detection_snr_threshold=2.0
        )
        assert r.is_detected

    def test_no_secondary_not_detected(self) -> None:
        t, f = _make_lc_with_secondary(secondary_depth=0.0)
        r = map_secondary_eclipse(t, f, 5.0, 2458000.0)
        assert r.flag in {"NOT_DETECTED", "INSUFFICIENT"}

    def test_snr_nonnegative(self) -> None:
        t, f = _make_lc_with_secondary()
        r = map_secondary_eclipse(t, f, 5.0, 2458000.0)
        if r.snr is not None:
            assert r.snr >= 0

    def test_phase_in_range(self) -> None:
        t, f = _make_lc_with_secondary()
        r = map_secondary_eclipse(t, f, 5.0, 2458000.0)
        if r.phase is not None:
            assert 0.4 <= r.phase <= 0.6

    def test_depth_positive_when_detected(self) -> None:
        t, f = _make_lc_with_secondary(secondary_depth=0.005)
        r = map_secondary_eclipse(
            t, f, 5.0, 2458000.0, duration_days=0.3, detection_snr_threshold=2.0
        )
        if r.is_detected:
            assert r.depth_ppm is not None and r.depth_ppm > 0

    def test_flag_values_valid(self) -> None:
        t, f = _make_lc_with_secondary()
        r = map_secondary_eclipse(t, f, 5.0, 2458000.0)
        assert r.flag in {"DETECTED", "NOT_DETECTED", "INSUFFICIENT"}


class TestFormatSecondaryEclipse:
    def test_returns_string(self) -> None:
        t, f = _make_lc_with_secondary()
        r = map_secondary_eclipse(t, f, 5.0, 2458000.0)
        assert isinstance(format_secondary_eclipse_result(r), str)

    def test_insufficient_handled(self) -> None:
        r = map_secondary_eclipse([], [], 5.0, 2458000.0)
        out = format_secondary_eclipse_result(r)
        assert "Insufficient" in out or "insufficient" in out.lower()
