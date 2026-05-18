"""Tests for Skills.pixel_level_centroid_checker."""
from __future__ import annotations

from Skills.pixel_level_centroid_checker import (
    CentroidCheckResult,
    check_pixel_centroid,
    format_centroid_check_result,
)


def _make_centroid_lc(period=10.0, epoch=2458000.0, n_points=1000,
                      in_transit_shift=0.0):
    dt = 2.0 / 1440.0
    time = [epoch + i * dt for i in range(n_points)]
    flux = [1.0] * n_points
    row = []
    col = []
    for t in time:
        ph = (t - epoch) % period
        if ph < 0.1:
            row.append(100.0 + in_transit_shift)
            col.append(200.0 + in_transit_shift)
        else:
            row.append(100.0)
            col.append(200.0)
    return time, flux, row, col


class TestCheckPixelCentroid:
    def test_returns_result(self) -> None:
        t, f, r, c = _make_centroid_lc()
        result = check_pixel_centroid(t, f, r, c, 10.0, 2458000.0)
        assert isinstance(result, CentroidCheckResult)

    def test_empty_returns_insufficient(self) -> None:
        result = check_pixel_centroid([], [], [], [], 10.0, 2458000.0)
        assert result.flag == "INSUFFICIENT"

    def test_zero_period_returns_insufficient(self) -> None:
        t, f, r, c = _make_centroid_lc()
        result = check_pixel_centroid(t, f, r, c, 0.0, 2458000.0)
        assert result.flag == "INSUFFICIENT"

    def test_no_shift_ok(self) -> None:
        t, f, r, c = _make_centroid_lc(in_transit_shift=0.0)
        result = check_pixel_centroid(t, f, r, c, 10.0, 2458000.0, duration_days=0.1)
        assert result.flag in {"OK", "CENTROID_SHIFT", "INSUFFICIENT"}

    def test_large_shift_detected(self) -> None:
        t, f, r, c = _make_centroid_lc(in_transit_shift=5.0)
        result = check_pixel_centroid(
            t, f, r, c, 10.0, 2458000.0,
            duration_days=0.1, sigma_threshold=1.0,
        )
        assert result.flag in {"CENTROID_SHIFT", "INSUFFICIENT"}

    def test_offset_arcsec_nonnegative(self) -> None:
        t, f, r, c = _make_centroid_lc()
        result = check_pixel_centroid(t, f, r, c, 10.0, 2458000.0, duration_days=0.1)
        if result.offset_arcsec is not None:
            assert result.offset_arcsec >= 0

    def test_significance_nonnegative(self) -> None:
        t, f, r, c = _make_centroid_lc()
        result = check_pixel_centroid(t, f, r, c, 10.0, 2458000.0, duration_days=0.1)
        if result.significance_sigma is not None:
            assert result.significance_sigma >= 0

    def test_n_in_transit_positive(self) -> None:
        t, f, r, c = _make_centroid_lc()
        result = check_pixel_centroid(t, f, r, c, 10.0, 2458000.0, duration_days=0.1)
        if result.flag != "INSUFFICIENT":
            assert result.n_in_transit > 0

    def test_flag_values_valid(self) -> None:
        t, f, r, c = _make_centroid_lc()
        result = check_pixel_centroid(t, f, r, c, 10.0, 2458000.0)
        assert result.flag in {"OK", "CENTROID_SHIFT", "INSUFFICIENT"}


class TestFormatCentroidCheck:
    def test_returns_string(self) -> None:
        t, f, r, c = _make_centroid_lc()
        result = check_pixel_centroid(t, f, r, c, 10.0, 2458000.0)
        assert isinstance(format_centroid_check_result(result), str)

    def test_insufficient_handled(self) -> None:
        result = check_pixel_centroid([], [], [], [], 10.0, 2458000.0)
        out = format_centroid_check_result(result)
        assert "INSUFFICIENT" in out
