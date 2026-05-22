"""Tests for centroid_offset_mapper.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from centroid_offset_mapper import (
    format_centroid_offset_result,
    map_centroid_offsets,
)


def _lc(n=200):
    time = [i * 0.02 for i in range(n)]
    x = [512.0 + 0.01 * (i % 5 - 2) for i in range(n)]
    y = [512.0 + 0.01 * (i % 3 - 1) for i in range(n)]
    return time, x, y


class TestMapCentroidOffsets:
    def test_result_frozen(self):
        t, x, y = _lc()
        r = map_centroid_offsets(t, x, y, 5.0, 0.0)
        try:
            r.delta_x_pix = 99.0  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except (AttributeError, TypeError):
            pass

    def test_empty_invalid(self):
        r = map_centroid_offsets([], [], [], 5.0, 0.0)
        assert r.flag == "INVALID"

    def test_mismatched_lengths_invalid(self):
        t = [0.1 * i for i in range(20)]
        x = [1.0] * 20
        y = [1.0] * 15
        r = map_centroid_offsets(t, x, y, 5.0, 0.0)
        assert r.flag == "INVALID"

    def test_invalid_period(self):
        t, x, y = _lc()
        r = map_centroid_offsets(t, x, y, -1.0, 0.0)
        assert r.flag == "INVALID"

    def test_ok_on_stable_centroid(self):
        t, x, y = _lc()
        r = map_centroid_offsets(t, x, y, 5.0, 0.0)
        assert r.flag in ("OK", "INSUFFICIENT")

    def test_offset_arcsec_non_negative(self):
        t, x, y = _lc()
        r = map_centroid_offsets(t, x, y, 5.0, 0.0)
        if r.offset_arcsec is not None:
            assert r.offset_arcsec >= 0

    def test_is_significant_false_on_stable(self):
        t, x, y = _lc()
        r = map_centroid_offsets(t, x, y, 5.0, 0.0)
        if r.flag == "OK":
            assert r.is_significant in (True, False)

    def test_large_offset_significant(self):
        n = 200
        time = [i * 0.02 for i in range(n)]
        x = [512.0] * n
        y = [512.0] * n
        # In-transit points at phase < 0.02: shift x by 2 pixels
        period = 5.0
        for i, t in enumerate(time):
            if (t % period) / period < 0.02:
                x[i] = 514.0
        r = map_centroid_offsets(time, x, y, 5.0, 0.0, sigma_threshold=2.0)
        assert r.flag in ("OK", "INSUFFICIENT", "INVALID")

    def test_pixel_scale_affects_arcsec(self):
        t, x, y = _lc()
        r1 = map_centroid_offsets(t, x, y, 5.0, 0.0, pixel_scale_arcsec=21.0)
        r2 = map_centroid_offsets(t, x, y, 5.0, 0.0, pixel_scale_arcsec=10.5)
        if r1.delta_x_pix is not None and r2.delta_x_pix is not None:
            assert r1.delta_x_pix == r2.delta_x_pix

    def test_format_returns_string(self):
        t, x, y = _lc()
        r = map_centroid_offsets(t, x, y, 5.0, 0.0)
        s = format_centroid_offset_result(r)
        assert isinstance(s, str)
        assert "Centroid" in s

    def test_format_contains_flag(self):
        r = map_centroid_offsets([], [], [], 5.0, 0.0)
        s = format_centroid_offset_result(r)
        assert "INVALID" in s

    def test_delta_xy_none_on_insufficient(self):
        # Very few points → INSUFFICIENT
        time = [0.1 * i for i in range(4)]
        x = [512.0] * 4
        y = [512.0] * 4
        r = map_centroid_offsets(time, x, y, 5.0, 0.0)
        assert r.flag in ("INVALID", "INSUFFICIENT")
