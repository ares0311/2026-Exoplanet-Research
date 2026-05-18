"""Tests for trapezoid_box_comparator.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from trapezoid_box_comparator import (
    compare_trapezoid_box,
    format_trapezoid_box_result,
)


def _box_lc(n=200, period=5.0, depth_ppm=5000.0, duration_hours=2.0):
    """Pure box transit LC."""
    half_width = (duration_hours / 24.0) / period / 2.0
    depth = depth_ppm / 1e6
    time = [i * period / n for i in range(n)]
    flux = []
    for t in time:
        ph = (t % period) / period
        if ph >= 0.5:
            ph -= 1.0
        flux.append(1.0 - depth if abs(ph) <= half_width else 1.0)
    return time, flux


def _trapezoid_lc(n=200, period=5.0, depth_ppm=5000.0, duration_hours=2.0, ing_frac=0.3):
    """Trapezoid transit LC."""
    half_width = (duration_hours / 24.0) / period / 2.0
    depth = depth_ppm / 1e6
    ing = ing_frac * half_width
    time = [i * period / n for i in range(n)]
    flux = []
    for t in time:
        ph = (t % period) / period
        if ph >= 0.5:
            ph -= 1.0
        aph = abs(ph)
        if aph >= half_width:
            flux.append(1.0)
        elif aph <= half_width - ing:
            flux.append(1.0 - depth)
        else:
            slope = (aph - (half_width - ing)) / max(ing, 1e-15)
            flux.append(1.0 - depth * (1.0 - slope))
    return time, flux


class TestCompareTrapezoidBox:
    def test_basic_ok(self):
        time, flux = _box_lc()
        r = compare_trapezoid_box(time, flux, 5.0, 0.0, duration_hours=2.0, depth_ppm=5000.0)
        assert r.flag == "OK"

    def test_box_preferred_for_box_lc(self):
        time, flux = _box_lc(depth_ppm=5000.0)
        r = compare_trapezoid_box(time, flux, 5.0, 0.0, duration_hours=2.0, depth_ppm=5000.0)
        assert r.preferred_model in ("box", "indeterminate")

    def test_chi2_values_nonnegative(self):
        time, flux = _box_lc()
        r = compare_trapezoid_box(time, flux, 5.0, 0.0, duration_hours=2.0, depth_ppm=5000.0)
        assert r.chi2_box >= 0
        assert r.chi2_trapezoid >= 0

    def test_delta_chi2_computed(self):
        time, flux = _box_lc()
        r = compare_trapezoid_box(time, flux, 5.0, 0.0, duration_hours=2.0, depth_ppm=5000.0)
        assert abs(r.delta_chi2 - (r.chi2_box - r.chi2_trapezoid)) < 0.01

    def test_invalid_zero_period(self):
        time, flux = _box_lc()
        r = compare_trapezoid_box(time, flux, 0.0, 0.0)
        assert r.flag == "INVALID"

    def test_invalid_zero_depth(self):
        time, flux = _box_lc()
        r = compare_trapezoid_box(time, flux, 5.0, 0.0, depth_ppm=0.0)
        assert r.flag == "INVALID"

    def test_insufficient_few_points(self):
        r = compare_trapezoid_box([1.0, 2.0], [1.0, 1.0], 5.0, 0.0)
        assert r.flag in ("INSUFFICIENT", "INVALID")

    def test_best_ingress_fraction_in_grid(self):
        time, flux = _box_lc()
        fracs = [0.1, 0.3, 0.5]
        r = compare_trapezoid_box(time, flux, 5.0, 0.0, depth_ppm=5000.0,
                                  ingress_fractions=fracs)
        assert r.best_ingress_fraction in fracs

    def test_result_frozen(self):
        time, flux = _box_lc()
        r = compare_trapezoid_box(time, flux, 5.0, 0.0, depth_ppm=5000.0)
        try:
            r.preferred_model = "x"  # type: ignore[misc]
            raise AssertionError()
        except (AttributeError, TypeError):
            pass

    def test_preferred_model_values(self):
        time, flux = _box_lc()
        r = compare_trapezoid_box(time, flux, 5.0, 0.0, depth_ppm=5000.0)
        assert r.preferred_model in ("box", "trapezoid", "indeterminate")


class TestFormatTrapezoidBoxResult:
    def test_returns_string(self):
        time, flux = _box_lc()
        r = compare_trapezoid_box(time, flux, 5.0, 0.0, depth_ppm=5000.0)
        assert isinstance(format_trapezoid_box_result(r), str)

    def test_contains_flag(self):
        time, flux = _box_lc()
        r = compare_trapezoid_box(time, flux, 5.0, 0.0, depth_ppm=5000.0)
        s = format_trapezoid_box_result(r)
        assert r.flag in s

    def test_contains_preferred(self):
        time, flux = _box_lc()
        r = compare_trapezoid_box(time, flux, 5.0, 0.0, depth_ppm=5000.0)
        s = format_trapezoid_box_result(r)
        assert r.preferred_model in s
