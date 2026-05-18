"""Tests for tsm_calculator.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from tsm_calculator import compute_tsm, format_tsm_result


class TestComputeTSM:
    def test_basic_ok(self):
        result = compute_tsm(2.0, 500.0, 1.0, 10.0, planet_mass_mearth=5.0)
        assert result.flag == "OK"
        assert result.tsm > 0

    def test_esm_positive(self):
        result = compute_tsm(2.0, 500.0, 1.0, 10.0, planet_mass_mearth=5.0,
                             depth_ppm=500.0, teff_star_k=5000.0)
        assert result.esm >= 0

    def test_size_class_terrestrial(self):
        result = compute_tsm(1.0, 300.0, 0.8, 12.0, planet_mass_mearth=1.0)
        assert result.size_class == "terrestrial"
        assert result.scale_factor == pytest_approx(0.190, abs=0.01)

    def test_size_class_super_earth(self):
        result = compute_tsm(1.8, 500.0, 1.0, 11.0, planet_mass_mearth=3.0)
        assert result.size_class == "super-Earth"

    def test_size_class_sub_neptune(self):
        result = compute_tsm(3.5, 600.0, 1.0, 10.0, planet_mass_mearth=10.0)
        assert result.size_class == "sub-Neptune"

    def test_size_class_giant(self):
        result = compute_tsm(12.0, 1200.0, 1.2, 9.0, planet_mass_mearth=300.0)
        assert result.size_class == "giant"

    def test_mass_from_mr_relation(self):
        result = compute_tsm(2.0, 500.0, 1.0, 10.0)
        assert result.flag == "OK"
        assert result.planet_mass_mearth > 0

    def test_invalid_zero_radius(self):
        result = compute_tsm(0.0, 500.0, 1.0, 10.0)
        assert result.flag == "INVALID"

    def test_invalid_negative_teq(self):
        result = compute_tsm(2.0, -100.0, 1.0, 10.0)
        assert result.flag == "INVALID"

    def test_invalid_zero_stellar_radius(self):
        result = compute_tsm(2.0, 500.0, 0.0, 10.0)
        assert result.flag == "INVALID"

    def test_result_frozen(self):
        result = compute_tsm(2.0, 500.0, 1.0, 10.0)
        try:
            result.tsm = 999  # type: ignore[misc]
            raise AssertionError()
        except (AttributeError, TypeError):
            pass


def pytest_approx(value, **kwargs):
    """Simple approximate comparison helper."""
    import pytest
    return pytest.approx(value, **kwargs)


class TestFormatTSMResult:
    def _make(self):
        return compute_tsm(2.0, 500.0, 1.0, 10.0, planet_mass_mearth=5.0)

    def test_returns_string(self):
        assert isinstance(format_tsm_result(self._make()), str)

    def test_contains_tsm(self):
        r = self._make()
        s = format_tsm_result(r)
        assert "TSM" in s

    def test_contains_flag(self):
        r = self._make()
        s = format_tsm_result(r)
        assert r.flag in s
