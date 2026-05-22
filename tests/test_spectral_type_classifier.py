"""Tests for Skills/spectral_type_classifier.py."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from spectral_type_classifier import (
    SpectralTypeResult,
    classify_spectral_type,
    format_spectral_type,
)


class TestClassifySpectralType:
    def test_returns_spectral_type_result(self):
        result = classify_spectral_type(5778.0)
        assert isinstance(result, SpectralTypeResult)

    def test_flag_ok_for_valid_teff(self):
        result = classify_spectral_type(5778.0)
        assert result.flag == "OK"

    def test_none_teff_unknown_flag(self):
        result = classify_spectral_type(None)
        assert result.flag == "UNKNOWN_TEFF"

    def test_zero_teff_returns_invalid(self):
        result = classify_spectral_type(0.0)
        assert result.flag == "INVALID"

    def test_negative_teff_returns_invalid(self):
        result = classify_spectral_type(-100.0)
        assert result.flag == "INVALID"

    def test_sun_like_is_g_type(self):
        result = classify_spectral_type(5778.0)
        assert result.spectral_type == "G"

    def test_hot_star_is_o_type(self):
        result = classify_spectral_type(35000.0)
        assert result.spectral_type == "O"

    def test_cool_star_is_m_type(self):
        result = classify_spectral_type(3000.0)
        assert result.spectral_type == "M"

    def test_a_type_star(self):
        result = classify_spectral_type(9000.0)
        assert result.spectral_type == "A"

    def test_k_type_star(self):
        result = classify_spectral_type(4500.0)
        assert result.spectral_type == "K"

    def test_logg_high_is_dwarf(self):
        result = classify_spectral_type(5778.0, logg=4.4)
        assert result.is_dwarf
        assert result.luminosity_class == "V"

    def test_logg_low_is_giant(self):
        result = classify_spectral_type(5778.0, logg=2.0)
        assert result.is_giant
        assert result.luminosity_class in ("III", "I")

    def test_logg_none_unknown_luminosity(self):
        result = classify_spectral_type(5778.0, logg=None)
        assert result.luminosity_class == "UNKNOWN"

    def test_type_string_format(self):
        result = classify_spectral_type(5778.0, logg=4.4)
        assert result.type_string.startswith("G")
        assert "V" in result.type_string

    def test_frozen_dataclass(self):
        result = classify_spectral_type(5778.0)
        with pytest.raises((AttributeError, TypeError)):
            result.flag = "CHANGED"  # type: ignore[misc]


class TestFormatSpectralType:
    def test_returns_string(self):
        result = classify_spectral_type(5778.0)
        md = format_spectral_type(result)
        assert isinstance(md, str)

    def test_contains_flag(self):
        result = classify_spectral_type(5778.0)
        md = format_spectral_type(result)
        assert result.flag in md

    def test_unknown_teff_format(self):
        result = classify_spectral_type(None)
        md = format_spectral_type(result)
        assert "UNKNOWN_TEFF" in md

    def test_invalid_teff_format(self):
        result = classify_spectral_type(0.0)
        md = format_spectral_type(result)
        assert "INVALID" in md
