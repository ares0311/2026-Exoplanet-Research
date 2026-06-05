"""Tests for Skills/spectroscopic_binary_checker.py."""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from spectroscopic_binary_checker import (
    SpectroscopicBinaryResult,
    check_spectroscopic_binary,
    format_spectroscopic_binary_result,
)


class TestCheckSpectroscopicBinary:
    def test_returns_result_type(self):
        r = check_spectroscopic_binary(rv_semiamplitude_ms=5000.0, period_days=3.0)
        assert isinstance(r, SpectroscopicBinaryResult)

    def test_flag_ok(self):
        r = check_spectroscopic_binary(rv_semiamplitude_ms=5000.0, period_days=3.0)
        assert r.flag == "OK"

    def test_mass_function_positive(self):
        r = check_spectroscopic_binary(rv_semiamplitude_ms=5000.0, period_days=3.0)
        assert r.mass_function_msun > 0.0

    def test_min_companion_mass_positive(self):
        r = check_spectroscopic_binary(rv_semiamplitude_ms=5000.0, period_days=3.0)
        assert r.min_companion_mass_msun > 0.0

    def test_sb_class_string(self):
        r = check_spectroscopic_binary(rv_semiamplitude_ms=5000.0, period_days=3.0)
        assert r.sb_class in ("SB1", "SB2_POSSIBLE", "NOT_SB", "AMBIGUOUS")

    def test_large_rv_implies_sb1_or_sb2(self):
        # K = 30 km/s at 3-day period → massive companion
        r = check_spectroscopic_binary(rv_semiamplitude_ms=30000.0, period_days=3.0)
        assert r.sb_class in ("SB1", "SB2_POSSIBLE")

    def test_small_rv_not_sb(self):
        # K = 5 m/s → sub-stellar companion
        r = check_spectroscopic_binary(rv_semiamplitude_ms=5.0, period_days=3.0)
        assert r.sb_class == "NOT_SB"

    def test_secondary_lines_detected_implies_sb2(self):
        r = check_spectroscopic_binary(rv_semiamplitude_ms=1000.0, period_days=3.0,
                                        secondary_lines_detected=True)
        assert r.sb_class == "SB2_POSSIBLE"

    def test_mass_ratio_positive(self):
        r = check_spectroscopic_binary(rv_semiamplitude_ms=5000.0, period_days=3.0)
        assert r.stellar_mass_ratio_q >= 0.0

    def test_invalid_rv(self):
        r = check_spectroscopic_binary(rv_semiamplitude_ms=0.0, period_days=3.0)
        assert r.flag != "OK"
        assert math.isnan(r.mass_function_msun)

    def test_invalid_period(self):
        r = check_spectroscopic_binary(rv_semiamplitude_ms=5000.0, period_days=0.0)
        assert r.flag != "OK"

    def test_eccentricity_effect(self):
        r0 = check_spectroscopic_binary(rv_semiamplitude_ms=5000.0, period_days=3.0,
                                         eccentricity=0.0)
        re = check_spectroscopic_binary(rv_semiamplitude_ms=5000.0, period_days=3.0,
                                         eccentricity=0.5)
        # Non-zero eccentricity changes (1-e²)^(3/2) factor
        assert r0.mass_function_msun != pytest.approx(re.mass_function_msun)

    def test_rv_stored(self):
        r = check_spectroscopic_binary(rv_semiamplitude_ms=5000.0, period_days=3.0)
        assert r.rv_semiamplitude_ms == pytest.approx(5000.0)

    def test_frozen_dataclass(self):
        r = check_spectroscopic_binary(rv_semiamplitude_ms=5000.0, period_days=3.0)
        try:
            r.sb_class = "SB1"  # type: ignore[misc]
            raise AssertionError("frozen dataclass should not allow setattr")
        except (AttributeError, TypeError):
            pass


class TestFormatSpectroscopicBinaryResult:
    def test_ok_returns_table(self):
        r = check_spectroscopic_binary(rv_semiamplitude_ms=5000.0, period_days=3.0)
        out = format_spectroscopic_binary_result(r)
        assert "Mass function" in out
        assert "|" in out

    def test_invalid_returns_flag(self):
        r = check_spectroscopic_binary(rv_semiamplitude_ms=0.0, period_days=3.0)
        out = format_spectroscopic_binary_result(r)
        assert "flag=" in out
