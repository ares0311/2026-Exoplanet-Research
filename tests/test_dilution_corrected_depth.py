"""Tests for Skills/dilution_corrected_depth.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from dilution_corrected_depth import DilutionCorrectionResult, correct_for_dilution


class TestDilutionCorrectedDepth:
    def test_no_dilution(self) -> None:
        r = correct_for_dilution(1000.0, 0.0)
        assert r.flag == "OK"
        assert abs(r.corrected_depth_ppm - 1000.0) < 1e-4

    def test_50pct_dilution(self) -> None:
        r = correct_for_dilution(500.0, 0.5)
        assert r.flag == "OK"
        assert abs(r.corrected_depth_ppm - 1000.0) < 1e-4

    def test_invalid_depth_negative(self) -> None:
        r = correct_for_dilution(-100.0, 0.3)
        assert r.flag == "INVALID_DEPTH"

    def test_invalid_dilution_negative(self) -> None:
        r = correct_for_dilution(1000.0, -0.1)
        assert r.flag == "INVALID_DILUTION"

    def test_invalid_dilution_one(self) -> None:
        r = correct_for_dilution(1000.0, 1.0)
        assert r.flag == "INVALID_DILUTION"

    def test_invalid_dilution_gt_one(self) -> None:
        r = correct_for_dilution(1000.0, 1.5)
        assert r.flag == "INVALID_DILUTION"

    def test_depth_ratio_correct(self) -> None:
        r = correct_for_dilution(1000.0, 0.5)
        assert abs(r.depth_ratio - 2.0) < 1e-4

    def test_error_propagation(self) -> None:
        r = correct_for_dilution(1000.0, 0.5, depth_err_ppm=50.0)
        assert r.depth_err_ppm is not None
        assert abs(r.depth_err_ppm - 100.0) < 1e-3

    def test_no_error_when_not_provided(self) -> None:
        r = correct_for_dilution(1000.0, 0.3)
        assert r.depth_err_ppm is None

    def test_result_frozen(self) -> None:
        r = correct_for_dilution(1000.0, 0.3)
        assert isinstance(r, DilutionCorrectionResult)
        try:
            object.__setattr__(r, "flag", "x")
            raise AssertionError()
        except Exception:
            pass

    def test_format_output(self) -> None:
        from dilution_corrected_depth import format_dilution_result
        r = correct_for_dilution(1000.0, 0.3)
        s = format_dilution_result(r)
        assert "|" in s

    def test_small_dilution(self) -> None:
        r = correct_for_dilution(1000.0, 0.01)
        assert r.flag == "OK"
        assert r.corrected_depth_ppm > 1000.0
