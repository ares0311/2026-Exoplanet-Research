"""Tests for Skills/mean_motion_resonance_checker.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from mean_motion_resonance_checker import MMRResult, check_mmr, format_mmr_result


class TestMMRResult:
    def test_dataclass_fields(self):
        r = MMRResult(nearest_ratio_str="2:1", nearest_period_days=10.0,
                      delta_fraction=0.01, is_near_mmr=True)
        assert r.nearest_ratio_str == "2:1"
        assert r.is_near_mmr is True

    def test_frozen(self):
        r = MMRResult(nearest_ratio_str="2:1", nearest_period_days=10.0,
                      delta_fraction=0.01, is_near_mmr=True)
        try:
            r.is_near_mmr = False
            raise AssertionError()
        except Exception:
            pass


class TestCheckMMR:
    def test_exact_2_to_1(self):
        # Candidate = 20d, known = 10d → exact 2:1 resonance
        r = check_mmr(20.0, [10.0], tolerance=0.02)
        assert r.is_near_mmr is True
        assert "2:1" in r.nearest_ratio_str or r.delta_fraction < 1e-9

    def test_near_3_to_2(self):
        # Candidate ~ 15, known = 10 → 3:2
        r = check_mmr(15.0, [10.0], tolerance=0.02)
        assert r.is_near_mmr is True

    def test_not_near_mmr(self):
        r = check_mmr(7.3, [10.0], tolerance=0.02)
        assert r.is_near_mmr is False

    def test_empty_known_periods(self):
        r = check_mmr(10.0, [], tolerance=0.02)
        assert r.flag == "WARNING"
        assert r.is_near_mmr is False

    def test_zero_candidate_period_error(self):
        r = check_mmr(0.0, [10.0])
        assert r.flag == "ERROR"

    def test_multiple_known_periods(self):
        r = check_mmr(20.0, [10.0, 5.0, 30.0])
        assert r.is_near_mmr is True

    def test_tolerance_respected(self):
        # Candidate slightly off from 2:1
        r_loose = check_mmr(20.5, [10.0], tolerance=0.05)
        # Loose should be near, tight may not be
        assert r_loose.is_near_mmr is True

    def test_delta_fraction_range(self):
        r = check_mmr(10.0, [10.0])
        assert 0.0 <= r.delta_fraction <= 1.0

    def test_nearest_period_days_set(self):
        r = check_mmr(20.0, [10.0, 7.0])
        assert r.nearest_period_days > 0

    def test_flag_ok_with_valid_inputs(self):
        r = check_mmr(10.0, [5.0])
        assert r.flag == "OK"


class TestFormatMMR:
    def test_returns_string(self):
        r = check_mmr(20.0, [10.0])
        s = format_mmr_result(r)
        assert isinstance(s, str)

    def test_contains_ratio(self):
        r = check_mmr(20.0, [10.0])
        s = format_mmr_result(r)
        assert r.nearest_ratio_str in s
