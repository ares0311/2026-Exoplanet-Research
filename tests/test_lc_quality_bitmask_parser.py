"""Tests for lc_quality_bitmask_parser.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from lc_quality_bitmask_parser import (
    format_bitmask_result,
    get_clean_mask,
    parse_quality_bitmask,
)


class TestParseQualityBitmask:
    def test_invalid_empty(self):
        r = parse_quality_bitmask([])
        assert r.flag == "INVALID"

    def test_all_clean(self):
        r = parse_quality_bitmask([0, 0, 0, 0])
        assert r.flag == "ALL_CLEAN"
        assert r.n_clean == 4
        assert r.n_flagged == 0

    def test_safe_mode_bit1_flagged(self):
        # Bit 1 = SafeMode, which is "bad"
        r = parse_quality_bitmask([0b10, 0])
        assert r.n_flagged == 1

    def test_flagged_fraction_correct(self):
        r = parse_quality_bitmask([0b10, 0b10, 0, 0])  # 2 of 4 flagged
        assert r.flagged_fraction == 0.5

    def test_n_cadences_correct(self):
        r = parse_quality_bitmask([0] * 50)
        assert r.n_cadences == 50

    def test_bit_counts_populated(self):
        # Bit 1 = SafeMode
        r = parse_quality_bitmask([0b10])
        assert "SafeMode" in r.bit_counts
        assert r.bit_counts["SafeMode"] == 1

    def test_zero_quality_clean(self):
        r = parse_quality_bitmask([0])
        assert r.n_clean == 1

    def test_result_frozen(self):
        r = parse_quality_bitmask([0])
        try:
            r.n_cadences = 99  # type: ignore[misc]
            raise AssertionError()
        except (AttributeError, TypeError):
            pass


class TestGetCleanMask:
    def test_all_zero_all_true(self):
        mask = get_clean_mask([0, 0, 0])
        assert all(mask)

    def test_bad_bit_excluded(self):
        # Bit 1 = SafeMode (bad) → should be False
        mask = get_clean_mask([0b10, 0])
        assert mask[0] is False
        assert mask[1] is True

    def test_length_preserved(self):
        q = [0, 0b10, 0, 0b10, 0]
        mask = get_clean_mask(q)
        assert len(mask) == len(q)


class TestFormatBitmaskResult:
    def test_returns_string(self):
        r = parse_quality_bitmask([0, 1, 2])
        assert isinstance(format_bitmask_result(r), str)

    def test_contains_flag(self):
        r = parse_quality_bitmask([0])
        assert r.flag in format_bitmask_result(r)
