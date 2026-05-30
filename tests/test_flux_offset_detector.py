"""Tests for Skills/flux_offset_detector.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from flux_offset_detector import (
    detect_flux_offset,
    format_flux_offset_result,
)


class TestDetectFluxOffset:
    def test_same_flux_no_offset(self) -> None:
        r = detect_flux_offset([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        assert r.offset_detected is False
        assert r.flag == "OK"

    def test_one_sector_offset(self) -> None:
        r = detect_flux_offset([[1.0, 1.0], [1.1, 1.1]], offset_threshold=0.005)
        assert r.offset_detected is True
        assert r.flag == "OFFSET_DETECTED"

    def test_single_sector_insufficient(self) -> None:
        r = detect_flux_offset([[1.0, 1.0]])
        assert r.flag == "INSUFFICIENT_SECTORS"

    def test_empty_sector(self) -> None:
        r = detect_flux_offset([[1.0, 1.0], []])
        assert r.flag == "EMPTY_SECTOR"

    def test_grand_mean_correct(self) -> None:
        r = detect_flux_offset([[1.0], [2.0]])
        assert abs(r.grand_mean - 1.5) < 1e-9

    def test_max_offset_correct(self) -> None:
        r = detect_flux_offset([[1.0], [1.1]], offset_threshold=0.2)
        # sector_means=[1.0,1.1], grand_mean=1.05
        # offsets=[0.05/1.05, 0.05/1.05] ≈ 0.0476
        assert r.max_offset > 0.0

    def test_offset_detected_true(self) -> None:
        r = detect_flux_offset([[1.0], [2.0]])
        assert r.offset_detected is True

    def test_offset_detected_false(self) -> None:
        r = detect_flux_offset([[1.0, 1.0], [1.001, 1.001]], offset_threshold=0.005)
        assert r.offset_detected is False

    def test_offset_fraction_non_negative(self) -> None:
        r = detect_flux_offset([[1.0, 1.0], [1.0, 1.0]])
        assert r.offset_fraction >= 0.0

    def test_flag_ok(self) -> None:
        r = detect_flux_offset([[1.0, 1.0], [1.0, 1.0]])
        assert r.flag == "OK"

    def test_custom_threshold(self) -> None:
        # grand_mean=1.01, offset≈0.0099 > 0.005 → detected
        r = detect_flux_offset([[1.0], [1.02]], offset_threshold=0.005)
        assert r.offset_detected is True

    def test_format_returns_string(self) -> None:
        r = detect_flux_offset([[1.0], [1.0]])
        s = format_flux_offset_result(r)
        assert isinstance(s, str)
        assert "Offset" in s
