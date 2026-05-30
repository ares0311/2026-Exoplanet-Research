"""Tests for Skills/flux_normalization_checker.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from flux_normalization_checker import check_flux_normalization, format_flux_norm_result


class TestCheckFluxNormalization:
    def test_well_normalized_ok(self) -> None:
        r = check_flux_normalization([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        assert r.flag == "OK"
        assert r.n_sectors_flagged == 0

    def test_offset_sector_flagged(self) -> None:
        r = check_flux_normalization([[1.0, 1.0], [1.05, 1.05]], tolerance=0.005)
        assert r.n_sectors_flagged > 0
        assert r.flag == "NORMALIZATION_ISSUE"

    def test_no_data(self) -> None:
        r = check_flux_normalization([])
        assert r.flag == "NO_DATA"

    def test_empty_sector(self) -> None:
        r = check_flux_normalization([[1.0, 1.0], []])
        assert r.flag == "EMPTY_SECTOR"

    def test_max_offset_correct(self) -> None:
        r = check_flux_normalization([[0.99], [1.01]])
        assert r.max_offset > 0

    def test_sector_medians_computed(self) -> None:
        r = check_flux_normalization([[1.0, 1.0], [1.01, 1.01]])
        assert len(r.sector_medians) == 2

    def test_custom_tolerance(self) -> None:
        r = check_flux_normalization([[1.0], [1.003]], tolerance=0.01)
        assert r.flag == "OK"

    def test_custom_target_median(self) -> None:
        r = check_flux_normalization([[0.0, 0.0], [0.0, 0.0]], target_median=0.0)
        assert r.max_offset == 0.0

    def test_n_sectors_flagged_correct(self) -> None:
        r = check_flux_normalization([[1.0], [1.1], [1.0]], tolerance=0.05)
        assert r.n_sectors_flagged == 1

    def test_offsets_length_matches_sectors(self) -> None:
        r = check_flux_normalization([[1.0], [1.0], [1.0]])
        assert len(r.offsets) == 3

    def test_format_returns_string(self) -> None:
        r = check_flux_normalization([[1.0, 1.0]])
        s = format_flux_norm_result(r)
        assert isinstance(s, str)
        assert "Normalization" in s

    def test_result_frozen(self) -> None:
        r = check_flux_normalization([[1.0]])
        try:
            r.flag = "X"  # type: ignore[misc]
            raise AssertionError("should be frozen")
        except Exception:
            pass
