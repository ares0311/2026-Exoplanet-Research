"""Tests for Skills/candidate_cross_sector_validator.py."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from candidate_cross_sector_validator import (
    format_cross_sector_result,
    validate_cross_sector,
)


class TestValidateCrossSector:
    def _consistent(self) -> list[dict]:
        return [
            {"period_days": 5.0, "period_err_days": 0.001,
             "depth_ppm": 1000.0, "depth_err_ppm": 20.0,
             "duration_hours": 3.0, "duration_err_hours": 0.1},
            {"period_days": 5.001, "period_err_days": 0.001,
             "depth_ppm": 998.0, "depth_err_ppm": 20.0,
             "duration_hours": 3.01, "duration_err_hours": 0.1},
        ]

    def test_consistent_result(self) -> None:
        r = validate_cross_sector(self._consistent())
        assert r.overall == "CONSISTENT"
        assert r.flag == "OK"

    def test_single_sector_always_consistent(self) -> None:
        r = validate_cross_sector([{"period_days": 5.0}])
        assert r.flag == "SINGLE_SECTOR"
        assert r.overall == "CONSISTENT"

    def test_inconsistent_period(self) -> None:
        meas = [
            {"period_days": 5.0, "period_err_days": 0.001},
            {"period_days": 6.0, "period_err_days": 0.001},
        ]
        r = validate_cross_sector(meas, period_rtol=0.01)
        assert not r.period_consistent

    def test_inconsistent_depth(self) -> None:
        meas = [
            {"depth_ppm": 1000.0, "depth_err_ppm": 10.0},
            {"depth_ppm": 2000.0, "depth_err_ppm": 10.0},
        ]
        r = validate_cross_sector(meas, depth_rtol=0.10)
        assert not r.depth_consistent

    def test_partial_inconsistency(self) -> None:
        meas = [
            {"period_days": 5.0, "depth_ppm": 1000.0, "duration_hours": 3.0},
            {"period_days": 5.0, "depth_ppm": 2000.0, "duration_hours": 3.0},
        ]
        r = validate_cross_sector(meas, depth_rtol=0.05)
        assert r.overall == "PARTIAL"

    def test_n_sectors_count(self) -> None:
        r = validate_cross_sector(self._consistent())
        assert r.n_sectors == 2

    def test_three_sectors(self) -> None:
        meas = self._consistent() + [{"period_days": 5.002}]
        r = validate_cross_sector(meas)
        assert r.n_sectors == 3

    def test_all_inconsistent(self) -> None:
        meas = [
            {"period_days": 5.0, "depth_ppm": 1000.0, "duration_hours": 3.0,
             "period_err_days": 0.001, "depth_err_ppm": 10.0, "duration_err_hours": 0.05},
            {"period_days": 10.0, "depth_ppm": 3000.0, "duration_hours": 6.0,
             "period_err_days": 0.001, "depth_err_ppm": 10.0, "duration_err_hours": 0.05},
        ]
        r = validate_cross_sector(meas, period_rtol=0.01, depth_rtol=0.05, duration_rtol=0.05)
        assert r.overall in ("INCONSISTENT", "PARTIAL")

    def test_n_inconsistent_count(self) -> None:
        r = validate_cross_sector(self._consistent())
        assert r.n_inconsistent == 0

    def test_empty_input(self) -> None:
        r = validate_cross_sector([])
        assert r.flag == "SINGLE_SECTOR"

    def test_format_output(self) -> None:
        r = validate_cross_sector(self._consistent())
        s = format_cross_sector_result(r)
        assert "|" in s
        assert "Period" in s or "period" in s

    def test_missing_keys_treated_as_consistent(self) -> None:
        meas = [{"sector": 1}, {"sector": 2}]
        r = validate_cross_sector(meas)
        assert r.flag == "OK"
        assert r.overall == "CONSISTENT"
