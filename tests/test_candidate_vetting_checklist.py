"""Tests for Skills/candidate_vetting_checklist.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from candidate_vetting_checklist import build_vetting_checklist, format_vetting_checklist


class TestBuildVettingChecklist:
    def _full_candidate(self) -> dict[str, object]:
        return {
            "period_days": 5.0, "fpp": 0.1, "depth_ppm": 1000.0,
            "duration_hours": 2.0, "n_transits": 5, "snr": 15.0,
            "odd_even_significance": 0.5, "secondary_snr": 0.2,
            "centroid_offset_arcsec": 0.1, "stellar_radius_rsun": 1.0,
        }

    def test_full_candidate_complete(self) -> None:
        r = build_vetting_checklist(self._full_candidate())
        assert r.flag == "COMPLETE"
        assert r.n_completed == r.n_total

    def test_empty_candidate_incomplete(self) -> None:
        r = build_vetting_checklist({})
        assert r.flag == "INCOMPLETE"
        assert r.n_completed == 0

    def test_n_completed_plus_missing_equals_total(self) -> None:
        r = build_vetting_checklist({"period_days": 5.0, "fpp": 0.1})
        assert r.n_completed + len(r.missing) == r.n_total

    def test_partial_candidate(self) -> None:
        r = build_vetting_checklist({"period_days": 5.0, "fpp": 0.1})
        assert 0 < r.n_completed < r.n_total

    def test_completeness_fraction_in_range(self) -> None:
        r = build_vetting_checklist(self._full_candidate())
        assert 0.0 <= r.completeness_fraction <= 1.0

    def test_completeness_fraction_full(self) -> None:
        r = build_vetting_checklist(self._full_candidate())
        assert r.completeness_fraction == 1.0

    def test_none_value_counts_as_missing(self) -> None:
        c = self._full_candidate()
        c["snr"] = None
        r = build_vetting_checklist(c)
        assert r.flag == "INCOMPLETE"

    def test_completed_list_type(self) -> None:
        r = build_vetting_checklist(self._full_candidate())
        assert isinstance(r.completed, list)

    def test_missing_list_type(self) -> None:
        r = build_vetting_checklist({})
        assert isinstance(r.missing, list)

    def test_format_returns_string(self) -> None:
        r = build_vetting_checklist(self._full_candidate())
        s = format_vetting_checklist(r)
        assert isinstance(s, str)
        assert "Checklist" in s

    def test_format_contains_completed_marker(self) -> None:
        r = build_vetting_checklist(self._full_candidate())
        s = format_vetting_checklist(r)
        assert "[x]" in s

    def test_n_total_is_ten(self) -> None:
        r = build_vetting_checklist({})
        assert r.n_total == 10
