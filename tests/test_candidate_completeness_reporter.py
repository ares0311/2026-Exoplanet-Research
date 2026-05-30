"""Tests for Skills/candidate_completeness_reporter.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from candidate_completeness_reporter import build_completeness_report, format_completeness_report


class TestBuildCompletenessReport:
    def _full(self) -> dict[str, object]:
        return {
            "snr": 15.0, "centroid_offset_arcsec": 0.1,
            "stellar_radius_rsun": 1.0, "fpp": 0.05,
        }

    def test_empty_candidates(self) -> None:
        r = build_completeness_report([])
        assert r.flag == "NO_CANDIDATES"
        assert r.n_candidates == 0

    def test_full_vetted_complete(self) -> None:
        r = build_completeness_report([self._full()])
        assert r.n_fully_vetted == 1
        assert r.flag == "COMPLETE"

    def test_partial_incomplete(self) -> None:
        r = build_completeness_report([{"snr": 15.0}])
        assert r.flag == "PARTIAL"

    def test_n_candidates_correct(self) -> None:
        r = build_completeness_report([self._full(), {"snr": 10.0}])
        assert r.n_candidates == 2

    def test_n_missing_snr(self) -> None:
        r = build_completeness_report([{"fpp": 0.1}])
        assert r.n_missing_snr == 1

    def test_n_missing_centroid(self) -> None:
        r = build_completeness_report([{"snr": 10.0}])
        assert r.n_missing_centroid == 1

    def test_n_missing_stellar(self) -> None:
        r = build_completeness_report([{"snr": 10.0, "centroid_offset_arcsec": 0.1}])
        assert r.n_missing_stellar == 1

    def test_mean_fpp_correct(self) -> None:
        c1 = dict(self._full())
        c1["fpp"] = 0.1
        c2 = dict(self._full())
        c2["fpp"] = 0.3
        r = build_completeness_report([c1, c2])
        assert abs(r.mean_fpp - 0.2) < 1e-9

    def test_mean_fpp_zero_if_no_fpp(self) -> None:
        r = build_completeness_report([{"snr": 10.0}])
        assert r.mean_fpp == 0.0

    def test_format_returns_string(self) -> None:
        r = build_completeness_report([self._full()])
        s = format_completeness_report(r)
        assert isinstance(s, str)
        assert "Completeness" in s

    def test_result_frozen(self) -> None:
        r = build_completeness_report([self._full()])
        try:
            r.flag = "X"  # type: ignore[misc]
            raise AssertionError("should be frozen")
        except Exception:
            pass

    def test_multiple_fully_vetted(self) -> None:
        r = build_completeness_report([self._full(), self._full(), self._full()])
        assert r.n_fully_vetted == 3
