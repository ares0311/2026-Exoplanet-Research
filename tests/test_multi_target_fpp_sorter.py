"""Tests for Skills/multi_target_fpp_sorter.py."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from multi_target_fpp_sorter import (
    format_fpp_tier_table,
    sort_by_fpp,
)


class TestSortByFpp:
    def _cand(self, tic: str, fpp: float) -> dict:
        return {"tic_id": tic, "false_positive_probability": fpp}

    def test_empty(self) -> None:
        r = sort_by_fpp([])
        assert r.flag == "NO_CANDIDATES"
        assert r.n_total == 0

    def test_single_tier_a(self) -> None:
        r = sort_by_fpp([self._cand("1", 0.02)])
        assert r.n_tier_a == 1
        assert r.entries[0].tier == "A"

    def test_tier_b(self) -> None:
        r = sort_by_fpp([self._cand("1", 0.10)])
        assert r.entries[0].tier == "B"

    def test_tier_c(self) -> None:
        r = sort_by_fpp([self._cand("1", 0.30)])
        assert r.entries[0].tier == "C"

    def test_tier_d(self) -> None:
        r = sort_by_fpp([self._cand("1", 0.80)])
        assert r.entries[0].tier == "D"

    def test_sorted_ascending(self) -> None:
        cands = [self._cand("a", 0.40), self._cand("b", 0.01), self._cand("c", 0.20)]
        r = sort_by_fpp(cands)
        fpps = [e.fpp for e in r.entries]
        assert fpps == sorted(fpps)

    def test_counts_correct(self) -> None:
        cands = [self._cand("1", 0.02), self._cand("2", 0.10), self._cand("3", 0.70)]
        r = sort_by_fpp(cands)
        assert r.n_tier_a == 1
        assert r.n_tier_b == 1
        assert r.n_tier_d == 1

    def test_nested_scores(self) -> None:
        cand = {"tic_id": "1", "scores": {"false_positive_probability": 0.03}}
        r = sort_by_fpp([cand])
        assert r.entries[0].tier == "A"

    def test_missing_fpp_defaults_to_d(self) -> None:
        r = sort_by_fpp([{"tic_id": "1"}])
        assert r.entries[0].tier == "D"

    def test_n_total(self) -> None:
        cands = [self._cand(str(i), i * 0.1) for i in range(5)]
        r = sort_by_fpp(cands)
        assert r.n_total == 5

    def test_format_output(self) -> None:
        cands = [self._cand("1", 0.02), self._cand("2", 0.60)]
        r = sort_by_fpp(cands)
        s = format_fpp_tier_table(r)
        assert "|" in s
        assert "Tier" in s

    def test_format_empty(self) -> None:
        r = sort_by_fpp([])
        s = format_fpp_tier_table(r)
        assert "No candidates" in s
