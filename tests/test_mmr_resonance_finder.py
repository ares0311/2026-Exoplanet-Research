"""Tests for Skills/mmr_resonance_finder.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from mmr_resonance_finder import find_mmr_resonances, format_mmr_result


class TestFindMmrResonances:
    def test_ok_flag(self) -> None:
        r = find_mmr_resonances(10.0, 20.0)
        assert r.flag == "OK"

    def test_exact_2to1_near_resonance(self) -> None:
        r = find_mmr_resonances(10.0, 20.0)
        assert r.nearest_resonance.resonance_label == "2:1"
        assert r.nearest_resonance.deviation_percent < 0.001

    def test_exact_3to2_resonance(self) -> None:
        r = find_mmr_resonances(10.0, 15.0)
        assert r.nearest_resonance.resonance_label == "3:2"
        assert r.nearest_resonance.deviation_percent < 0.001

    def test_near_resonance_flag_set(self) -> None:
        r = find_mmr_resonances(10.0, 20.1)
        assert r.is_near_resonance

    def test_far_from_resonance_flag_not_set(self) -> None:
        r = find_mmr_resonances(10.0, 17.0, threshold_percent=1.0)
        assert not r.is_near_resonance

    def test_period_ratio_correct(self) -> None:
        r = find_mmr_resonances(10.0, 30.0)
        assert abs(r.period_ratio - 3.0) < 1e-10

    def test_all_matches_non_empty(self) -> None:
        r = find_mmr_resonances(10.0, 20.0)
        assert len(r.all_matches) > 0

    def test_deviation_sorted_ascending(self) -> None:
        r = find_mmr_resonances(10.0, 20.0)
        devs = [m.deviation_percent for m in r.all_matches]
        assert devs == sorted(devs)

    def test_invalid_inner_period(self) -> None:
        r = find_mmr_resonances(0.0, 20.0)
        assert r.flag == "INVALID_INNER_PERIOD"

    def test_invalid_outer_period(self) -> None:
        r = find_mmr_resonances(10.0, 0.0)
        assert r.flag == "INVALID_OUTER_PERIOD"

    def test_outer_not_longer(self) -> None:
        r = find_mmr_resonances(20.0, 10.0)
        assert r.flag == "OUTER_NOT_LONGER"

    def test_format_returns_string(self) -> None:
        r = find_mmr_resonances(10.0, 20.0)
        s = format_mmr_result(r)
        assert isinstance(s, str)
        assert r.flag in s
