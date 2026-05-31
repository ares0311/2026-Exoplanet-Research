"""Tests for Skills/sector_cadence_recommender.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from sector_cadence_recommender import format_cadence_recommendation, recommend_cadence


class TestSectorCadenceRecommender:
    def test_normal_transit_target(self) -> None:
        r = recommend_cadence(12.0)
        assert r.flag == "OK"
        assert r.recommended_cadence in ("2min", "10min")

    def test_bright_target_only_20s_available(self) -> None:
        # Tmag=6: 2min and 10min saturate (6 < 7.5, 6 < 9.0); only 20s available
        r = recommend_cadence(6.0)
        assert r.flag == "OK"
        assert r.recommended_cadence == "20s"

    def test_very_bright_all_saturated(self) -> None:
        r = recommend_cadence(1.0)
        assert r.flag == "ALL_MODES_SATURATED"

    def test_asteroseismology_prefers_short(self) -> None:
        r = recommend_cadence(10.0, science_goal="asteroseismology")
        assert r.flag == "OK"
        assert r.recommended_cadence in ("20s", "2min")

    def test_flares_prefer_short(self) -> None:
        r = recommend_cadence(10.0, science_goal="flares")
        assert r.flag == "OK"
        assert r.recommended_cadence in ("20s", "2min")

    def test_faint_star_gets_2min(self) -> None:
        r = recommend_cadence(14.0)
        assert r.flag == "OK"
        assert r.recommended_cadence == "2min"

    def test_saturated_modes_populated(self) -> None:
        r = recommend_cadence(3.0)
        assert len(r.saturated_modes) > 0

    def test_no_saturation_for_dim_star(self) -> None:
        r = recommend_cadence(12.0)
        assert len(r.saturated_modes) == 0

    def test_prefer_short_flag(self) -> None:
        r = recommend_cadence(12.0, prefer_short=True)
        # Should prefer shorter cadence when explicitly requested
        assert r.recommended_cadence in ("20s", "2min")

    def test_invalid_tmag_nan(self) -> None:
        r = recommend_cadence(float("nan"))
        assert r.flag == "INVALID_TMAG"

    def test_reason_not_empty(self) -> None:
        r = recommend_cadence(12.0)
        assert len(r.reason) > 0

    def test_format_returns_string(self) -> None:
        r = recommend_cadence(12.0)
        s = format_cadence_recommendation(r)
        assert isinstance(s, str)
        assert "cadence" in s.lower()
