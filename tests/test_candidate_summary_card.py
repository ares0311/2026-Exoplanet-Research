"""Tests for Skills/candidate_summary_card.py."""
import pytest
from Skills.candidate_summary_card import (
    SummaryCardResult,
    build_summary_card,
    format_summary_card,
)


class TestBuildSummaryCard:
    def _candidate(self, **kwargs):
        base = {
            "tic_id": 150428135,
            "period_days": 37.42,
            "depth_ppm": 2500,
            "false_positive_probability": 0.08,
            "pathway": "tfop_ready",
        }
        base.update(kwargs)
        return base

    def test_returns_result_type(self):
        result = build_summary_card(self._candidate())
        assert isinstance(result, SummaryCardResult)

    def test_flag_ok(self):
        result = build_summary_card(self._candidate())
        assert result.flag == "OK"

    def test_flag_empty_for_empty_dict(self):
        result = build_summary_card({})
        assert result.flag == "EMPTY"

    def test_flag_invalid_for_non_dict(self):
        result = build_summary_card("not a dict")
        assert result.flag == "INVALID"

    def test_tic_id_extracted(self):
        result = build_summary_card(self._candidate())
        assert "150428135" in result.tic_id

    def test_period_extracted(self):
        result = build_summary_card(self._candidate())
        assert result.period_days == 37.42

    def test_depth_extracted(self):
        result = build_summary_card(self._candidate())
        assert result.depth_ppm == 2500

    def test_fpp_from_top_level(self):
        result = build_summary_card(self._candidate())
        assert result.fpp == 0.08

    def test_fpp_from_scores_dict(self):
        d = {"tic_id": 1, "scores": {"false_positive_probability": 0.15}}
        result = build_summary_card(d)
        assert result.fpp == 0.15

    def test_fpp_from_fpp_key(self):
        d = {"tic_id": 1, "fpp": 0.23}
        result = build_summary_card(d)
        assert result.fpp == 0.23

    def test_pathway_extracted(self):
        result = build_summary_card(self._candidate())
        assert result.pathway == "tfop_ready"

    def test_submission_pathway_fallback(self):
        d = {"tic_id": 1, "submission_pathway": "planet_hunters_discussion"}
        result = build_summary_card(d)
        assert result.pathway == "planet_hunters_discussion"

    def test_flags_from_explanation(self):
        d = self._candidate()
        d["explanation"] = {"blocking_evidence": ["centroid_shift", "v_shaped"]}
        result = build_summary_card(d)
        assert len(result.top_flags) >= 1

    def test_formatted_card_contains_tic(self):
        result = build_summary_card(self._candidate())
        assert "150428135" in result.formatted_card

    def test_formatted_card_contains_period(self):
        result = build_summary_card(self._candidate())
        assert "37.42" in result.formatted_card

    def test_missing_fields_use_dash(self):
        result = build_summary_card({"tic_id": 99})
        assert "—" in result.formatted_card


class TestFormatSummaryCard:
    def test_returns_string(self):
        result = build_summary_card({"tic_id": 1, "period_days": 5.0})
        s = format_summary_card(result)
        assert isinstance(s, str)

    def test_empty_flag_output(self):
        result = build_summary_card({})
        s = format_summary_card(result)
        assert "EMPTY" in s

    def test_invalid_flag_output(self):
        result = build_summary_card(None)
        s = format_summary_card(result)
        assert "INVALID" in s
