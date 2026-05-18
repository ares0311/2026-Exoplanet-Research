"""Tests for Skills.candidate_report_card."""
from __future__ import annotations

import json
from pathlib import Path

from Skills.candidate_report_card import (
    CandidateReportCard,
    build_report_card,
    format_report_card,
    save_report_card,
)

_ROW = {
    "tic_id": 150428135,
    "period_days": 37.42,
    "epoch_bjd": 2458360.0,
    "depth_ppm": 1050.0,
    "false_positive_probability": 0.05,
    "pathway": "tfop_ready",
    "n_transits": 4,
}


class TestBuildReportCard:
    def test_returns_result(self) -> None:
        r = build_report_card(_ROW)
        assert isinstance(r, CandidateReportCard)

    def test_tic_id_stored(self) -> None:
        r = build_report_card(_ROW)
        assert r.tic_id == 150428135

    def test_sections_present(self) -> None:
        r = build_report_card(_ROW)
        assert len(r.sections) >= 2

    def test_promising_flag_good_candidate(self) -> None:
        r = build_report_card(_ROW)
        assert r.overall_flag in {"PROMISING", "MARGINAL", "INSUFFICIENT"}

    def test_likely_fp_high_fpp(self) -> None:
        row = {**_ROW, "false_positive_probability": 0.95}
        r = build_report_card(row)
        assert r.overall_flag in {"LIKELY_FP", "MARGINAL"}

    def test_custom_diagnostics_added(self) -> None:
        diag = [{"title": "Centroid", "status": "PASS", "details": "No shift"}]
        r = build_report_card(_ROW, diagnostics=diag)
        titles = [s.title for s in r.sections]
        assert "Centroid" in titles

    def test_generated_at_is_string(self) -> None:
        r = build_report_card(_ROW)
        assert isinstance(r.generated_at, str)

    def test_recommendation_is_string(self) -> None:
        r = build_report_card(_ROW)
        assert isinstance(r.recommendation, str)
        assert len(r.recommendation) > 0

    def test_section_statuses_valid(self) -> None:
        r = build_report_card(_ROW)
        for s in r.sections:
            assert s.status in {"PASS", "WARN", "FAIL", "SKIP"}

    def test_missing_fpp_section_skip(self) -> None:
        row = {k: v for k, v in _ROW.items() if k != "false_positive_probability"}
        r = build_report_card(row)
        fpp_section = next((s for s in r.sections if "False Positive" in s.title), None)
        assert fpp_section is not None
        assert fpp_section.status == "SKIP"


class TestFormatReportCard:
    def test_returns_string(self) -> None:
        r = build_report_card(_ROW)
        assert isinstance(format_report_card(r), str)

    def test_contains_tic_id(self) -> None:
        r = build_report_card(_ROW)
        assert "150428135" in format_report_card(r)

    def test_contains_overall_flag(self) -> None:
        r = build_report_card(_ROW)
        assert r.overall_flag in format_report_card(r)


class TestSaveReportCard:
    def test_creates_file(self, tmp_path: Path) -> None:
        r = build_report_card(_ROW)
        p = save_report_card(r, tmp_path / "card.json")
        assert p.exists()

    def test_json_parseable(self, tmp_path: Path) -> None:
        r = build_report_card(_ROW)
        p = save_report_card(r, tmp_path / "card.json")
        data = json.loads(p.read_text())
        assert data["tic_id"] == 150428135
