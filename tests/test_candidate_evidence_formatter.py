"""Tests for Skills/candidate_evidence_formatter.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from candidate_evidence_formatter import (
    format_evidence_report,
    format_evidence_text,
)


class TestFormatEvidenceReport:
    def test_clean_signal_pc(self) -> None:
        r = format_evidence_report("12345", 5.0, 0.05, {})
        assert r.disposition == "PC"
        assert r.flag == "OK"

    def test_high_fpp_fp(self) -> None:
        r = format_evidence_report("12345", 5.0, 0.9, {})
        assert r.disposition == "FP"

    def test_centroid_score_counter_evidence(self) -> None:
        r = format_evidence_report("12345", 5.0, 0.1, {"centroid": 0.8})
        assert any("centroid" in e.lower() for e in r.counter_evidence)

    def test_depth_consistency_supporting(self) -> None:
        r = format_evidence_report("12345", 5.0, 0.1, {"depth_consistency": 0.9})
        assert any("depth" in e.lower() for e in r.supporting_evidence)

    def test_invalid_fpp(self) -> None:
        r = format_evidence_report("12345", 5.0, 1.5, {})
        assert r.flag == "INVALID_FPP"

    def test_n_supporting_matches_list(self) -> None:
        r = format_evidence_report("12345", 5.0, 0.05, {"depth_consistency": 0.9})
        assert r.n_supporting == len(r.supporting_evidence)

    def test_n_counter_matches_list(self) -> None:
        r = format_evidence_report("12345", 5.0, 0.8, {"centroid": 0.9})
        assert r.n_counter == len(r.counter_evidence)

    def test_confidence_high_for_clean(self) -> None:
        r = format_evidence_report("12345", 5.0, 0.05, {})
        assert r.confidence_label == "HIGH"

    def test_confidence_low_for_high_fpp(self) -> None:
        r = format_evidence_report("12345", 5.0, 0.8, {})
        assert r.confidence_label == "LOW"

    def test_format_returns_string(self) -> None:
        r = format_evidence_report("12345", 5.0, 0.1, {})
        s = format_evidence_text(r)
        assert isinstance(s, str)

    def test_format_contains_tic_id(self) -> None:
        r = format_evidence_report("99999", 5.0, 0.1, {})
        s = format_evidence_text(r)
        assert "99999" in s

    def test_format_contains_disposition(self) -> None:
        r = format_evidence_report("12345", 5.0, 0.05, {})
        s = format_evidence_text(r)
        assert r.disposition in s
