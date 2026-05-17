"""Tests for Skills.false_positive_vetter."""
from __future__ import annotations

from Skills.false_positive_vetter import format_vetting_report, vet_candidate


def _row(features: dict | None = None) -> dict:
    return {"candidate_id": "TIC1-001", "features": features or {}}


class TestVetCandidate:
    def test_missing_features_give_missing_verdict(self) -> None:
        verdicts = vet_candidate(_row({}))
        assert all(v.verdict == "missing" for v in verdicts)

    def test_fp_indicator_above_fail_threshold_gives_fail(self) -> None:
        row = _row({"odd_even_mismatch_score": 0.60})
        verdicts = vet_candidate(row)
        oe = next(v for v in verdicts if v.feature_name == "odd_even_mismatch_score")
        assert oe.verdict == "fail"

    def test_fp_indicator_above_warn_below_fail_gives_warn(self) -> None:
        row = _row({"odd_even_mismatch_score": 0.30})
        verdicts = vet_candidate(row)
        oe = next(v for v in verdicts if v.feature_name == "odd_even_mismatch_score")
        assert oe.verdict == "warn"

    def test_fp_indicator_below_warn_gives_pass(self) -> None:
        row = _row({"odd_even_mismatch_score": 0.05})
        verdicts = vet_candidate(row)
        oe = next(v for v in verdicts if v.feature_name == "odd_even_mismatch_score")
        assert oe.verdict == "pass"

    def test_quality_score_below_fail_gives_fail(self) -> None:
        row = _row({"log_snr_score": 0.10})
        verdicts = vet_candidate(row)
        snr = next(v for v in verdicts if v.feature_name == "log_snr_score")
        assert snr.verdict == "fail"

    def test_quality_score_above_warn_gives_pass(self) -> None:
        row = _row({"log_snr_score": 0.95})
        verdicts = vet_candidate(row)
        snr = next(v for v in verdicts if v.feature_name == "log_snr_score")
        assert snr.verdict == "pass"

    def test_total_verdict_count_is_fp_plus_quality(self) -> None:
        # 23 FP indicators + 9 quality = 32 total
        verdicts = vet_candidate(_row({}))
        assert len(verdicts) == 32

    def test_fp_indicator_flag_set_correctly(self) -> None:
        row = _row({"odd_even_mismatch_score": 0.10, "log_snr_score": 0.90})
        verdicts = vet_candidate(row)
        fp = [v for v in verdicts if v.is_fp_indicator]
        qual = [v for v in verdicts if not v.is_fp_indicator]
        assert len(fp) == 23
        assert len(qual) == 9

    def test_warn_threshold_scale_raises_effective_threshold(self) -> None:
        # score=0.25 is above default warn=0.20 but below scaled warn=0.40
        row = _row({"odd_even_mismatch_score": 0.25})
        default_v = vet_candidate(row)
        scaled_v = vet_candidate(row, warn_threshold_scale=2.0)
        oe_default = next(v for v in default_v if v.feature_name == "odd_even_mismatch_score")
        oe_scaled = next(v for v in scaled_v if v.feature_name == "odd_even_mismatch_score")
        assert oe_default.verdict == "warn"
        assert oe_scaled.verdict == "pass"

    def test_score_stored_in_verdict(self) -> None:
        row = _row({"secondary_eclipse_score": 0.42})
        verdicts = vet_candidate(row)
        sec = next(v for v in verdicts if v.feature_name == "secondary_eclipse_score")
        assert sec.score == 0.42

    def test_none_score_gives_missing(self) -> None:
        row = {"candidate_id": "x", "features": {"odd_even_mismatch_score": None}}
        verdicts = vet_candidate(row)
        oe = next(v for v in verdicts if v.feature_name == "odd_even_mismatch_score")
        assert oe.verdict == "missing"


class TestFormatVettingReport:
    def test_report_contains_candidate_id(self) -> None:
        verdicts = vet_candidate(_row({}))
        report = format_vetting_report(verdicts, "MY-CANDIDATE")
        assert "MY-CANDIDATE" in report

    def test_report_contains_fp_indicators_section(self) -> None:
        verdicts = vet_candidate(_row({}))
        report = format_vetting_report(verdicts)
        assert "FP Indicators" in report
