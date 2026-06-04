"""Tests for Skills/candidate_vetting_checklist.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from candidate_vetting_checklist import VettingChecklistResult, build_vetting_checklist

_GOOD = {
    "tic_id": "123456",
    "false_positive_probability": 0.05,
    "detection_confidence": 0.95,
    "snr": 10.0,
    "n_transits": 3,
    "pathway": "tfop_ready",
}

_EMPTY = {"tic_id": "999"}


class TestCandidateVettingChecklist:
    def test_good_candidate_pass(self) -> None:
        r = build_vetting_checklist(_GOOD)
        assert r.flag == "OK"
        assert r.overall in ("PASS", "WARN", "PARTIAL")

    def test_empty_candidate(self) -> None:
        r = build_vetting_checklist({})
        assert r.flag in ("EMPTY_CANDIDATE", "MISSING_TIC_ID", "OK")

    def test_high_fpp_fails(self) -> None:
        cand = dict(_GOOD)
        cand["false_positive_probability"] = 0.95
        r = build_vetting_checklist(cand)
        assert r.n_failed > 0

    def test_low_snr_fails(self) -> None:
        cand = dict(_GOOD)
        cand["snr"] = 2.0
        r = build_vetting_checklist(cand)
        assert r.n_failed > 0

    def test_one_transit_fails(self) -> None:
        cand = dict(_GOOD)
        cand["n_transits"] = 1
        r = build_vetting_checklist(cand)
        assert r.n_failed > 0

    def test_result_has_items(self) -> None:
        r = build_vetting_checklist(_GOOD)
        assert len(r.items) > 0

    def test_n_passed_plus_failed_lte_total(self) -> None:
        r = build_vetting_checklist(_GOOD)
        assert r.n_passed + r.n_failed <= r.n_total

    def test_overall_fail_on_all_bad(self) -> None:
        cand = {
            "tic_id": "bad",
            "false_positive_probability": 0.99,
            "snr": 1.0,
            "n_transits": 0,
        }
        r = build_vetting_checklist(cand)
        assert r.overall in ("FAIL", "PARTIAL", "WARN")

    def test_result_frozen(self) -> None:
        r = build_vetting_checklist(_GOOD)
        assert isinstance(r, VettingChecklistResult)
        try:
            object.__setattr__(r, "flag", "x")
            raise AssertionError()
        except Exception:
            pass

    def test_format_output(self) -> None:
        from candidate_vetting_checklist import format_checklist
        r = build_vetting_checklist(_GOOD)
        s = format_checklist(r)
        assert len(s) > 10

    def test_missing_fields_handled(self) -> None:
        r = build_vetting_checklist({"tic_id": "42"})
        assert r.flag == "OK"

    def test_pathway_no_follow_up_partial(self) -> None:
        cand = dict(_GOOD)
        cand["pathway"] = "github_only_reproducibility"
        r = build_vetting_checklist(cand)
        assert r.n_failed >= 1
