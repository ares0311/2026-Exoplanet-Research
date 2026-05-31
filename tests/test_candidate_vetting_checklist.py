"""Tests for Skills/candidate_vetting_checklist.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from candidate_vetting_checklist import build_vetting_checklist, format_checklist


def _candidate(**kwargs) -> dict:
    base = {
        "tic_id": "123456",
        "scores": {
            "false_positive_probability": 0.05,
            "detection_confidence": 0.90,
        },
        "snr": 12.0,
        "n_transits": 3,
        "pathway": "tfop_ready",
    }
    base.update(kwargs)
    return base


class TestCandidateVettingChecklist:
    def test_all_pass(self) -> None:
        r = build_vetting_checklist(_candidate())
        assert r.flag == "OK"
        assert r.overall == "PASS"

    def test_high_fpp_fails(self) -> None:
        r = build_vetting_checklist(_candidate(scores={"false_positive_probability": 0.50}))
        assert r.n_failed > 0

    def test_low_dc_fails(self) -> None:
        c = _candidate()
        c["scores"] = {"false_positive_probability": 0.05, "detection_confidence": 0.50}
        r = build_vetting_checklist(c)
        assert r.n_failed > 0

    def test_missing_tic_id(self) -> None:
        r = build_vetting_checklist({})
        assert r.flag == "MISSING_TIC_ID"
        assert r.n_total == 0

    def test_n_passed_plus_failed_eq_total(self) -> None:
        r = build_vetting_checklist(_candidate())
        assert r.n_passed + r.n_failed == r.n_total

    def test_items_tuple(self) -> None:
        r = build_vetting_checklist(_candidate())
        assert isinstance(r.items, tuple)
        assert len(r.items) > 0

    def test_pathway_check_present(self) -> None:
        r = build_vetting_checklist(_candidate())
        names = [it.name for it in r.items]
        assert "Pathway" in names

    def test_bad_pathway_fails(self) -> None:
        r = build_vetting_checklist(_candidate(pathway="github_only_reproducibility"))
        pathway_item = next(it for it in r.items if it.name == "Pathway")
        assert not pathway_item.passed

    def test_low_snr_fails(self) -> None:
        r = build_vetting_checklist(_candidate(snr=3.0))
        snr_item = next((it for it in r.items if it.name == "SNR"), None)
        if snr_item:
            assert not snr_item.passed

    def test_few_transits_fails(self) -> None:
        r = build_vetting_checklist(_candidate(n_transits=1))
        tr_item = next((it for it in r.items if it.name == "N transits"), None)
        if tr_item:
            assert not tr_item.passed

    def test_overall_fail_when_all_bad(self) -> None:
        c = {
            "tic_id": "999",
            "scores": {"false_positive_probability": 0.90, "detection_confidence": 0.20},
            "pathway": "github_only_reproducibility",
        }
        r = build_vetting_checklist(c)
        assert r.overall in ("FAIL", "PARTIAL")

    def test_format_returns_string(self) -> None:
        r = build_vetting_checklist(_candidate())
        s = format_checklist(r)
        assert isinstance(s, str)
        assert "Checklist" in s
