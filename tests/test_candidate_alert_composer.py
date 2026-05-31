"""Tests for Skills/candidate_alert_composer.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from candidate_alert_composer import compose_alert, format_alert_message


def _candidate(**kwargs) -> dict:
    base = {
        "tic_id": "123456",
        "period_days": 5.0,
        "depth_ppm": 1000.0,
        "scores": {"false_positive_probability": 0.05, "detection_confidence": 0.90},
        "pathway": "tfop_ready",
    }
    base.update(kwargs)
    return base


class TestCandidateAlertComposer:
    def test_basic_alert(self) -> None:
        r = compose_alert(_candidate())
        assert r.flag == "OK"
        assert r.tic_id == "123456"

    def test_subject_contains_tic(self) -> None:
        r = compose_alert(_candidate())
        assert "123456" in r.subject

    def test_body_contains_period(self) -> None:
        r = compose_alert(_candidate())
        assert "5.0000" in r.body or "5.000" in r.body

    def test_body_contains_fpp(self) -> None:
        r = compose_alert(_candidate())
        assert "0.05" in r.body

    def test_slack_text_not_empty(self) -> None:
        r = compose_alert(_candidate())
        assert len(r.slack_text) > 0
        assert "123456" in r.slack_text

    def test_pathway_label_in_body(self) -> None:
        r = compose_alert(_candidate())
        assert "TFOP" in r.body or "tfop" in r.body.lower()

    def test_missing_tic_id(self) -> None:
        r = compose_alert({"period_days": 5.0})
        assert r.flag == "MISSING_TIC_ID"

    def test_custom_sender(self) -> None:
        r = compose_alert(_candidate(), sender="my-tool")
        assert "my-tool" in r.body

    def test_scores_nested_fpp(self) -> None:
        c = _candidate()
        r = compose_alert(c)
        assert r.flag == "OK"

    def test_top_level_fpp(self) -> None:
        c = {"tic_id": "999", "period_days": 3.0, "false_positive_probability": 0.02}
        r = compose_alert(c)
        assert r.flag == "OK"

    def test_no_period_in_body_graceful(self) -> None:
        c = {"tic_id": "777"}
        r = compose_alert(c)
        assert r.flag == "OK"

    def test_format_returns_string(self) -> None:
        r = compose_alert(_candidate())
        s = format_alert_message(r)
        assert isinstance(s, str)
        assert "Subject" in s
