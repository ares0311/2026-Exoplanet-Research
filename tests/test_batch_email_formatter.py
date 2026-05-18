"""Tests for batch_email_formatter.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from batch_email_formatter import (
    format_batch_email,
    format_single_candidate_email,
)


def _cand(tic_id, fpp, period=5.0, pathway="tfop_ready", n_signals=2):
    return {
        "tic_id": tic_id,
        "best_fpp": fpp,
        "best_period_days": period,
        "best_pathway": pathway,
        "n_signals": n_signals,
    }


class TestFormatBatchEmail:
    def test_basic_ok(self):
        candidates = [_cand(1, 0.05), _cand(2, 0.1)]
        msg = format_batch_email(candidates)
        assert msg.flag == "OK"
        assert msg.n_candidates >= 1

    def test_empty_input(self):
        msg = format_batch_email([])
        assert msg.flag == "EMPTY"

    def test_no_candidates_above_threshold(self):
        candidates = [_cand(1, 0.8), _cand(2, 0.9)]
        msg = format_batch_email(candidates, fpp_threshold=0.3)
        assert msg.flag == "NO_CANDIDATES"

    def test_fpp_filter_applied(self):
        candidates = [_cand(1, 0.05), _cand(2, 0.5)]
        msg = format_batch_email(candidates, fpp_threshold=0.1)
        assert msg.n_candidates == 1

    def test_subject_contains_count(self):
        candidates = [_cand(1, 0.05)]
        msg = format_batch_email(candidates)
        assert "1" in msg.subject

    def test_plain_text_nonempty(self):
        candidates = [_cand(1, 0.05)]
        msg = format_batch_email(candidates)
        assert len(msg.plain_text) > 10

    def test_html_nonempty_by_default(self):
        candidates = [_cand(1, 0.05)]
        msg = format_batch_email(candidates)
        assert "<html" in msg.html.lower()

    def test_no_html_when_disabled(self):
        candidates = [_cand(1, 0.05)]
        msg = format_batch_email(candidates, include_html=False)
        assert msg.html == ""

    def test_custom_title(self):
        candidates = [_cand(1, 0.05)]
        msg = format_batch_email(candidates, title="My Custom Title")
        assert "My Custom Title" in msg.subject

    def test_result_frozen(self):
        msg = format_batch_email([_cand(1, 0.05)])
        try:
            msg.flag = "x"  # type: ignore[misc]
            raise AssertionError()
        except (AttributeError, TypeError):
            pass

    def test_plain_contains_tic(self):
        candidates = [_cand(12345, 0.05)]
        msg = format_batch_email(candidates)
        assert "12345" in msg.plain_text


class TestFormatSingleCandidateEmail:
    def test_basic_ok(self):
        row = _cand(999, 0.05)
        msg = format_single_candidate_email(row)
        assert msg.flag == "OK"
        assert msg.n_candidates == 1

    def test_subject_contains_tic(self):
        row = _cand(12345, 0.05)
        msg = format_single_candidate_email(row)
        assert "12345" in msg.subject

    def test_html_not_empty(self):
        msg = format_single_candidate_email(_cand(1, 0.05))
        assert len(msg.html) > 0
