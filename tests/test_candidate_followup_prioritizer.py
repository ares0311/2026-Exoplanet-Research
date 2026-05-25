"""Tests for Skills/candidate_followup_prioritizer.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from candidate_followup_prioritizer import format_followup_priorities, prioritize_followup

CANDIDATES = [
    {
        "candidate_id": "c1", "fpp": 0.05, "detection_confidence": 0.90,
        "pathway": "tfop_ready", "provenance_score": 0.95,
    },
    {
        "candidate_id": "c2", "fpp": 0.30, "detection_confidence": 0.60,
        "pathway": "planet_hunters_discussion", "provenance_score": 0.60,
    },
    {
        "candidate_id": "c3", "fpp": 0.85, "detection_confidence": 0.10,
        "pathway": "github_only_reproducibility", "provenance_score": 0.30,
    },
]


def test_basic_ok():
    r = prioritize_followup(CANDIDATES)
    assert r.flag == "OK"


def test_n_candidates():
    r = prioritize_followup(CANDIDATES)
    assert r.n_candidates == 3


def test_urgent_count():
    r = prioritize_followup(CANDIDATES)
    assert r.n_urgent >= 1


def test_skip_high_fpp():
    r = prioritize_followup(CANDIDATES)
    recs = [e.recommendation for e in r.entries if e.candidate_id == "c3"]
    assert recs == ["skip"]


def test_priority_ordering():
    r = prioritize_followup(CANDIDATES)
    scores = [e.priority_score for e in r.entries]
    assert scores == sorted(scores, reverse=True)


def test_priority_ranks():
    r = prioritize_followup(CANDIDATES)
    ranks = [e.priority_rank for e in r.entries]
    assert ranks == list(range(1, len(r.entries) + 1))


def test_empty_returns_empty():
    r = prioritize_followup([])
    assert r.flag == "EMPTY"


def test_invalid_input():
    r = prioritize_followup("not a list")
    assert r.flag == "INVALID"


def test_format_returns_string():
    r = prioritize_followup(CANDIDATES)
    assert isinstance(format_followup_priorities(r), str)


def test_format_contains_key_words():
    r = prioritize_followup(CANDIDATES)
    text = format_followup_priorities(r)
    assert "Follow-Up" in text
    assert "Flag" in text


def test_missing_fields_uses_defaults():
    r = prioritize_followup([{"candidate_id": "cx"}])
    assert r.flag == "OK"
    assert r.entries[0].priority_score > 0


def test_scores_in_range():
    r = prioritize_followup(CANDIDATES)
    for e in r.entries:
        assert 0.0 <= e.priority_score <= 1.0


def test_pathway_bonus_tfop():
    r = prioritize_followup([
        {"candidate_id": "a", "fpp": 0.10, "pathway": "tfop_ready"},
        {"candidate_id": "b", "fpp": 0.10, "pathway": "github_only_reproducibility"},
    ])
    scores = {e.candidate_id: e.priority_score for e in r.entries}
    assert scores["a"] > scores["b"]
