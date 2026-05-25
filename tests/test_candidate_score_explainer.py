"""Tests for Skills/candidate_score_explainer.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from candidate_score_explainer import (
    explain_candidate_score,
    format_score_explanation,
)


def _make_candidate(**kwargs):
    base = {
        "tic_id": 12345678,
        "period_days": 5.0,
        "pathway": "planet_hunters_discussion",
        "provenance_score": 0.85,
        "scores": {
            "false_positive_probability": 0.15,
            "detection_confidence": 0.90,
        },
        "posterior": {
            "planet_candidate": 0.65,
            "eclipsing_binary": 0.10,
            "instrumental_artifact": 0.05,
            "stellar_variability": 0.10,
            "background_eclipsing_binary": 0.05,
            "known_object": 0.05,
        },
    }
    base.update(kwargs)
    return base


def test_ok_flag():
    r = explain_candidate_score(_make_candidate())
    assert r.flag == "OK"


def test_low_fpp_supports_planet():
    r = explain_candidate_score(_make_candidate())
    labels = [e.feature for e in r.top_positive]
    assert any("false_positive_probability" in lbl or "fpp" in lbl.lower() for lbl in labels)


def test_high_fpp_against_planet():
    cand = _make_candidate()
    cand["scores"]["false_positive_probability"] = 0.85
    r = explain_candidate_score(cand)
    labels = [e.feature for e in r.top_negative]
    assert any("false_positive_probability" in lbl for lbl in labels)


def test_high_eb_posterior_against():
    cand = _make_candidate()
    cand["posterior"]["eclipsing_binary"] = 0.45
    r = explain_candidate_score(cand)
    assert any("eclipsing_binary" in e.feature for e in r.top_negative)


def test_tfop_ready_supports_planet():
    cand = _make_candidate(pathway="tfop_ready")
    r = explain_candidate_score(cand)
    assert any("pathway" in e.feature for e in r.top_positive)


def test_planet_posterior_extracted():
    r = explain_candidate_score(_make_candidate())
    assert abs(r.planet_posterior - 0.65) < 1e-9


def test_fpp_extracted():
    r = explain_candidate_score(_make_candidate())
    assert abs(r.fpp - 0.15) < 1e-9


def test_tic_id_extracted():
    r = explain_candidate_score(_make_candidate())
    assert r.tic_id == 12345678


def test_period_extracted():
    r = explain_candidate_score(_make_candidate())
    assert r.period_days == 5.0


def test_invalid_non_dict():
    r = explain_candidate_score("not a dict")
    assert r.flag == "INVALID"


def test_empty_dict_ok():
    r = explain_candidate_score({})
    assert r.flag == "OK"
    assert r.planet_posterior == 0.0


def test_instrumental_artifact_against():
    cand = _make_candidate()
    cand["posterior"]["instrumental_artifact"] = 0.40
    r = explain_candidate_score(cand)
    assert any("instrumental" in e.feature for e in r.top_negative)


def test_format_contains_keywords():
    r = explain_candidate_score(_make_candidate())
    text = format_score_explanation(r)
    assert "Score Explainer" in text
    assert "Supporting Evidence" in text
    assert "OK" in text
