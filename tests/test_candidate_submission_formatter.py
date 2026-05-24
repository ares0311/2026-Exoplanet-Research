"""Tests for Skills/candidate_submission_formatter.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from candidate_submission_formatter import format_submission


_GOOD_CANDIDATE = {
    "tic_id": 150428135,
    "period_days": 37.4228,
    "epoch_bjd": 2458355.12,
    "depth_ppm": 2500.0,
    "duration_hours": 3.2,
    "planet_radius_rearth": 1.4,
    "false_positive_probability": 0.04,
    "detection_confidence": 0.92,
    "transit_count": 3,
    "sectors": [1, 2, 28],
    "stellar_radius_rsun": 0.55,
    "stellar_mass_msun": 0.58,
    "stellar_teff_k": 3800.0,
    "pathway": "tfop_ready",
}


def test_tfop_ok():
    r = format_submission(_GOOD_CANDIDATE, template="tfop_wg")
    assert r.flag == "OK"


def test_planet_hunters_ok():
    r = format_submission(_GOOD_CANDIDATE, template="planet_hunters")
    assert r.flag == "OK"


def test_tic_id_extracted():
    r = format_submission(_GOOD_CANDIDATE)
    assert r.tic_id == 150428135


def test_period_extracted():
    r = format_submission(_GOOD_CANDIDATE)
    assert r.period_days == 37.4228


def test_sectors_as_string():
    r = format_submission(_GOOD_CANDIDATE)
    assert r.sectors is not None
    assert "1" in r.sectors and "2" in r.sectors


def test_missing_tic_id_incomplete():
    cand = dict(_GOOD_CANDIDATE)
    del cand["tic_id"]
    r = format_submission(cand)
    assert r.flag == "INCOMPLETE"


def test_missing_period_incomplete():
    cand = dict(_GOOD_CANDIDATE)
    del cand["period_days"]
    r = format_submission(cand)
    assert r.flag == "INCOMPLETE"


def test_invalid_template():
    r = format_submission(_GOOD_CANDIDATE, template="unknown_template")
    assert r.flag == "INVALID"


def test_invalid_non_dict():
    r = format_submission("not a dict")
    assert r.flag == "INVALID"


def test_formatted_text_nonempty():
    r = format_submission(_GOOD_CANDIDATE)
    assert len(r.formatted_text) > 50


def test_tfop_text_contains_tic():
    r = format_submission(_GOOD_CANDIDATE, template="tfop_wg")
    assert "150428135" in r.formatted_text


def test_ph_text_contains_period():
    r = format_submission(_GOOD_CANDIDATE, template="planet_hunters")
    assert "37.4228" in r.formatted_text


def test_extra_notes_included():
    r = format_submission(_GOOD_CANDIDATE, extra_notes="Follow-up with NEID.")
    assert "NEID" in r.notes


def test_disclaimer_present():
    r = format_submission(_GOOD_CANDIDATE, template="tfop_wg")
    assert "candidate" in r.formatted_text.lower() or "DISCLAIMER" in r.formatted_text


def test_fpp_from_nested_scores():
    cand = {
        "tic_id": 123, "period_days": 5.0, "epoch_bjd": 2459000.0, "depth_ppm": 1000.0,
        "scores": {"false_positive_probability": 0.07},
    }
    r = format_submission(cand)
    assert r.fpp == 0.07
