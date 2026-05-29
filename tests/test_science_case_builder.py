"""Tests for Skills/science_case_builder.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from science_case_builder import (
    ScienceCase,
    build_science_case,
    format_science_case,
)

_ROW = {
    "tic_id": 150428135,
    "period_days": 9.9765,
    "depth_ppm": 1200.0,
    "duration_hours": 1.5,
    "false_positive_probability": 0.03,
    "pathway": "tfop_ready",
    "scores": {"detection_confidence": 0.95},
    "posterior": {"planet_candidate": 0.88},
}


def test_returns_science_case():
    sc = build_science_case(_ROW)
    assert isinstance(sc, ScienceCase)


def test_tic_id_stored():
    sc = build_science_case(_ROW)
    assert sc.tic_id == 150428135


def test_period_stored():
    sc = build_science_case(_ROW)
    assert abs(sc.period_days - 9.9765) < 1e-4


def test_depth_stored():
    sc = build_science_case(_ROW)
    assert sc.depth_ppm == 1200.0


def test_pathway_stored():
    sc = build_science_case(_ROW)
    assert sc.pathway == "tfop_ready"


def test_flag_ok_with_period_and_depth():
    sc = build_science_case(_ROW)
    assert sc.flag == "OK"


def test_flag_incomplete_without_period():
    row = {"tic_id": 1, "depth_ppm": 1000.0}
    sc = build_science_case(row)
    assert sc.flag == "INCOMPLETE"


def test_sections_nonempty():
    sc = build_science_case(_ROW)
    assert len(sc.sections) > 0


def test_vetting_in_sections():
    sc = build_science_case(_ROW)
    headings = [s[0] for s in sc.sections]
    assert any("Vet" in h for h in headings)


def test_host_star_summary_stored():
    sc = build_science_case(_ROW, host_star_summary="G-type, Tmag=10")
    assert "G-type" in sc.host_star_summary


def test_extra_notes_added():
    sc = build_science_case(_ROW, extra_notes="Priority target")
    headings = [s[0] for s in sc.sections]
    assert any("Note" in h for h in headings)


def test_planet_radius_stored():
    sc = build_science_case(_ROW, planet_radius_rearth=1.5)
    assert sc.planet_radius_rearth == 1.5


def test_format_returns_string():
    sc = build_science_case(_ROW)
    text = format_science_case(sc)
    assert isinstance(text, str)
    assert "Science Case" in text


def test_format_contains_tic():
    sc = build_science_case(_ROW)
    text = format_science_case(sc)
    assert "150428135" in text
