"""Tests for Skills/candidate_prioritization_report.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from candidate_prioritization_report import (
    CandidatePriority,
    build_prioritization_report,
    write_prioritization_report,
)

_ROWS = [
    {"tic_id": 1, "period_days": 5.0, "snr": 20.0,
     "false_positive_probability": 0.05, "novelty_score": 0.9,
     "pathway": "tfop_ready",
     "scores": {"detection_confidence": 0.95}},
    {"tic_id": 2, "period_days": 10.0, "snr": 8.0,
     "false_positive_probability": 0.45, "novelty_score": 0.5,
     "pathway": "planet_hunters_discussion",
     "scores": {"detection_confidence": 0.60}},
    {"tic_id": 3, "period_days": 3.0, "snr": 5.0,
     "false_positive_probability": 0.85, "novelty_score": 0.2,
     "pathway": "github_only_reproducibility",
     "scores": {"detection_confidence": 0.30}},
]


def test_returns_string():
    report = build_prioritization_report(_ROWS)
    assert isinstance(report, str)


def test_empty_input():
    report = build_prioritization_report([])
    assert "No candidates" in report


def test_title_in_report():
    report = build_prioritization_report(_ROWS, title="My Report")
    assert "My Report" in report


def test_top_n_limits():
    report = build_prioritization_report(_ROWS, top_n=2)
    # Should only contain 2 ranked entries
    lines = [l for l in report.split("\n") if l.startswith("| ") and "Rank" not in l]
    assert len(lines) <= 2


def test_urgent_section_present():
    report = build_prioritization_report(_ROWS)
    # TIC 1 with fpp=0.05 and tfop pathway should be URGENT
    assert "Urgent" in report


def test_contains_tic_ids():
    report = build_prioritization_report(_ROWS)
    assert "1" in report
    assert "2" in report


def test_write_creates_file(tmp_path):
    output = tmp_path / "report.md"
    p = write_prioritization_report(_ROWS, output)
    assert p.exists()


def test_write_contents_valid(tmp_path):
    output = tmp_path / "report.md"
    write_prioritization_report(_ROWS, output, title="Test Title")
    content = output.read_text()
    assert "Test Title" in content


def test_prioritization_sorted(tmp_path):
    report = build_prioritization_report(_ROWS)
    # Best candidate (tic=1) should appear before worst (tic=3)
    pos1 = report.find("| 1 |")
    pos3 = report.find("| 3 |")
    assert pos1 < pos3


def test_fpp_shown():
    report = build_prioritization_report(_ROWS)
    assert "0.05" in report or "0.050" in report


def test_pathway_in_action():
    report = build_prioritization_report(_ROWS)
    assert "TFOP" in report or "tfop" in report.lower()


def test_write_creates_parent_dirs(tmp_path):
    output = tmp_path / "sub" / "report.md"
    write_prioritization_report(_ROWS, output)
    assert output.exists()
