"""Tests for Skills/observation_run_debriefer.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from observation_run_debriefer import (  # noqa: E402
    RunDebrief,
    debrief_run,
    format_run_debrief,
)


def test_returns_dataclass():
    r = debrief_run(200, 10)
    assert isinstance(r, RunDebrief)


def test_excellent_flag():
    r = debrief_run(200, 0, airmass_max=1.2, seeing_arcsec=1.5, cloud_cover_pct=0.0)
    assert r.flag == "EXCELLENT"


def test_poor_flag():
    r = debrief_run(200, 180, airmass_max=2.5, seeing_arcsec=4.0, cloud_cover_pct=90.0)
    assert r.flag == "POOR"


def test_marginal_flag():
    r = debrief_run(200, 80, airmass_max=1.8, seeing_arcsec=2.5, cloud_cover_pct=40.0)
    assert r.flag in ("MARGINAL", "POOR", "GOOD")


def test_n_usable_correct():
    r = debrief_run(200, 30)
    assert r.n_usable == 170


def test_quality_score_range():
    r = debrief_run(200, 10)
    assert 0.0 <= r.quality_score <= 1.0


def test_issues_tuple():
    r = debrief_run(200, 10, cloud_cover_pct=50.0)
    assert isinstance(r.issues, tuple)


def test_recommendations_tuple():
    r = debrief_run(200, 10)
    assert isinstance(r.recommendations, tuple)


def test_custom_issues_appended():
    r = debrief_run(200, 10, issues_list=["Guiding lost at 23:30"])
    assert any("Guiding" in issue for issue in r.issues)


def test_high_cloud_adds_issue():
    r = debrief_run(200, 10, cloud_cover_pct=60.0)
    assert any("cloud" in i.lower() for i in r.issues)


def test_format_returns_string():
    r = debrief_run(200, 10)
    s = format_run_debrief(r)
    assert isinstance(s, str)


def test_format_contains_flag():
    r = debrief_run(200, 10)
    s = format_run_debrief(r)
    assert "Flag" in s
