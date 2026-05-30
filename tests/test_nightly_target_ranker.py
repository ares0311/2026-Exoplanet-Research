"""Tests for Skills/nightly_target_ranker.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from nightly_target_ranker import (  # noqa: E402
    NightlyRankResult,
    format_nightly_rank_result,
    rank_nightly_targets,
)

TARGETS = [
    {"name": "TIC-1", "airmass": 1.2, "mag": 11.0, "transit_tonight": True},
    {"name": "TIC-2", "airmass": 1.8, "mag": 12.5, "transit_tonight": False},
    {"name": "TIC-3", "airmass": 2.0, "mag": 13.0, "transit_tonight": False},
]


def test_returns_dataclass():
    r = rank_nightly_targets(TARGETS)
    assert isinstance(r, NightlyRankResult)


def test_empty_targets():
    r = rank_nightly_targets([])
    assert r.flag == "NO_TARGETS"
    assert r.n_targets == 0
    assert r.best_target == ""


def test_transit_target_ranked_first():
    r = rank_nightly_targets(TARGETS)
    assert r.ranked_targets[0]["name"] == "TIC-1"


def test_n_targets_correct():
    r = rank_nightly_targets(TARGETS)
    assert r.n_targets == 3


def test_composite_score_range():
    r = rank_nightly_targets(TARGETS)
    for t in r.ranked_targets:
        assert 0.0 <= t["composite_score"] <= 1.0


def test_rank_field_assigned():
    r = rank_nightly_targets(TARGETS)
    ranks = [t["rank"] for t in r.ranked_targets]
    assert ranks == list(range(1, len(TARGETS) + 1))


def test_best_target_matches_first():
    r = rank_nightly_targets(TARGETS)
    assert r.best_target == r.ranked_targets[0]["name"]


def test_all_same_score_flag():
    same = [
        {"name": f"T{i}", "airmass": 1.5, "mag": 12.0, "transit_tonight": False}
        for i in range(3)
    ]
    r = rank_nightly_targets(same)
    assert r.flag == "ALL_SAME_SCORE"


def test_single_target_ok_flag():
    r = rank_nightly_targets([TARGETS[0]])
    assert r.flag == "OK"


def test_format_returns_string():
    r = rank_nightly_targets(TARGETS)
    s = format_nightly_rank_result(r)
    assert isinstance(s, str)


def test_format_contains_flag():
    r = rank_nightly_targets(TARGETS)
    s = format_nightly_rank_result(r)
    assert "Flag" in s


def test_night_date_param():
    r = rank_nightly_targets(TARGETS, night_date="2026-06-01")
    assert r.n_targets == 3
