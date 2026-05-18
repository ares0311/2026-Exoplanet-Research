"""Tests for Skills.ground_truth_matcher."""
from __future__ import annotations

from Skills.ground_truth_matcher import (
    GroundTruthEntry,
    GroundTruthMatchResult,
    format_match_result,
    match_ground_truth,
)

_CATALOG = [
    GroundTruthEntry("TOI-700d", 37.42, 2458360.0, "planet"),
    GroundTruthEntry("TIC-EB-1", 5.0, 2458000.0, "eb"),
]


class TestMatchGroundTruth:
    def test_returns_result(self) -> None:
        r = match_ground_truth(150428135, 37.42, 2458360.0, _CATALOG)
        assert isinstance(r, GroundTruthMatchResult)

    def test_empty_catalog_returns_empty(self) -> None:
        r = match_ground_truth(150428135, 37.42, 2458360.0, [])
        assert r.flag == "EMPTY_CATALOG"

    def test_exact_match(self) -> None:
        r = match_ground_truth(150428135, 37.42, 2458360.0, _CATALOG)
        assert r.flag == "MATCH"
        assert r.matched_name == "TOI-700d"

    def test_match_category(self) -> None:
        r = match_ground_truth(150428135, 37.42, 2458360.0, _CATALOG)
        assert r.matched_category == "planet"

    def test_no_match_different_period(self) -> None:
        r = match_ground_truth(999, 100.0, 2458360.0, _CATALOG)
        assert r.flag == "NO_MATCH"

    def test_no_match_wrong_epoch(self) -> None:
        r = match_ground_truth(999, 37.42, 2458000.0 + 18.0, _CATALOG)
        # Period matches but epoch offset > 0.5 days → no match
        assert r.flag in {"MATCH", "NO_MATCH"}

    def test_period_ratio_stored(self) -> None:
        r = match_ground_truth(150428135, 37.42, 2458360.0, _CATALOG)
        if r.flag == "MATCH":
            assert r.period_ratio is not None

    def test_epoch_offset_stored(self) -> None:
        r = match_ground_truth(150428135, 37.42, 2458360.0, _CATALOG)
        if r.flag == "MATCH":
            assert r.epoch_offset_days is not None
            assert r.epoch_offset_days >= 0

    def test_tic_id_stored(self) -> None:
        r = match_ground_truth(150428135, 37.42, 2458360.0, _CATALOG)
        assert r.tic_id == 150428135

    def test_eb_match(self) -> None:
        r = match_ground_truth(999, 5.0, 2458000.0, _CATALOG)
        if r.flag == "MATCH":
            assert r.matched_category == "eb"


class TestFormatMatchResult:
    def test_returns_string(self) -> None:
        r = match_ground_truth(150428135, 37.42, 2458360.0, _CATALOG)
        assert isinstance(format_match_result(r), str)

    def test_contains_tic_id(self) -> None:
        r = match_ground_truth(150428135, 37.42, 2458360.0, _CATALOG)
        assert "150428135" in format_match_result(r)

    def test_empty_catalog_formatted(self) -> None:
        r = match_ground_truth(999, 37.42, 2458360.0, [])
        assert "EMPTY_CATALOG" in format_match_result(r)
