"""Tests for stellar_contamination_scorer.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from stellar_contamination_scorer import (
    format_contamination_result,
    score_contamination,
)


class TestScoreContamination:
    def test_result_frozen(self):
        r = score_contamination(0.2, 3, [0.1, 0.2])
        try:
            r.composite_score = 99.0  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except (AttributeError, TypeError):
            pass

    def test_clean_star(self):
        r = score_contamination(0.0, 0, [])
        assert r.composite_score == 0.0
        assert not r.is_contaminated
        assert r.flag == "OK"

    def test_highly_contaminated(self):
        r = score_contamination(0.9, 5, [1.0, 2.0, 3.0])
        assert r.composite_score > 0.5
        assert r.is_contaminated
        assert r.flag == "OK"

    def test_contratio_only(self):
        r = score_contamination(0.5, None, None)
        assert r.flag == "OK"
        assert r.crowdsap_score == 0.5
        assert r.neighbour_score is None

    def test_n_contaminants_only(self):
        r = score_contamination(None, 3, None)
        assert r.flag == "OK"
        assert r.neighbour_score is not None

    def test_flux_ratios_only(self):
        r = score_contamination(None, None, [0.5, 0.3])
        assert r.flag == "OK"
        assert r.dilution_score is not None

    def test_no_inputs_insufficient(self):
        r = score_contamination(None, None, None)
        assert r.flag == "INSUFFICIENT"

    def test_invalid_contratio_negative(self):
        r = score_contamination(-0.1, 0, None)
        assert r.flag == "INVALID"

    def test_invalid_contratio_gt_1(self):
        r = score_contamination(1.5, 0, None)
        assert r.flag == "INVALID"

    def test_invalid_n_contaminants_negative(self):
        r = score_contamination(None, -1, None)
        assert r.flag == "INVALID"

    def test_invalid_flux_ratio_negative(self):
        r = score_contamination(None, None, [-0.1, 0.5])
        assert r.flag == "INVALID"

    def test_neighbour_score_saturates_at_5(self):
        r5 = score_contamination(None, 5, None)
        r10 = score_contamination(None, 10, None)
        assert r5.neighbour_score == r10.neighbour_score == 1.0

    def test_threshold_controls_is_contaminated(self):
        r = score_contamination(0.1, None, None, contamination_threshold=0.5)
        assert not r.is_contaminated
        r2 = score_contamination(0.1, None, None, contamination_threshold=0.05)
        assert r2.is_contaminated

    def test_format_returns_string(self):
        r = score_contamination(0.2, 2, [0.1])
        s = format_contamination_result(r)
        assert isinstance(s, str)
        assert "Contamination" in s

    def test_composite_score_in_range(self):
        r = score_contamination(0.3, 2, [0.2])
        assert 0.0 <= r.composite_score <= 1.0
