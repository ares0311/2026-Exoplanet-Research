"""Tests for planet_habitability_scorer.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from planet_habitability_scorer import (
    format_habitability_result,
    score_habitability,
)


class TestScoreHabitability:
    def test_earth_like(self):
        r = score_habitability(255.0, 1.0, 5778.0)
        assert r.flag == "OK"
        assert r.classification in ("potentially_habitable", "marginal")

    def test_hot_jupiter_uninhabitable(self):
        r = score_habitability(1500.0, 12.0, 6000.0)
        assert r.classification == "uninhabitable"

    def test_cold_planet_uninhabitable(self):
        # T_eq=50K → very low teq_score, but radius and stellar scores are high
        r = score_habitability(50.0, 1.0, 5778.0)
        assert r.classification in ("uninhabitable", "marginal")

    def test_all_none_insufficient(self):
        r = score_habitability(None, None, None)
        assert r.flag == "INSUFFICIENT"

    def test_teq_only(self):
        r = score_habitability(255.0, None, None)
        assert r.flag == "OK"
        assert r.teq_score is not None
        assert r.radius_score is None

    def test_radius_only(self):
        r = score_habitability(None, 1.0, None)
        assert r.flag == "OK"
        assert r.radius_score is not None

    def test_k_dwarf_high_stellar_score(self):
        r = score_habitability(255.0, 1.0, 4500.0)
        assert r.stellar_score is not None
        assert r.stellar_score == 1.0

    def test_hot_star_low_stellar_score(self):
        r = score_habitability(255.0, 1.0, 8000.0)
        assert r.stellar_score is not None
        assert r.stellar_score == 0.0

    def test_composite_in_range(self):
        r = score_habitability(300.0, 1.5, 5000.0)
        assert 0.0 <= r.composite_score <= 1.0

    def test_result_frozen(self):
        r = score_habitability(255.0, 1.0, 5778.0)
        try:
            r.composite_score = 99.0  # type: ignore[misc]
            raise AssertionError()
        except (AttributeError, TypeError):
            pass

    def test_invalid_weights_zero(self):
        r = score_habitability(
            255.0, 1.0, 5778.0, teq_weight=0.0, radius_weight=0.0, stellar_weight=0.0
        )
        assert r.flag == "INVALID"


class TestFormatHabitabilityResult:
    def test_returns_string(self):
        r = score_habitability(255.0, 1.0, 5778.0)
        assert isinstance(format_habitability_result(r), str)

    def test_contains_flag(self):
        r = score_habitability(255.0, 1.0, 5778.0)
        assert r.flag in format_habitability_result(r)
