"""Tests for Skills.eb_classifier."""
from __future__ import annotations

from Skills.eb_classifier import (
    EBClassifierResult,
    classify_eb,
    format_eb_result,
)


class TestClassifyEB:
    def test_returns_result(self) -> None:
        r = classify_eb(1000.0)
        assert isinstance(r, EBClassifierResult)

    def test_ok_flag(self) -> None:
        r = classify_eb(1000.0)
        assert r.flag == "OK"

    def test_zero_depth_insufficient(self) -> None:
        r = classify_eb(0.0)
        assert r.flag == "INSUFFICIENT"

    def test_planet_candidate_low_depth(self) -> None:
        r = classify_eb(1000.0)
        assert r.classification == "planet_candidate"

    def test_likely_eb_very_deep(self) -> None:
        r = classify_eb(60000.0)
        assert r.classification in {"likely_eb", "possible_eb"}

    def test_secondary_eclipse_raises_eb_probability(self) -> None:
        r_no_sec = classify_eb(1000.0)
        r_with_sec = classify_eb(1000.0, secondary_depth_ppm=1000.0)
        assert r_with_sec.eb_probability >= r_no_sec.eb_probability

    def test_odd_even_raises_eb_probability(self) -> None:
        r_no_oe = classify_eb(1000.0)
        r_with_oe = classify_eb(1000.0, odd_even_sigma=5.0)
        assert r_with_oe.eb_probability >= r_no_oe.eb_probability

    def test_rho_ratio_raises_eb_probability(self) -> None:
        r_no_rho = classify_eb(1000.0)
        r_with_rho = classify_eb(1000.0, rho_ratio=5.0)
        assert r_with_rho.eb_probability >= r_no_rho.eb_probability

    def test_eb_probability_in_range(self) -> None:
        r = classify_eb(5000.0, secondary_depth_ppm=1000.0, odd_even_sigma=4.0)
        assert 0.0 <= r.eb_probability <= 1.0

    def test_reasons_populated_for_deep_transit(self) -> None:
        r = classify_eb(60000.0)
        assert len(r.reasons) >= 1

    def test_no_reasons_for_clean_candidate(self) -> None:
        r = classify_eb(500.0)
        assert len(r.reasons) == 0

    def test_classification_values_valid(self) -> None:
        r = classify_eb(1000.0)
        assert r.classification in {"planet_candidate", "possible_eb", "likely_eb"}


class TestFormatEBResult:
    def test_returns_string(self) -> None:
        r = classify_eb(1000.0)
        assert isinstance(format_eb_result(r), str)

    def test_contains_classification(self) -> None:
        r = classify_eb(1000.0)
        assert r.classification in format_eb_result(r)

    def test_insufficient_formatted(self) -> None:
        r = classify_eb(0.0)
        out = format_eb_result(r)
        assert "INSUFFICIENT" in out
