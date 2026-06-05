"""Tests for Skills/planetary_resonance_chain_analyzer.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from planetary_resonance_chain_analyzer import (
    analyze_resonance_chain,
    format_resonance_chain_result,
)


class TestPlanetaryResonanceChainAnalyzer:
    def test_2_1_resonance_detected(self) -> None:
        # Period ratio exactly 2:1
        r = analyze_resonance_chain([4.0, 8.0])
        assert r.flag == "OK"
        assert len(r.pairs) == 1
        assert r.pairs[0].nearest_p == 2
        assert r.pairs[0].nearest_q == 1
        assert r.pairs[0].near_resonant is True

    def test_3_2_resonance_detected(self) -> None:
        r = analyze_resonance_chain([6.0, 9.0])
        assert r.flag == "OK"
        assert r.pairs[0].nearest_p == 3
        assert r.pairs[0].nearest_q == 2
        assert r.pairs[0].near_resonant is True

    def test_non_resonant_chain(self) -> None:
        r = analyze_resonance_chain([1.0, 7.3, 45.2])
        assert r.flag == "OK"
        assert r.chain_string == "non_resonant"

    def test_laplace_resonance(self) -> None:
        # Io:Europa:Ganymede ≈ 1:2:4 (P = 1.77, 3.55, 7.15 d)
        r = analyze_resonance_chain([1.77, 3.55, 7.15])
        assert r.flag == "OK"
        assert r.n_near_resonant >= 2

    def test_insufficient_planets(self) -> None:
        r = analyze_resonance_chain([5.0])
        assert r.flag == "INSUFFICIENT_PLANETS"

    def test_invalid_period(self) -> None:
        r = analyze_resonance_chain([5.0, -1.0])
        assert r.flag == "INVALID_PERIOD"

    def test_n_near_resonant_count(self) -> None:
        r = analyze_resonance_chain([4.0, 8.0, 16.0])
        assert r.n_near_resonant == len([p for p in r.pairs if p.near_resonant])

    def test_pairs_count_equals_n_minus_one(self) -> None:
        r = analyze_resonance_chain([1.0, 2.0, 4.0, 8.0])
        assert len(r.pairs) == 3

    def test_period_ratio_stored(self) -> None:
        r = analyze_resonance_chain([4.0, 8.0])
        assert abs(r.pairs[0].period_ratio - 2.0) < 0.01

    def test_deviation_percent_near_zero_for_exact_mmr(self) -> None:
        r = analyze_resonance_chain([4.0, 8.0])
        assert r.pairs[0].deviation_percent < 0.01

    def test_result_is_frozen(self) -> None:
        r = analyze_resonance_chain([4.0, 8.0])
        try:
            r.chain_string = "X"  # type: ignore[misc]
            raise AssertionError("frozen dataclass should not allow setattr")
        except Exception:
            pass

    def test_format_ok(self) -> None:
        r = analyze_resonance_chain([4.0, 8.0])
        s = format_resonance_chain_result(r)
        assert "Chain" in s or "chain" in s

    def test_format_error(self) -> None:
        r = analyze_resonance_chain([5.0])
        s = format_resonance_chain_result(r)
        assert "INSUFFICIENT" in s
