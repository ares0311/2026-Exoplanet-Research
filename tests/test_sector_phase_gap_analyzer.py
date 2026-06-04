"""Tests for Skills/sector_phase_gap_analyzer.py."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from sector_phase_gap_analyzer import (
    analyze_phase_gaps,
    format_phase_gap_result,
)


class TestAnalyzePhaseGaps:
    def _uniform_phase(self, n: int = 100) -> list[float]:
        return [(-0.5 + i / (n - 1)) for i in range(n)]

    def test_uniform_coverage(self) -> None:
        phase = self._uniform_phase(200)
        r = analyze_phase_gaps(phase)
        assert r.flag == "OK"
        assert r.coverage_fraction > 0.9

    def test_no_gaps_full_coverage(self) -> None:
        phase = self._uniform_phase(500)
        r = analyze_phase_gaps(phase, n_bins=50)
        assert r.coverage_fraction == 1.0

    def test_insufficient_data(self) -> None:
        r = analyze_phase_gaps([0.0])
        assert r.flag == "INSUFFICIENT_DATA"

    def test_invalid_n_bins(self) -> None:
        r = analyze_phase_gaps(self._uniform_phase(), n_bins=1)
        assert r.flag == "INVALID_N_BINS"

    def test_coverage_fraction_range(self) -> None:
        r = analyze_phase_gaps(self._uniform_phase(50))
        assert 0.0 <= r.coverage_fraction <= 1.0

    def test_gap_with_missing_half(self) -> None:
        phase = [p for p in self._uniform_phase(100) if p < 0.0]
        r = analyze_phase_gaps(phase, n_bins=20)
        assert r.flag == "OK"
        assert r.coverage_fraction < 0.6

    def test_largest_gap_nonneg(self) -> None:
        phase = self._uniform_phase(100)
        r = analyze_phase_gaps(phase)
        assert r.largest_gap_width >= 0

    def test_n_points_matches(self) -> None:
        phase = self._uniform_phase(80)
        r = analyze_phase_gaps(phase)
        assert r.n_points == 80

    def test_n_bins_preserved(self) -> None:
        r = analyze_phase_gaps(self._uniform_phase(), n_bins=30)
        assert r.n_bins == 30

    def test_gap_phases_in_range(self) -> None:
        phase = self._uniform_phase(100)
        r = analyze_phase_gaps(phase)
        for gp in r.gap_phases:
            assert -0.5 <= gp <= 0.5

    def test_format_output(self) -> None:
        r = analyze_phase_gaps(self._uniform_phase())
        s = format_phase_gap_result(r)
        assert "|" in s
        assert "Coverage" in s or "coverage" in s

    def test_empty_phase(self) -> None:
        r = analyze_phase_gaps([])
        assert r.flag == "INSUFFICIENT_DATA"
