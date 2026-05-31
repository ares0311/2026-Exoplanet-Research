"""Tests for Skills/period_power_spectrum_combiner.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from period_power_spectrum_combiner import combine_power_spectra, format_combined_spectrum_result


class TestCombinePowerSpectra:
    def test_basic_combination(self) -> None:
        periods = [1.0, 2.0, 3.0]
        powers = [[0.1, 0.9, 0.2], [0.2, 0.8, 0.1]]
        r = combine_power_spectra(periods, powers)
        assert r.flag == "OK"
        assert r.peak_period == 2.0

    def test_single_spectrum(self) -> None:
        periods = [1.0, 2.0, 3.0]
        powers = [[0.1, 0.5, 0.9]]
        r = combine_power_spectra(periods, powers)
        assert r.peak_period == 3.0

    def test_insufficient_periods(self) -> None:
        r = combine_power_spectra([1.0], [[0.5]])
        assert r.flag == "INSUFFICIENT_PERIODS"

    def test_no_spectra(self) -> None:
        r = combine_power_spectra([1.0, 2.0], [])
        assert r.flag == "NO_SPECTRA"

    def test_length_mismatch(self) -> None:
        r = combine_power_spectra([1.0, 2.0, 3.0], [[0.1, 0.9]])
        assert r.flag == "LENGTH_MISMATCH"

    def test_n_spectra_correct(self) -> None:
        periods = [1.0, 2.0, 3.0]
        powers = [[0.1, 0.5, 0.9], [0.2, 0.3, 0.8]]
        r = combine_power_spectra(periods, powers)
        assert r.n_spectra == 2

    def test_n_periods_correct(self) -> None:
        periods = [1.0, 2.0, 3.0, 4.0]
        powers = [[0.1, 0.2, 0.3, 0.9]]
        r = combine_power_spectra(periods, powers)
        assert r.n_periods == 4

    def test_peak_power_positive(self) -> None:
        periods = [1.0, 2.0]
        powers = [[0.3, 0.7], [0.4, 0.6]]
        r = combine_power_spectra(periods, powers)
        assert r.peak_power > 0.0

    def test_no_normalize_option(self) -> None:
        periods = [1.0, 2.0, 3.0]
        powers = [[1.0, 10.0, 2.0]]
        r = combine_power_spectra(periods, powers, normalize=False)
        assert r.flag == "OK"
        assert r.peak_period == 2.0

    def test_peak_index_in_range(self) -> None:
        periods = [1.0, 2.0, 3.0]
        powers = [[0.9, 0.1, 0.5]]
        r = combine_power_spectra(periods, powers)
        assert 0 <= r.peak_index < 3

    def test_flat_spectrum_any_peak(self) -> None:
        periods = [1.0, 2.0, 3.0]
        powers = [[1.0, 1.0, 1.0]]
        r = combine_power_spectra(periods, powers)
        assert r.flag == "OK"

    def test_format_returns_string(self) -> None:
        periods = [1.0, 2.0, 3.0]
        powers = [[0.1, 0.9, 0.2]]
        r = combine_power_spectra(periods, powers)
        s = format_combined_spectrum_result(r)
        assert isinstance(s, str)
        assert "Peak" in s
