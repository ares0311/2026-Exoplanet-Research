"""Tests for Skills/snr_sector_stacker.py."""
import math

from Skills.snr_sector_stacker import (
    StackedSNRResult,
    format_stacked_snr_result,
    project_stacked_snr,
)


class TestProjectStackedSnr:
    def test_returns_result_type(self):
        result = project_stacked_snr(5.0, 4)
        assert isinstance(result, StackedSNRResult)

    def test_flag_ok(self):
        result = project_stacked_snr(5.0, 4)
        assert result.flag == "OK"

    def test_sqrt_n_scaling(self):
        result = project_stacked_snr(5.0, 4)
        assert abs(result.snr_stacked - 5.0 * math.sqrt(4)) < 1e-6

    def test_single_sector_no_gain(self):
        result = project_stacked_snr(7.0, 1)
        assert abs(result.snr_stacked - 7.0) < 1e-6
        assert abs(result.snr_gain - 1.0) < 1e-6

    def test_snr_gain_is_sqrt_n(self):
        result = project_stacked_snr(3.0, 9)
        assert abs(result.snr_gain - 3.0) < 1e-6

    def test_sectors_for_threshold_computed(self):
        # Need SNR >= 7.1; start at 5.0 → need ceil((7.1/5.0)^2) ≈ 3 sectors
        result = project_stacked_snr(5.0, 10, snr_threshold=7.1)
        assert result.sectors_for_threshold is not None
        # stacked_snr(n) = 5*sqrt(n) >= 7.1 → n >= 2.02 → 3
        assert result.sectors_for_threshold >= 1

    def test_already_above_threshold(self):
        result = project_stacked_snr(10.0, 3, snr_threshold=7.1)
        assert result.sectors_for_threshold == 1

    def test_invalid_snr_negative(self):
        result = project_stacked_snr(-1.0, 4)
        assert result.flag == "INVALID"

    def test_invalid_n_sectors_zero(self):
        result = project_stacked_snr(5.0, 0)
        assert result.flag == "INVALID"

    def test_heterogeneous_noise(self):
        result = project_stacked_snr(5.0, 3, per_sector_noise=[1.0, 1.5, 0.8])
        assert result.flag == "OK"
        assert result.snr_stacked > 0

    def test_snr_single_stored(self):
        result = project_stacked_snr(6.5, 4)
        assert result.snr_single == 6.5

    def test_n_sectors_stored(self):
        result = project_stacked_snr(5.0, 7)
        assert result.n_sectors == 7


class TestFormatStackedSnrResult:
    def test_returns_string(self):
        result = project_stacked_snr(5.0, 4)
        s = format_stacked_snr_result(result)
        assert isinstance(s, str)

    def test_contains_snr(self):
        result = project_stacked_snr(5.0, 4)
        s = format_stacked_snr_result(result)
        assert "SNR" in s or "snr" in s.lower()
