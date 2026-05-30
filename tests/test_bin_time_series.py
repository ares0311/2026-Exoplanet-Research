"""Tests for Skills/bin_time_series.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from bin_time_series import bin_time_series, format_bin_result


class TestBinTimeSeries:
    def test_basic_ok(self) -> None:
        t = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        f = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        r = bin_time_series(t, f, 0.2)
        assert r.flag == "OK"

    def test_bins_non_empty(self) -> None:
        t = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        f = [1.0] * 6
        r = bin_time_series(t, f, 0.2)
        assert r.n_bins > 0

    def test_bin_flux_mean_correct(self) -> None:
        t = [0.0, 0.05, 0.1, 0.15]
        f = [1.0, 3.0, 2.0, 4.0]
        r = bin_time_series(t, f, 0.1)
        assert abs(r.bin_flux[0] - 2.0) < 1e-9

    def test_insufficient_data(self) -> None:
        r = bin_time_series([0.0], [1.0], 0.1)
        assert r.flag == "INSUFFICIENT_DATA"

    def test_invalid_bin_width(self) -> None:
        r = bin_time_series([0.0, 1.0], [1.0, 1.0], 0.0)
        assert r.flag == "INVALID_BIN_WIDTH"

    def test_rms_zero_for_uniform(self) -> None:
        t = [0.0, 0.05]
        f = [1.0, 1.0]
        r = bin_time_series(t, f, 0.1)
        assert r.bin_rms[0] == 0.0

    def test_bin_counts_positive(self) -> None:
        t = [0.0, 0.1, 0.2]
        f = [1.0, 1.0, 1.0]
        r = bin_time_series(t, f, 0.15)
        assert all(c > 0 for c in r.bin_counts)

    def test_n_input_correct(self) -> None:
        t = list(range(10))
        f = [1.0] * 10
        r = bin_time_series(t, f, 2.0)
        assert r.n_input == 10

    def test_bin_width_stored(self) -> None:
        t = [0.0, 1.0, 2.0]
        f = [1.0, 1.0, 1.0]
        r = bin_time_series(t, f, 0.5)
        assert r.bin_width_days == 0.5

    def test_empty_list_insufficient(self) -> None:
        r = bin_time_series([], [], 0.1)
        assert r.flag == "INSUFFICIENT_DATA"

    def test_negative_bin_width(self) -> None:
        r = bin_time_series([0.0, 1.0], [1.0, 1.0], -0.1)
        assert r.flag == "INVALID_BIN_WIDTH"

    def test_format_returns_string(self) -> None:
        t = [0.0, 0.1, 0.2, 0.3]
        f = [1.0] * 4
        r = bin_time_series(t, f, 0.15)
        s = format_bin_result(r)
        assert isinstance(s, str)
        assert "Bin" in s
