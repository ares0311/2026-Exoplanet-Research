"""Tests for Skills/rv_mass_upper_limit.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from rv_mass_upper_limit import (
    compute_rv_upper_limit,
    format_rv_mass_limit,
)


class TestComputeRvUpperLimit:
    def test_basic_calculation(self) -> None:
        r = compute_rv_upper_limit(1.0, 10.0, 1.0)
        assert r.flag == "OK"
        assert r.mass_upper_limit_mjup > 0.0

    def test_flag_ok(self) -> None:
        r = compute_rv_upper_limit(1.0, 10.0, 1.0)
        assert r.flag == "OK"

    def test_k_amplitude_equals_precision_times_n_sigma(self) -> None:
        r = compute_rv_upper_limit(2.0, 10.0, 1.0, n_sigma=3.0)
        assert abs(r.k_amplitude_ms - 6.0) < 1e-9

    def test_invalid_precision(self) -> None:
        r = compute_rv_upper_limit(0.0, 10.0, 1.0)
        assert r.flag == "INVALID_PRECISION"

    def test_negative_precision_invalid(self) -> None:
        r = compute_rv_upper_limit(-1.0, 10.0, 1.0)
        assert r.flag == "INVALID_PRECISION"

    def test_invalid_period(self) -> None:
        r = compute_rv_upper_limit(1.0, 0.0, 1.0)
        assert r.flag == "INVALID_PERIOD"

    def test_invalid_stellar_mass(self) -> None:
        r = compute_rv_upper_limit(1.0, 10.0, 0.0)
        assert r.flag == "INVALID_STELLAR_MASS"

    def test_n_sigma_1_smaller_than_3(self) -> None:
        r1 = compute_rv_upper_limit(1.0, 10.0, 1.0, n_sigma=1.0)
        r3 = compute_rv_upper_limit(1.0, 10.0, 1.0, n_sigma=3.0)
        assert r1.mass_upper_limit_mjup < r3.mass_upper_limit_mjup

    def test_longer_period_larger_limit(self) -> None:
        r_short = compute_rv_upper_limit(1.0, 10.0, 1.0)
        r_long = compute_rv_upper_limit(1.0, 100.0, 1.0)
        assert r_long.mass_upper_limit_mjup > r_short.mass_upper_limit_mjup

    def test_period_stored(self) -> None:
        r = compute_rv_upper_limit(1.0, 7.3, 1.0)
        assert r.period_days == 7.3

    def test_mass_proportional_to_precision(self) -> None:
        r1 = compute_rv_upper_limit(1.0, 10.0, 1.0)
        r2 = compute_rv_upper_limit(2.0, 10.0, 1.0)
        assert abs(r2.mass_upper_limit_mjup / r1.mass_upper_limit_mjup - 2.0) < 1e-6

    def test_format_returns_string(self) -> None:
        r = compute_rv_upper_limit(1.0, 10.0, 1.0)
        s = format_rv_mass_limit(r)
        assert isinstance(s, str)
        assert "M_Jup" in s or "Mass" in s
