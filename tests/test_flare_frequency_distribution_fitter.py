"""Tests for Skills/flare_frequency_distribution_fitter.py"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from flare_frequency_distribution_fitter import (
    fit_flare_frequency_distribution,
    format_ffd_fit_result,
)


def _make_power_law_flares(n: int = 20, alpha: float = -1.5, e_min: float = 30.0) -> list[float]:
    import random
    random.seed(42)
    e_max = e_min + 3.0
    return [random.uniform(e_min, e_max) for _ in range(n)]


class TestFitFlareFrequencyDistribution:
    def test_ok_flag(self) -> None:
        log_e = _make_power_law_flares()
        r = fit_flare_frequency_distribution(log_e, observing_time_days=100.0)
        assert r.flag == "OK"

    def test_alpha_is_finite(self) -> None:
        log_e = _make_power_law_flares()
        r = fit_flare_frequency_distribution(log_e, observing_time_days=100.0)
        assert math.isfinite(r.alpha)

    def test_rate_at_e0_positive(self) -> None:
        log_e = _make_power_law_flares()
        r = fit_flare_frequency_distribution(log_e, observing_time_days=100.0)
        assert r.rate_at_e0 > 0

    def test_r_squared_in_range(self) -> None:
        log_e = _make_power_law_flares()
        r = fit_flare_frequency_distribution(log_e, observing_time_days=100.0)
        assert -1.0 <= r.r_squared <= 1.0

    def test_n_flares_matches_input_above_threshold(self) -> None:
        log_e = [30.5, 31.0, 31.5, 32.0, 32.5]
        r = fit_flare_frequency_distribution(log_e, 50.0, completeness_log_energy=31.0)
        assert r.n_flares == 4

    def test_insufficient_flares(self) -> None:
        r = fit_flare_frequency_distribution([30.5, 31.0], observing_time_days=100.0)
        assert r.flag == "INSUFFICIENT_FLARES"

    def test_invalid_baseline(self) -> None:
        log_e = _make_power_law_flares()
        r = fit_flare_frequency_distribution(log_e, observing_time_days=0.0)
        assert r.flag == "INVALID_BASELINE"

    def test_completeness_threshold_used(self) -> None:
        log_e = [30.0, 30.5, 31.0, 31.5, 32.0]
        r = fit_flare_frequency_distribution(log_e, 50.0, completeness_log_energy=31.0)
        assert r.completeness_threshold_log_energy == 31.0
        assert r.n_flares == 3

    def test_longer_baseline_lower_rate(self) -> None:
        log_e = _make_power_law_flares()
        r1 = fit_flare_frequency_distribution(log_e, observing_time_days=10.0)
        r2 = fit_flare_frequency_distribution(log_e, observing_time_days=100.0)
        assert r1.rate_at_e0 > r2.rate_at_e0

    def test_result_frozen(self) -> None:
        log_e = _make_power_law_flares()
        r = fit_flare_frequency_distribution(log_e, observing_time_days=100.0)
        try:
            r.alpha = 0.0  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except (AttributeError, TypeError):
            pass

    def test_format_returns_string(self) -> None:
        log_e = _make_power_law_flares()
        r = fit_flare_frequency_distribution(log_e, observing_time_days=100.0)
        s = format_ffd_fit_result(r)
        assert isinstance(s, str)
        assert r.flag in s
