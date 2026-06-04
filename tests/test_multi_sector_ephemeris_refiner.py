"""Tests for Skills/multi_sector_ephemeris_refiner.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from multi_sector_ephemeris_refiner import (
    format_ephemeris_refinement_result,
    refine_ephemeris,
)


def _perfect_transits(period: float = 3.0, epoch: float = 2457000.0, n: int = 10) -> list[float]:
    return [epoch + i * period for i in range(n)]


class TestRefineEphemeris:
    def test_ok_flag(self) -> None:
        mids = _perfect_transits()
        r = refine_ephemeris(mids, 3.0)
        assert r.flag == "OK"

    def test_perfect_transits_rms_zero(self) -> None:
        mids = _perfect_transits()
        r = refine_ephemeris(mids, 3.0)
        assert r.rms_oc_minutes < 1e-6

    def test_refined_period_matches_input(self) -> None:
        mids = _perfect_transits(period=5.0)
        r = refine_ephemeris(mids, 5.0)
        assert abs(r.refined_period_days - 5.0) < 1e-6

    def test_refined_epoch_matches_first_midpoint(self) -> None:
        mids = _perfect_transits(epoch=2457100.0)
        r = refine_ephemeris(mids, 3.0, initial_epoch_bjd=2457100.0)
        assert abs(r.refined_epoch_bjd - 2457100.0) < 1e-6

    def test_n_transits_correct(self) -> None:
        mids = _perfect_transits(n=8)
        r = refine_ephemeris(mids, 3.0)
        assert r.n_transits == 8

    def test_noisy_transits_rms_nonzero(self) -> None:
        import random
        random.seed(42)
        mids = [t + random.gauss(0, 0.001) for t in _perfect_transits()]
        r = refine_ephemeris(mids, 3.0)
        assert r.rms_oc_minutes > 0

    def test_period_uncertainty_finite(self) -> None:
        mids = _perfect_transits()
        r = refine_ephemeris(mids, 3.0)
        assert r.period_uncertainty_days >= 0

    def test_initial_period_offset_corrected(self) -> None:
        mids = _perfect_transits(period=3.0)
        r = refine_ephemeris(mids, 3.01)
        assert abs(r.refined_period_days - 3.0) < 0.01

    def test_insufficient_transits(self) -> None:
        r = refine_ephemeris([2457000.0], 3.0)
        assert r.flag == "INSUFFICIENT_TRANSITS"

    def test_invalid_period(self) -> None:
        mids = _perfect_transits()
        r = refine_ephemeris(mids, 0.0)
        assert r.flag == "INVALID_PERIOD"

    def test_weighted_fit_with_errors(self) -> None:
        mids = _perfect_transits()
        errs = [0.001] * len(mids)
        r = refine_ephemeris(mids, 3.0, midpoint_errors_days=errs)
        assert r.flag == "OK"
        assert r.rms_oc_minutes < 1e-6

    def test_result_frozen(self) -> None:
        mids = _perfect_transits()
        r = refine_ephemeris(mids, 3.0)
        try:
            r.refined_period_days = 0.0  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except (AttributeError, TypeError):
            pass

    def test_format_returns_string(self) -> None:
        mids = _perfect_transits()
        r = refine_ephemeris(mids, 3.0)
        s = format_ephemeris_refinement_result(r)
        assert isinstance(s, str)
        assert r.flag in s
