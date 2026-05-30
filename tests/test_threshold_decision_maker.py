"""Tests for Skills/threshold_decision_maker.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from threshold_decision_maker import format_decision_result, make_threshold_decision


class TestMakeThresholdDecision:
    def test_go_all_pass(self) -> None:
        r = make_threshold_decision(0.05, 10.0, 0.90, 3)
        assert r.go is True
        assert r.flag == "GO"

    def test_no_go_high_fpp(self) -> None:
        r = make_threshold_decision(0.50, 10.0, 0.90, 3)
        assert r.go is False
        assert r.flag == "NO_GO"

    def test_no_go_low_snr(self) -> None:
        r = make_threshold_decision(0.05, 5.0, 0.90, 3)
        assert r.go is False

    def test_no_go_low_dc(self) -> None:
        r = make_threshold_decision(0.05, 10.0, 0.50, 3)
        assert r.go is False

    def test_no_go_few_transits(self) -> None:
        r = make_threshold_decision(0.05, 10.0, 0.90, 1)
        assert r.go is False

    def test_failed_criteria_populated(self) -> None:
        r = make_threshold_decision(0.50, 5.0, 0.50, 1)
        assert len(r.failed_criteria) >= 3

    def test_invalid_fpp(self) -> None:
        r = make_threshold_decision(1.5, 10.0, 0.90, 3)
        assert r.flag == "INVALID_INPUT"

    def test_invalid_dc(self) -> None:
        r = make_threshold_decision(0.05, 10.0, -0.1, 3)
        assert r.flag == "INVALID_INPUT"

    def test_custom_thresholds(self) -> None:
        r = make_threshold_decision(0.15, 6.0, 0.75, 2,
                                     fpp_max=0.20, snr_min=5.0, dc_min=0.70, n_transits_min=2)
        assert r.go is True

    def test_fpp_stored(self) -> None:
        r = make_threshold_decision(0.05, 10.0, 0.90, 3)
        assert r.fpp == 0.05

    def test_n_transits_stored(self) -> None:
        r = make_threshold_decision(0.05, 10.0, 0.90, 5)
        assert r.n_transits == 5

    def test_format_returns_string(self) -> None:
        r = make_threshold_decision(0.05, 10.0, 0.90, 3)
        s = format_decision_result(r)
        assert isinstance(s, str)
        assert "GO" in s
