"""Tests for Skills/fpp_bayesian_updater.py"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from fpp_bayesian_updater import format_fpp_update, update_fpp


class TestFppBayesianUpdater:
    def test_lr_one_no_change(self) -> None:
        r = update_fpp(0.10, 1.0)
        assert r.flag == "OK"
        assert abs(r.fpp_posterior - 0.10) < 1e-6

    def test_lr_gt1_increases_fpp(self) -> None:
        r = update_fpp(0.10, 10.0)
        assert r.flag == "OK"
        assert r.fpp_posterior > 0.10

    def test_lr_lt1_decreases_fpp(self) -> None:
        r = update_fpp(0.10, 0.1)
        assert r.flag == "OK"
        assert r.fpp_posterior < 0.10

    def test_log_bayes_factor_positive_for_lr_gt1(self) -> None:
        r = update_fpp(0.05, 100.0)
        assert r.log_bayes_factor > 0.0

    def test_log_bayes_factor_negative_for_lr_lt1(self) -> None:
        r = update_fpp(0.05, 0.01)
        assert r.log_bayes_factor < 0.0

    def test_posterior_stays_in_0_1(self) -> None:
        r = update_fpp(0.50, 1000.0)
        assert 0.0 < r.fpp_posterior < 1.0

    def test_invalid_fpp_zero(self) -> None:
        r = update_fpp(0.0, 1.0)
        assert r.flag == "INVALID_FPP_PRIOR"
        assert math.isnan(r.fpp_posterior)

    def test_invalid_fpp_one(self) -> None:
        r = update_fpp(1.0, 1.0)
        assert r.flag == "INVALID_FPP_PRIOR"

    def test_invalid_lr_zero(self) -> None:
        r = update_fpp(0.10, 0.0)
        assert r.flag == "INVALID_LIKELIHOOD_RATIO"

    def test_invalid_lr_negative(self) -> None:
        r = update_fpp(0.10, -1.0)
        assert r.flag == "INVALID_LIKELIHOOD_RATIO"

    def test_evidence_label_stored(self) -> None:
        r = update_fpp(0.10, 2.0, evidence_label="odd_even_test")
        assert r.evidence_label == "odd_even_test"

    def test_format_returns_string(self) -> None:
        r = update_fpp(0.10, 2.0)
        s = format_fpp_update(r)
        assert isinstance(s, str)
        assert "FPP" in s
