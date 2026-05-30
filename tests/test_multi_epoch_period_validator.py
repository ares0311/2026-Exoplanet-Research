"""Tests for Skills/multi_epoch_period_validator.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from multi_epoch_period_validator import format_epoch_validation_result, validate_period_multi_epoch


class TestValidatePeriodMultiEpoch:
    def test_consistent_perfect(self) -> None:
        # Two seasons with exact period
        s1 = [0.0, 5.0, 10.0]
        s2 = [100.0, 105.0, 110.0]
        r = validate_period_multi_epoch(5.0, [s1, s2])
        assert r.consistent is True
        assert r.flag == "CONSISTENT"

    def test_inconsistent_wrong_period(self) -> None:
        s1 = [0.0, 5.0, 10.0]
        s2 = [100.0, 106.0, 112.0]  # actual period=6, not 5
        r = validate_period_multi_epoch(5.0, [s1, s2], tolerance_minutes=5.0)
        assert r.consistent is False

    def test_invalid_period(self) -> None:
        r = validate_period_multi_epoch(0.0, [[0.0, 5.0], [10.0, 15.0]])
        assert r.flag == "INVALID_PERIOD"

    def test_insufficient_epochs(self) -> None:
        r = validate_period_multi_epoch(5.0, [[0.0, 5.0, 10.0]])
        assert r.flag == "INSUFFICIENT_EPOCHS"

    def test_n_transits_total_correct(self) -> None:
        s1 = [0.0, 5.0]
        s2 = [100.0, 105.0]
        r = validate_period_multi_epoch(5.0, [s1, s2])
        assert r.n_transits_total == 4

    def test_n_epochs_correct(self) -> None:
        r = validate_period_multi_epoch(5.0, [[0.0, 5.0], [100.0, 105.0], [200.0, 205.0]])
        assert r.n_epochs == 3

    def test_rms_oc_float(self) -> None:
        r = validate_period_multi_epoch(5.0, [[0.0, 5.0], [100.0, 105.0]])
        assert isinstance(r.rms_oc_minutes, float)

    def test_max_oc_ge_rms(self) -> None:
        r = validate_period_multi_epoch(5.0, [[0.0, 5.0], [100.0, 105.0]])
        assert r.max_oc_minutes >= r.rms_oc_minutes

    def test_insufficient_transits(self) -> None:
        r = validate_period_multi_epoch(5.0, [[0.0], []])
        assert r.flag == "INSUFFICIENT_TRANSITS"

    def test_custom_tolerance(self) -> None:
        s1 = [0.0, 5.0]
        s2 = [100.0, 105.0]
        r = validate_period_multi_epoch(5.0, [s1, s2], tolerance_minutes=0.001)
        assert r.consistent is True  # perfect data

    def test_period_stored(self) -> None:
        r = validate_period_multi_epoch(7.3, [[0.0, 7.3], [100.0, 107.3]])
        assert r.period_days == 7.3

    def test_format_returns_string(self) -> None:
        r = validate_period_multi_epoch(5.0, [[0.0, 5.0], [100.0, 105.0]])
        s = format_epoch_validation_result(r)
        assert isinstance(s, str)
        assert "Period" in s
