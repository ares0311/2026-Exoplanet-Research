"""Tests for Skills/toi_alert_formatter.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from toi_alert_formatter import format_toi_alert, format_toi_text


class TestFormatToiAlert:
    def test_valid_alert_ok(self) -> None:
        r = format_toi_alert("12345", 5.0, 2457000.0, 1000.0, 2.0, 0.1)
        assert r.flag == "OK"

    def test_invalid_period(self) -> None:
        r = format_toi_alert("12345", 0.0, 2457000.0, 1000.0, 2.0, 0.1)
        assert r.flag == "INVALID_PERIOD"

    def test_invalid_depth(self) -> None:
        r = format_toi_alert("12345", 5.0, 2457000.0, 0.0, 2.0, 0.1)
        assert r.flag == "INVALID_DEPTH"

    def test_invalid_fpp_high(self) -> None:
        r = format_toi_alert("12345", 5.0, 2457000.0, 1000.0, 2.0, 1.5)
        assert r.flag == "INVALID_FPP"

    def test_invalid_fpp_low(self) -> None:
        r = format_toi_alert("12345", 5.0, 2457000.0, 1000.0, 2.0, -0.1)
        assert r.flag == "INVALID_FPP"

    def test_tic_id_stored(self) -> None:
        r = format_toi_alert("99999", 5.0, 2457000.0, 1000.0, 2.0, 0.1)
        assert r.tic_id == "99999"

    def test_disposition_stored(self) -> None:
        r = format_toi_alert("12345", 5.0, 2457000.0, 1000.0, 2.0, 0.1, "FP")
        assert r.disposition == "FP"

    def test_period_stored(self) -> None:
        r = format_toi_alert("12345", 7.3, 2457000.0, 1000.0, 2.0, 0.1)
        assert r.period_days == 7.3

    def test_result_frozen(self) -> None:
        r = format_toi_alert("12345", 5.0, 2457000.0, 1000.0, 2.0, 0.1)
        try:
            r.flag = "X"  # type: ignore[misc]
            raise AssertionError("should be frozen")
        except Exception:
            pass

    def test_format_returns_string(self) -> None:
        r = format_toi_alert("12345", 5.0, 2457000.0, 1000.0, 2.0, 0.1)
        s = format_toi_text(r)
        assert isinstance(s, str)
        assert "TOI" in s

    def test_format_contains_tic_id(self) -> None:
        r = format_toi_alert("77777", 5.0, 2457000.0, 1000.0, 2.0, 0.1)
        s = format_toi_text(r)
        assert "77777" in s

    def test_default_disposition_pc(self) -> None:
        r = format_toi_alert("12345", 5.0, 2457000.0, 1000.0, 2.0, 0.1)
        assert r.disposition == "PC"
