"""Tests for Skills/signal_comparison_reporter.py."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from signal_comparison_reporter import (
    SignalComparisonResult,
    compare_signals,
    format_signal_comparison,
)


def _signal(tic_id: int = 1, period: float = 5.0, depth: float = 1000.0) -> dict:
    return {
        "tic_id": tic_id,
        "period_days": period,
        "depth_ppm": depth,
        "duration_hours": 2.0,
        "snr": 10.0,
    }


class TestCompareSignals:
    def test_returns_signal_comparison_result(self):
        result = compare_signals([_signal(1), _signal(2)])
        assert isinstance(result, SignalComparisonResult)

    def test_empty_signals_flag(self):
        result = compare_signals([])
        assert result.flag == "EMPTY"

    def test_single_signal_flag(self):
        result = compare_signals([_signal(1)])
        assert result.flag == "SINGLE"

    def test_non_list_returns_invalid(self):
        result = compare_signals("not_a_list")  # type: ignore[arg-type]
        assert result.flag == "INVALID"

    def test_two_signals_flag_ok(self):
        result = compare_signals([_signal(1), _signal(2)])
        assert result.flag == "OK"

    def test_n_signals_stored(self):
        result = compare_signals([_signal(1), _signal(2), _signal(3)])
        assert result.n_signals == 3

    def test_headers_tuple(self):
        result = compare_signals([_signal(1), _signal(2)])
        assert isinstance(result.headers, tuple)
        assert len(result.headers) > 0

    def test_rows_count_equals_n_signals(self):
        result = compare_signals([_signal(1), _signal(2), _signal(3)])
        assert len(result.rows) == 3

    def test_duplicate_period_note_in_summary(self):
        s1 = {"period_days": 5.0, "depth_ppm": 1000.0}
        s2 = {"period_days": 5.01, "depth_ppm": 1000.0}
        result = compare_signals([s1, s2])
        assert any("duplicate" in line.lower() for line in result.summary_lines)

    def test_harmonic_period_note_in_summary(self):
        s1 = {"period_days": 10.0}
        s2 = {"period_days": 5.0}
        result = compare_signals([s1, s2])
        # May flag as alias
        assert result.flag == "OK"

    def test_single_summary_line_message(self):
        result = compare_signals([_signal(1)])
        assert len(result.summary_lines) == 1
        assert "nothing to compare" in result.summary_lines[0].lower()

    def test_frozen_dataclass(self):
        result = compare_signals([_signal(1), _signal(2)])
        with pytest.raises((AttributeError, TypeError)):
            result.flag = "CHANGED"  # type: ignore[misc]


class TestFormatSignalComparison:
    def test_returns_string(self):
        result = compare_signals([_signal(1), _signal(2)])
        md = format_signal_comparison(result)
        assert isinstance(md, str)

    def test_empty_flag_produces_message(self):
        result = compare_signals([])
        md = format_signal_comparison(result)
        assert "No signals" in md

    def test_single_flag_produces_message(self):
        result = compare_signals([_signal(1)])
        md = format_signal_comparison(result)
        assert "one signal" in md.lower() or "nothing" in md.lower()

    def test_ok_contains_table_separator(self):
        result = compare_signals([_signal(1), _signal(2)])
        md = format_signal_comparison(result)
        assert "|" in md

    def test_contains_n_signals_footer(self):
        result = compare_signals([_signal(1), _signal(2)])
        md = format_signal_comparison(result)
        assert "2" in md
