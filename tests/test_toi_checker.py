"""Tests for Skills/toi_checker.py."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Skills.toi_checker import check_toi, format_toi_result  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_csv(*rows: dict[str, str]) -> str:
    """Build a minimal ExoFOP-style CSV string."""
    headers = ["TIC ID", "TOI", "TFOPWG Disposition", "Period (days)",
               "Epoch (BJD)", "Depth (ppm)", "Duration (hours)"]
    lines = [",".join(headers)]
    for row in rows:
        lines.append(",".join(row.get(h, "") for h in headers))
    return "\n".join(lines)


def _fn(*rows: dict[str, str]):
    csv = _make_csv(*rows)
    return lambda: csv


_MATCH_ROW = {
    "TIC ID": "150428135",
    "TOI": "700.01",
    "TFOPWG Disposition": "CP",
    "Period (days)": "37.4237",
    "Epoch (BJD)": "2458325.1",
    "Depth (ppm)": "1730",
    "Duration (hours)": "2.8",
}

_OTHER_ROW = {
    "TIC ID": "99999",
    "TOI": "1.01",
    "TFOPWG Disposition": "FP",
    "Period (days)": "1.234",
    "Epoch (BJD)": "2458000.0",
    "Depth (ppm)": "500",
    "Duration (hours)": "1.0",
}


# ---------------------------------------------------------------------------
# check_toi
# ---------------------------------------------------------------------------


class TestCheckToi:
    def test_returns_dict_when_found(self) -> None:
        result = check_toi(150428135, toi_table_fn=_fn(_MATCH_ROW))
        assert isinstance(result, dict)

    def test_returns_none_when_not_found(self) -> None:
        result = check_toi(999, toi_table_fn=_fn(_MATCH_ROW))
        assert result is None

    def test_toi_number_parsed(self) -> None:
        result = check_toi(150428135, toi_table_fn=_fn(_MATCH_ROW))
        assert result is not None and result["toi"] == "700.01"

    def test_disposition_parsed(self) -> None:
        result = check_toi(150428135, toi_table_fn=_fn(_MATCH_ROW))
        assert result is not None and result["disposition"] == "CP"

    def test_period_parsed_as_float(self) -> None:
        result = check_toi(150428135, toi_table_fn=_fn(_MATCH_ROW))
        assert result is not None
        assert isinstance(result["period_days"], float)
        assert result["period_days"] == pytest.approx(37.4237)

    def test_epoch_parsed_as_float(self) -> None:
        result = check_toi(150428135, toi_table_fn=_fn(_MATCH_ROW))
        assert result is not None and isinstance(result["epoch_bjd"], float)

    def test_correct_tic_returned_when_multiple_rows(self) -> None:
        result = check_toi(150428135, toi_table_fn=_fn(_OTHER_ROW, _MATCH_ROW))
        assert result is not None and result["tic_id"] == 150428135

    def test_empty_table_returns_none(self) -> None:
        result = check_toi(150428135, toi_table_fn=lambda: "TIC ID,TOI\n")
        assert result is None

    def test_depth_parsed(self) -> None:
        result = check_toi(150428135, toi_table_fn=_fn(_MATCH_ROW))
        assert result is not None and result["depth_ppm"] == pytest.approx(1730.0)


# ---------------------------------------------------------------------------
# format_toi_result
# ---------------------------------------------------------------------------


class TestFormatToiResult:
    def test_none_result_says_not_found(self) -> None:
        s = format_toi_result(None, tic_id=12345)
        assert "Not found" in s

    def test_found_result_contains_toi_number(self) -> None:
        result = check_toi(150428135, toi_table_fn=_fn(_MATCH_ROW))
        s = format_toi_result(result)
        assert "700.01" in s

    def test_found_result_contains_disposition(self) -> None:
        result = check_toi(150428135, toi_table_fn=_fn(_MATCH_ROW))
        s = format_toi_result(result)
        assert "CP" in s
