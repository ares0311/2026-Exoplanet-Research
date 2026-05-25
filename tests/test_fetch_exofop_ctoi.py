"""Tests for Skills/fetch_exofop_ctoi.py (13 tests)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Skills"))

from fetch_exofop_ctoi import fetch_ctoi_table, format_ctoi_result

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_GOOD_CSV = """CTOI,TIC,User Disposition,Period (days),Duration (hours),Epoch (BJD),Num Reports
1234.01,111222333,CP,3.5,2.1,2458000.5,5
5678.01,444555666,FP,7.0,1.8,2458001.0,3
9999.01,777888999,EB,1.2,0.9,2458002.0,2
8888.01,123456789,PC,15.0,3.0,2458003.0,1
"""

_EMPTY_CSV = ""

_COMMENT_CSV = """# This is a comment
# Another comment
"""


def _make_fetch(csv_text: str):
    def _fn(url: str) -> str:
        return csv_text
    return _fn


def _raise_fetch(url: str) -> str:
    raise OSError("network error")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_ok_basic():
    result = fetch_ctoi_table(fetch_fn=_make_fetch(_GOOD_CSV), min_ratings=1)
    assert result.flag == "OK"
    assert len(result.rows) == 4


def test_n_cp_count():
    result = fetch_ctoi_table(fetch_fn=_make_fetch(_GOOD_CSV), min_ratings=1)
    assert result.n_cp == 1


def test_n_fp_count():
    result = fetch_ctoi_table(fetch_fn=_make_fetch(_GOOD_CSV), min_ratings=1)
    # FP + EB both map to fp
    assert result.n_fp == 2


def test_n_pc_count():
    result = fetch_ctoi_table(fetch_fn=_make_fetch(_GOOD_CSV), min_ratings=1)
    assert result.n_pc == 1


def test_fetch_error_flag():
    result = fetch_ctoi_table(fetch_fn=_raise_fetch)
    assert result.flag == "FETCH_ERROR"
    assert len(result.rows) == 0


def test_empty_csv_flag():
    result = fetch_ctoi_table(fetch_fn=_make_fetch(_EMPTY_CSV))
    assert result.flag == "EMPTY"


def test_comment_only_csv_flag():
    result = fetch_ctoi_table(fetch_fn=_make_fetch(_COMMENT_CSV))
    assert result.flag == "EMPTY"


def test_min_ratings_filter():
    result = fetch_ctoi_table(fetch_fn=_make_fetch(_GOOD_CSV), min_ratings=4)
    # Only CTOI 1234.01 has 5 ratings
    assert result.n_cp == 1
    assert result.n_fp == 0


def test_row_structure():
    result = fetch_ctoi_table(fetch_fn=_make_fetch(_GOOD_CSV), min_ratings=1)
    row = result.rows[0]
    assert "tic_id" in row
    assert "toi" in row
    assert "disposition" in row
    assert "period_days" in row
    assert "epoch_bjd" in row
    assert "duration_hours" in row
    assert "n_ratings" in row


def test_period_parsed():
    result = fetch_ctoi_table(fetch_fn=_make_fetch(_GOOD_CSV), min_ratings=1)
    assert result.rows[0]["period_days"] == pytest.approx(3.5)


def test_disposition_normalisation_eb():
    result = fetch_ctoi_table(fetch_fn=_make_fetch(_GOOD_CSV), min_ratings=1)
    disps = {r["toi"]: r["disposition"] for r in result.rows}
    # 9999.01 is EB → fp
    assert disps["9999.01"] == "fp"


def test_formatter_contains_flag():
    result = fetch_ctoi_table(fetch_fn=_make_fetch(_GOOD_CSV), min_ratings=1)
    text = format_ctoi_result(result)
    assert "OK" in text
    assert len(text) > 0


def test_formatter_on_error():
    result = fetch_ctoi_table(fetch_fn=_raise_fetch)
    text = format_ctoi_result(result)
    assert "FETCH_ERROR" in text
