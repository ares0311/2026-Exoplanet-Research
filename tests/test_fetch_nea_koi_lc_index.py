"""Tests for Skills/fetch_nea_koi_lc_index.py (13 tests)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Skills"))

from fetch_nea_koi_lc_index import KoiRecord, fetch_koi_lc_index, format_koi_lc_index

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_GOOD_CSV = """kepid,kepoi_name,koi_disposition,koi_period,koi_time0bk,koi_duration
10000001,K00001.01,CONFIRMED,2.204735,134.452987,2.12300
10000002,K00002.01,FALSE POSITIVE,5.000000,135.123456,1.50000
10000003,K00003.01,CONFIRMED,10.123456,136.789012,3.25000
"""

_EMPTY_CSV = ""

_COMMENT_CSV = "# comment only\n"

_UNKNOWN_DISP_CSV = """kepid,kepoi_name,koi_disposition,koi_period,koi_time0bk,koi_duration
10000001,K00001.01,CANDIDATE,2.204735,134.452987,2.123
"""


def _make_tap(csv_text: str):
    def _fn(url: str) -> str:
        return csv_text
    return _fn


def _raise_tap(url: str) -> str:
    raise ConnectionError("no network")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_ok_flag():
    result = fetch_koi_lc_index(tap_fn=_make_tap(_GOOD_CSV))
    assert result.flag == "OK"


def test_n_confirmed():
    result = fetch_koi_lc_index(tap_fn=_make_tap(_GOOD_CSV))
    assert result.n_confirmed == 2


def test_n_fp():
    result = fetch_koi_lc_index(tap_fn=_make_tap(_GOOD_CSV))
    assert result.n_fp == 1


def test_records_count():
    result = fetch_koi_lc_index(tap_fn=_make_tap(_GOOD_CSV))
    assert len(result.records) == 3


def test_record_type():
    result = fetch_koi_lc_index(tap_fn=_make_tap(_GOOD_CSV))
    rec = result.records[0]
    assert isinstance(rec, KoiRecord)
    assert rec.kepoi_name == "K00001.01"
    assert rec.period_days == pytest.approx(2.204735)


def test_fetch_error_flag():
    result = fetch_koi_lc_index(tap_fn=_raise_tap)
    assert result.flag == "FETCH_ERROR"
    assert len(result.records) == 0


def test_empty_csv_flag():
    result = fetch_koi_lc_index(tap_fn=_make_tap(_EMPTY_CSV))
    assert result.flag == "EMPTY"


def test_comment_only_flag():
    result = fetch_koi_lc_index(tap_fn=_make_tap(_COMMENT_CSV))
    assert result.flag == "EMPTY"


def test_unknown_disposition_skipped():
    result = fetch_koi_lc_index(tap_fn=_make_tap(_UNKNOWN_DISP_CSV))
    assert result.flag == "EMPTY"
    assert len(result.records) == 0


def test_max_rows_respected():
    result = fetch_koi_lc_index(tap_fn=_make_tap(_GOOD_CSV), max_rows=2)
    assert len(result.records) <= 2


def test_disposition_normalised_false_positive():
    result = fetch_koi_lc_index(tap_fn=_make_tap(_GOOD_CSV))
    fp_recs = [r for r in result.records if r.disposition == "FALSE_POSITIVE"]
    assert len(fp_recs) == 1


def test_formatter_contains_flag():
    result = fetch_koi_lc_index(tap_fn=_make_tap(_GOOD_CSV))
    text = format_koi_lc_index(result)
    assert "OK" in text
    assert len(text) > 0


def test_formatter_on_error():
    result = fetch_koi_lc_index(tap_fn=_raise_tap)
    text = format_koi_lc_index(result)
    assert "FETCH_ERROR" in text
