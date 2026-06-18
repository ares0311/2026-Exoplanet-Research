"""Tests for Skills/tess_tce_fetcher.py"""
import sys
from pathlib import Path
from urllib.error import HTTPError

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from tess_tce_fetcher import (
    fetch_tce_table,
    format_tce_summary,
    tce_to_label_rows,
)

_ROWS = [
    {"ticid": 100, "tce_num": 1, "tce_period": 5.0, "tce_time0bt": 1325.0,
     "tce_duration": 2.5, "tce_depth": 10000, "tce_snr": 15.0,
     "tce_disposition": "PC", "sectors": "1,2"},
    {"ticid": 200, "tce_num": 1, "tce_period": 10.0, "tce_time0bt": 1330.0,
     "tce_duration": 3.0, "tce_depth": 5000, "tce_snr": 8.0,
     "tce_disposition": "FP", "sectors": "3"},
    {"ticid": 300, "tce_num": 1, "tce_period": 7.0, "tce_time0bt": 1340.0,
     "tce_duration": 1.5, "tce_depth": 2000, "tce_snr": 6.0,
     "tce_disposition": "ND", "sectors": "1"},
]


def _mock_fetch(url: str) -> list[dict]:
    return _ROWS


def test_fetch_returns_ok():
    r = fetch_tce_table(fetch_fn=_mock_fetch)
    assert r.flag == "OK"


def test_fetch_counts_pc():
    r = fetch_tce_table(fetch_fn=_mock_fetch)
    assert r.n_planet_candidate == 1


def test_fetch_counts_fp():
    r = fetch_tce_table(fetch_fn=_mock_fetch)
    assert r.n_false_positive == 1


def test_fetch_counts_nd():
    r = fetch_tce_table(fetch_fn=_mock_fetch)
    assert r.n_not_dispositioned == 1


def test_total_matches_sum():
    r = fetch_tce_table(fetch_fn=_mock_fetch)
    assert r.n_total == r.n_planet_candidate + r.n_false_positive + r.n_not_dispositioned


def test_disposition_filter():
    r = fetch_tce_table(fetch_fn=_mock_fetch, disposition_filter=["PC"])
    assert r.n_total == 1
    assert r.n_planet_candidate == 1
    assert r.n_false_positive == 0


def test_empty_response():
    r = fetch_tce_table(fetch_fn=lambda url: [])
    assert r.flag == "EMPTY"


def test_invalid_response():
    r = fetch_tce_table(fetch_fn=lambda url: "not-a-list")
    assert r.flag == "INVALID"


def test_network_error():
    def bad_fetch(url: str) -> list[dict]:
        raise ConnectionError("no network")
    r = fetch_tce_table(fetch_fn=bad_fetch)
    assert r.flag == "INVALID"
    assert r.error_message is not None


def test_stale_endpoint_404_reports_unavailable():
    def bad_fetch(url: str) -> list[dict]:
        raise HTTPError(url, 404, "Not Found", hdrs=None, fp=None)

    r = fetch_tce_table(fetch_fn=bad_fetch)

    assert r.flag == "UNAVAILABLE"
    assert r.error_message is not None
    assert "404" in r.error_message


def test_tce_to_label_rows_pc():
    r = fetch_tce_table(fetch_fn=_mock_fetch)
    rows = tce_to_label_rows(r)
    pc_rows = [row for row in rows if row["label"] == "planet_candidate"]
    assert len(pc_rows) == 1


def test_tce_to_label_rows_fp():
    r = fetch_tce_table(fetch_fn=_mock_fetch)
    rows = tce_to_label_rows(r)
    fp_rows = [row for row in rows if row["label"] == "false_positive"]
    assert len(fp_rows) == 1


def test_tce_to_label_rows_nd_excluded():
    r = fetch_tce_table(fetch_fn=_mock_fetch)
    rows = tce_to_label_rows(r)
    # ND disposition should not produce label rows
    assert len(rows) == 2


def test_format_returns_string():
    r = fetch_tce_table(fetch_fn=_mock_fetch)
    s = format_tce_summary(r)
    assert isinstance(s, str)
    assert "TCE" in s


def test_format_gate_fail():
    r = fetch_tce_table(fetch_fn=_mock_fetch)
    s = format_tce_summary(r)
    assert "FAIL" in s  # only 2 usable, < 5000
