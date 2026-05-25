"""Tests for Skills/label_coverage_reporter.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from label_coverage_reporter import format_label_coverage, report_label_coverage


def test_empty_records_flag():
    r = report_label_coverage([])
    assert r.flag == "EMPTY"


def test_empty_records_counts():
    r = report_label_coverage([])
    assert r.n_total == 0
    assert r.n_positive == 0
    assert r.n_negative == 0


def test_below_gate_flag():
    records = [{"label": 1}, {"label": 0}]
    r = report_label_coverage(records, gate_threshold=5000)
    assert r.flag == "BELOW_GATE"
    assert not r.gate_open


def test_at_gate_flag():
    records = [{"label": 1}] * 2500 + [{"label": 0}] * 2500
    r = report_label_coverage(records, gate_threshold=5000)
    assert r.flag == "OK"
    assert r.gate_open


def test_by_source_counts():
    records = [
        {"label": 1, "source": "tess"},
        {"label": 0, "source": "tess"},
        {"label": 1, "source": "kepler"},
    ]
    r = report_label_coverage(records)
    assert r.by_source["tess"]["positive"] == 1
    assert r.by_source["tess"]["negative"] == 1
    assert r.by_source["kepler"]["positive"] == 1
    assert r.by_source["kepler"]["negative"] == 0


def test_period_bin_less_than_1d():
    records = [{"label": 1, "period_days": 0.5}]
    r = report_label_coverage(records)
    assert r.by_period_bin["<1d"] == 1


def test_period_bin_1_to_10d():
    records = [{"label": 0, "period_days": 5.0}]
    r = report_label_coverage(records)
    assert r.by_period_bin["1-10d"] == 1


def test_missing_period_goes_to_unknown():
    records = [{"label": 1}]
    r = report_label_coverage(records)
    assert r.by_period_bin["unknown"] == 1


def test_depth_binning():
    records = [
        {"label": 1, "depth_ppm": 300.0},
        {"label": 0, "depth_ppm": 1000.0},
        {"label": 1, "depth_ppm": 5000.0},
        {"label": 0, "depth_ppm": 15000.0},
    ]
    r = report_label_coverage(records)
    assert r.by_depth_bin["<500ppm"] == 1
    assert r.by_depth_bin["500-2000ppm"] == 1
    assert r.by_depth_bin["2000-10000ppm"] == 1
    assert r.by_depth_bin[">10000ppm"] == 1


def test_n_pos_plus_n_neg_le_n_total():
    records = [{"label": 1}, {"label": 0}, {"label": 99}]
    r = report_label_coverage(records)
    assert r.n_positive + r.n_negative <= r.n_total


def test_format_returns_str():
    r = report_label_coverage([{"label": 1}])
    assert isinstance(format_label_coverage(r), str)


def test_format_contains_gate():
    r = report_label_coverage([{"label": 1}])
    text = format_label_coverage(r)
    assert "gate" in text.lower()


def test_custom_gate_threshold():
    records = [{"label": 1}, {"label": 0}]
    r = report_label_coverage(records, gate_threshold=2)
    assert r.gate_open
    assert r.flag == "OK"


def test_label_not_0_or_1_counted_in_total_not_pos_neg():
    records = [{"label": 99}, {"label": 1}, {"label": 0}]
    r = report_label_coverage(records)
    assert r.n_total == 3
    assert r.n_positive == 1
    assert r.n_negative == 1
