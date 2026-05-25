"""Tests for Skills/toi_disposition_tracker.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from toi_disposition_tracker import DispositionChange, diff_toi_snapshots, format_toi_diff

OLD_CSV = """TOI,TIC ID,TFOPWG Disposition
101.01,12345,PC
102.01,23456,PC
103.01,34567,FP
"""

NEW_CSV = """TOI,TIC ID,TFOPWG Disposition
101.01,12345,CP
102.01,23456,FP
104.01,45678,PC
"""


def test_confirmed_detection():
    r = diff_toi_snapshots(OLD_CSV, NEW_CSV)
    assert r.n_confirmed == 1


def test_new_fp_detection():
    r = diff_toi_snapshots(OLD_CSV, NEW_CSV)
    assert r.n_new_fp == 1


def test_added_detection():
    r = diff_toi_snapshots(OLD_CSV, NEW_CSV)
    assert r.n_added == 1


def test_removed_detection():
    r = diff_toi_snapshots(OLD_CSV, NEW_CSV)
    assert r.n_removed == 1


def test_flag_ok():
    r = diff_toi_snapshots(OLD_CSV, NEW_CSV)
    assert r.flag == "OK"


def test_no_changes():
    r = diff_toi_snapshots(OLD_CSV, OLD_CSV)
    assert r.n_confirmed == 0
    assert r.n_new_fp == 0
    assert r.n_added == 0
    assert r.n_removed == 0
    assert len(r.changes) == 0


def test_empty_inputs():
    r = diff_toi_snapshots("", "")
    assert r.flag in ("EMPTY", "INVALID")


def test_change_types_in_changes():
    r = diff_toi_snapshots(OLD_CSV, NEW_CSV)
    types = {c.change_type for c in r.changes}
    assert "confirmed" in types
    assert "fp" in types
    assert "added" in types
    assert "removed" in types


def test_custom_columns():
    old = "ID,TIC,Disp\n1.01,111,PC\n"
    new = "ID,TIC,Disp\n1.01,111,CP\n"
    r = diff_toi_snapshots(old, new, toi_col="ID", tic_col="TIC", disp_col="Disp")
    assert r.n_confirmed == 1


def test_format_contains_key_words():
    r = diff_toi_snapshots(OLD_CSV, NEW_CSV)
    text = format_toi_diff(r)
    assert "TOI Disposition" in text
    assert "confirmed" in text.lower() or "Confirmed" in text


def test_format_returns_string():
    r = diff_toi_snapshots(OLD_CSV, NEW_CSV)
    assert isinstance(format_toi_diff(r), str)


def test_changes_are_tuple():
    r = diff_toi_snapshots(OLD_CSV, NEW_CSV)
    assert isinstance(r.changes, tuple)


def test_disposition_change_fields():
    r = diff_toi_snapshots(OLD_CSV, NEW_CSV)
    for c in r.changes:
        assert isinstance(c, DispositionChange)
        assert c.toi
        assert c.change_type in ("added", "removed", "confirmed", "fp", "changed")
