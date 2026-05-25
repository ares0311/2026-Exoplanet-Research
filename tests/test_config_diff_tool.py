"""Tests for Skills/config_diff_tool.py"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from config_diff_tool import diff_configs, format_config_diff, load_and_diff_configs

OLD = {"a": 1, "b": {"x": 10, "y": 20}, "c": "hello"}
NEW = {"a": 2, "b": {"x": 10, "y": 30}, "d": "world"}


def test_added_key():
    r = diff_configs(OLD, NEW)
    assert r.n_added == 1  # "d" is new


def test_removed_key():
    r = diff_configs(OLD, NEW)
    assert r.n_removed == 1  # "c" is removed


def test_changed_key():
    r = diff_configs(OLD, NEW)
    assert r.n_changed >= 2  # "a" and "b.y" changed


def test_flag_ok():
    r = diff_configs(OLD, NEW)
    assert r.flag == "OK"


def test_no_change():
    r = diff_configs(OLD, OLD)
    assert r.flag == "NO_CHANGE"
    assert r.n_added == 0
    assert r.n_removed == 0
    assert r.n_changed == 0


def test_nested_dot_keys():
    r = diff_configs(OLD, NEW)
    keys = [e.key for e in r.entries]
    assert any("b.y" in k for k in keys)


def test_include_unchanged():
    r = diff_configs(OLD, NEW, include_unchanged=True)
    assert r.n_unchanged > 0
    types = {e.change_type for e in r.entries}
    assert "unchanged" in types


def test_invalid_non_dict():
    r = diff_configs("not a dict", {})
    assert r.flag == "INVALID"


def test_entries_tuple():
    r = diff_configs(OLD, NEW)
    assert isinstance(r.entries, tuple)


def test_format_returns_string():
    r = diff_configs(OLD, NEW)
    assert isinstance(format_config_diff(r), str)


def test_format_contains_key_words():
    r = diff_configs(OLD, NEW)
    text = format_config_diff(r)
    assert "Config Diff" in text
    assert "Flag" in text


def test_load_and_diff(tmp_path):
    old_file = tmp_path / "old.json"
    new_file = tmp_path / "new.json"
    old_file.write_text(json.dumps(OLD))
    new_file.write_text(json.dumps(NEW))
    r = load_and_diff_configs(old_file, new_file)
    assert r.flag == "OK"


def test_load_invalid_path():
    r = load_and_diff_configs("/nonexistent/a.json", "/nonexistent/b.json")
    assert r.flag == "INVALID"
