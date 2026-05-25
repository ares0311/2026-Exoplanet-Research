"""Tests for Skills/multi_run_diff_reporter.py"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from multi_run_diff_reporter import diff_pipeline_runs, format_run_diff, load_and_diff

OLD_ROWS = [
    {
        "candidate_id": "sig1", "fpp": 0.30, "planet_posterior": 0.50,
        "pathway": "planet_hunters_discussion", "period_days": 5.0,
    },
    {
        "candidate_id": "sig2", "fpp": 0.80, "planet_posterior": 0.10,
        "pathway": "github_only_reproducibility",
    },
]
NEW_ROWS = [
    {
        "candidate_id": "sig1", "fpp": 0.10, "planet_posterior": 0.70,
        "pathway": "tfop_ready", "period_days": 5.0,
    },
    {
        "candidate_id": "sig2", "fpp": 0.80, "planet_posterior": 0.10,
        "pathway": "github_only_reproducibility",
    },
]


def test_improved_signal():
    r = diff_pipeline_runs(OLD_ROWS, NEW_ROWS)
    assert r.n_improved >= 1


def test_pathway_change():
    r = diff_pipeline_runs(OLD_ROWS, NEW_ROWS)
    assert r.n_pathway_changes >= 1


def test_flag_ok():
    r = diff_pipeline_runs(OLD_ROWS, NEW_ROWS)
    assert r.flag == "OK"


def test_no_change():
    r = diff_pipeline_runs(OLD_ROWS, OLD_ROWS)
    assert r.flag == "NO_CHANGE"


def test_empty_both():
    r = diff_pipeline_runs([], [])
    assert r.flag == "EMPTY"


def test_signal_counts():
    r = diff_pipeline_runs(OLD_ROWS, NEW_ROWS)
    assert r.n_signals_old == 2
    assert r.n_signals_new == 2


def test_added_signal():
    new = NEW_ROWS + [{"candidate_id": "sig3", "fpp": 0.05}]
    r = diff_pipeline_runs(OLD_ROWS, new)
    types = {d.change_type for d in r.diffs}
    assert "added" in types


def test_removed_signal():
    old = OLD_ROWS + [{"candidate_id": "sig3", "fpp": 0.05}]
    r = diff_pipeline_runs(old, NEW_ROWS)
    types = {d.change_type for d in r.diffs}
    assert "removed" in types


def test_regressed_signal():
    new = [{"candidate_id": "sig1", "fpp": 0.70, "period_days": 5.0}]
    old = [{"candidate_id": "sig1", "fpp": 0.20, "period_days": 5.0}]
    r = diff_pipeline_runs(old, new)
    assert r.n_regressed >= 1


def test_format_returns_string():
    r = diff_pipeline_runs(OLD_ROWS, NEW_ROWS)
    assert isinstance(format_run_diff(r), str)


def test_format_contains_key_words():
    r = diff_pipeline_runs(OLD_ROWS, NEW_ROWS)
    text = format_run_diff(r)
    assert "Pipeline Run Diff" in text
    assert "Flag" in text


def test_load_and_diff(tmp_path):
    old_file = tmp_path / "old.json"
    new_file = tmp_path / "new.json"
    old_file.write_text(json.dumps(OLD_ROWS))
    new_file.write_text(json.dumps(NEW_ROWS))
    r = load_and_diff(old_file, new_file)
    assert r.flag == "OK"


def test_invalid_input():
    r = diff_pipeline_runs("not a list", [])
    assert r.flag == "INVALID"
