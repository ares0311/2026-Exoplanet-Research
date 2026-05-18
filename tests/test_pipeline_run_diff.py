"""Tests for pipeline_run_diff.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from pipeline_run_diff import (
    diff_pipeline_runs,
    format_run_diff,
)


def _row(tic_id, fpp, rank):
    return {"tic_id": tic_id, "false_positive_probability": fpp, "rank_score": rank}


class TestDiffPipelineRuns:
    def test_empty_both(self):
        r = diff_pipeline_runs([], [])
        assert r.flag == "EMPTY"

    def test_added(self):
        after = [_row(1, 0.1, 0.9)]
        r = diff_pipeline_runs([], after)
        assert r.n_added == 1
        assert r.flag == "OK"

    def test_removed(self):
        before = [_row(1, 0.1, 0.9)]
        r = diff_pipeline_runs(before, [])
        assert r.n_removed == 1

    def test_stable(self):
        row = _row(1, 0.1, 0.9)
        r = diff_pipeline_runs([row], [row])
        changes = [c for c in r.changes if c.change_type == "STABLE"]
        assert len(changes) == 1
        assert r.flag == "NO_CHANGE"

    def test_improved_lower_fpp(self):
        before = [_row(1, 0.3, 0.7)]
        after = [_row(1, 0.1, 0.7)]  # FPP dropped by 0.2
        r = diff_pipeline_runs(before, after)
        assert r.changes[0].change_type == "IMPROVED"

    def test_degraded_higher_fpp(self):
        before = [_row(1, 0.1, 0.7)]
        after = [_row(1, 0.3, 0.7)]  # FPP rose by 0.2
        r = diff_pipeline_runs(before, after)
        assert r.changes[0].change_type == "DEGRADED"

    def test_delta_fpp_computed(self):
        before = [_row(1, 0.3, 0.7)]
        after = [_row(1, 0.1, 0.7)]
        r = diff_pipeline_runs(before, after)
        assert r.changes[0].delta_fpp is not None
        assert abs(r.changes[0].delta_fpp - (-0.2)) < 0.01

    def test_n_before_n_after(self):
        before = [_row(1, 0.1, 0.9), _row(2, 0.2, 0.8)]
        after = [_row(1, 0.1, 0.9)]
        r = diff_pipeline_runs(before, after)
        assert r.n_before == 2
        assert r.n_after == 1

    def test_custom_key(self):
        before = [{"name": "star_A", "fpp": 0.2, "rank_score": 0.7}]
        after = [{"name": "star_A", "fpp": 0.05, "rank_score": 0.9}]
        r = diff_pipeline_runs(before, after, key="name")
        assert len(r.changes) == 1

    def test_summary_markdown_nonempty(self):
        r = diff_pipeline_runs([_row(1, 0.1, 0.9)], [_row(2, 0.2, 0.8)])
        assert len(r.summary_markdown) > 0

    def test_result_frozen(self):
        r = diff_pipeline_runs([], [])
        try:
            r.flag = "x"  # type: ignore[misc]
            raise AssertionError()
        except (AttributeError, TypeError):
            pass


class TestFormatRunDiff:
    def test_returns_string(self):
        r = diff_pipeline_runs([_row(1, 0.1, 0.9)], [_row(1, 0.05, 0.95)])
        assert isinstance(format_run_diff(r), str)

    def test_contains_flag(self):
        r = diff_pipeline_runs([_row(1, 0.1, 0.9)], [_row(1, 0.05, 0.95)])
        s = format_run_diff(r)
        assert r.flag in s

    def test_contains_counts(self):
        r = diff_pipeline_runs([_row(1, 0.1, 0.9)], [_row(2, 0.2, 0.8)])
        s = format_run_diff(r)
        assert "Added" in s or "Removed" in s
