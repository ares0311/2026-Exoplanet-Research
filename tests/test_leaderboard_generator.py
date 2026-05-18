"""Tests for leaderboard_generator.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from leaderboard_generator import (
    format_leaderboard,
    generate_leaderboard,
)


def _make_records():
    return [
        {"tic_id": 1, "status": "candidate_found", "best_fpp": 0.05, "author": "alice"},
        {"tic_id": 2, "status": "scanned_clear", "best_fpp": None, "author": "bob"},
        {"tic_id": 3, "status": "candidate_found", "best_fpp": 0.10, "author": "alice"},
        {"tic_id": 4, "status": "error", "best_fpp": None, "author": "bob"},
        {"tic_id": 5, "status": "scanned_clear", "best_fpp": None, "author": "carol"},
    ]


class TestGenerateLeaderboard:
    def test_empty_records(self):
        r = generate_leaderboard([])
        assert r.flag == "EMPTY"
        assert r.n_entries == 0

    def test_target_mode(self):
        records = _make_records()
        r = generate_leaderboard(records, mode="target")
        assert r.mode == "target"
        assert r.n_entries == 5  # 5 unique TIC IDs

    def test_contributor_mode(self):
        records = _make_records()
        r = generate_leaderboard(records, mode="contributor")
        assert r.mode == "contributor"
        assert r.n_entries == 3  # alice, bob, carol

    def test_alice_ranked_first_contributor(self):
        records = _make_records()
        r = generate_leaderboard(records, mode="contributor", metric="candidates")
        assert r.entries[0].name == "alice"
        assert r.entries[0].n_candidates == 2

    def test_top_n_limits_entries(self):
        records = _make_records()
        r = generate_leaderboard(records, mode="target", top_n=3)
        assert r.n_entries <= 3

    def test_rank_increments(self):
        records = _make_records()
        r = generate_leaderboard(records, mode="contributor")
        for i, e in enumerate(r.entries):
            assert e.rank == i + 1

    def test_scores_positive(self):
        records = _make_records()
        r = generate_leaderboard(records, mode="contributor")
        for e in r.entries:
            assert e.score >= 0

    def test_metric_scanned(self):
        records = _make_records()
        r = generate_leaderboard(records, mode="contributor", metric="scanned")
        assert r.metric == "scanned"

    def test_flag_ok(self):
        r = generate_leaderboard(_make_records())
        assert r.flag == "OK"

    def test_generated_at_nonempty(self):
        r = generate_leaderboard(_make_records())
        assert len(r.generated_at) > 0

    def test_result_frozen(self):
        r = generate_leaderboard(_make_records())
        try:
            r.flag = "x"  # type: ignore[misc]
            raise AssertionError()
        except (AttributeError, TypeError):
            pass


class TestFormatLeaderboard:
    def test_returns_string(self):
        r = generate_leaderboard(_make_records())
        assert isinstance(format_leaderboard(r), str)

    def test_contains_flag(self):
        r = generate_leaderboard(_make_records())
        s = format_leaderboard(r)
        assert r.flag in s

    def test_markdown_table_header(self):
        r = generate_leaderboard(_make_records())
        s = format_leaderboard(r)
        assert "Rank" in s and "Name" in s
