"""Tests for Skills/candidate_timeline.py."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Skills.candidate_timeline import CandidateTimeline  # noqa: E402


def _row(**kwargs) -> dict:
    base = {
        "candidate_id": "TIC1_01",
        "period_days": 5.0,
        "pathway": "tfop_ready",
        "scores": {"false_positive_probability": 0.10},
        "posterior": {"planet_candidate": 0.70},
    }
    base.update(kwargs)
    return base


class TestCandidateTimeline:
    def test_new_file_created_on_first_record(self, tmp_path: Path) -> None:
        tl = CandidateTimeline(tmp_path)
        tl.record(_row())
        # record() also creates a per-candidate ".json.lock" file alongside
        # the data file (see TestCandidateTimelineConcurrency).
        json_files = list(tmp_path.glob("*.json"))
        assert len(json_files) == 1

    def test_entries_empty_on_new_timeline(self, tmp_path: Path) -> None:
        tl = CandidateTimeline(tmp_path)
        assert tl.entries("TIC1_01") == []

    def test_latest_none_when_empty(self, tmp_path: Path) -> None:
        tl = CandidateTimeline(tmp_path)
        assert tl.latest("TIC1_01") is None

    def test_single_record_stored_correctly(self, tmp_path: Path) -> None:
        tl = CandidateTimeline(tmp_path)
        tl.record(_row(period_days=7.5))
        entries = tl.entries("TIC1_01")
        assert len(entries) == 1
        assert entries[0].period_days == pytest.approx(7.5)

    def test_latest_returns_most_recent(self, tmp_path: Path) -> None:
        tl = CandidateTimeline(tmp_path)
        tl.record(_row(scores={"false_positive_probability": 0.30}))
        tl.record(_row(scores={"false_positive_probability": 0.10}))
        latest = tl.latest("TIC1_01")
        assert latest is not None
        assert latest.fpp == pytest.approx(0.10)

    def test_genuine_zero_fpp_is_not_replaced_by_stale_best_fpp(
        self, tmp_path: Path
    ) -> None:
        # Regression: `or`-chaining treated a real 0.0 (perfect FPP) as
        # falsy and silently fell through to a different, possibly stale
        # field. A genuine 0.0 in scores.false_positive_probability must
        # win outright, never best_fpp.
        tl = CandidateTimeline(tmp_path)
        tl.record(
            _row(
                scores={"false_positive_probability": 0.0},
                best_fpp=0.85,
            )
        )
        latest = tl.latest("TIC1_01")
        assert latest is not None
        assert latest.fpp == pytest.approx(0.0)

    def test_genuine_zero_planet_posterior_is_preserved(self, tmp_path: Path) -> None:
        tl = CandidateTimeline(tmp_path)
        tl.record(_row(posterior={"planet_candidate": 0.0}))
        latest = tl.latest("TIC1_01")
        assert latest is not None
        assert latest.planet_posterior == pytest.approx(0.0)

    def test_summary_n_runs_correct(self, tmp_path: Path) -> None:
        tl = CandidateTimeline(tmp_path)
        tl.record(_row())
        tl.record(_row())
        tl.record(_row())
        s = tl.summary("TIC1_01")
        assert s["n_runs"] == 3

    def test_summary_trend_fpp_positive_when_worsening(self, tmp_path: Path) -> None:
        tl = CandidateTimeline(tmp_path)
        tl.record(_row(scores={"false_positive_probability": 0.10}))
        tl.record(_row(scores={"false_positive_probability": 0.50}))
        s = tl.summary("TIC1_01")
        assert s["trend_fpp"] > 0

    def test_summary_trend_fpp_negative_when_improving(self, tmp_path: Path) -> None:
        tl = CandidateTimeline(tmp_path)
        tl.record(_row(scores={"false_positive_probability": 0.50}))
        tl.record(_row(scores={"false_positive_probability": 0.10}))
        s = tl.summary("TIC1_01")
        assert s["trend_fpp"] < 0

    def test_to_markdown_non_empty(self, tmp_path: Path) -> None:
        tl = CandidateTimeline(tmp_path)
        tl.record(_row())
        md = tl.to_markdown("TIC1_01")
        assert isinstance(md, str)
        assert len(md) > 0

    def test_to_markdown_contains_pathway(self, tmp_path: Path) -> None:
        tl = CandidateTimeline(tmp_path)
        tl.record(_row(pathway="planet_hunters_discussion"))
        md = tl.to_markdown("TIC1_01")
        assert "planet_hunters_discussion" in md

    def test_atomic_write_produces_valid_json(self, tmp_path: Path) -> None:
        tl = CandidateTimeline(tmp_path)
        tl.record(_row())
        file = tmp_path / "TIC1_01.json"
        data = json.loads(file.read_text())
        assert isinstance(data, list)

    def test_parent_dirs_created(self, tmp_path: Path) -> None:
        nested = tmp_path / "deep" / "nested"
        tl = CandidateTimeline(nested)
        tl.record(_row())
        assert nested.exists()


class TestCandidateTimelineConcurrency:
    def test_concurrent_record_calls_do_not_lose_either_entry(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Regression: record() used to load, append, and save with no lock
        # around the read-modify-write, so two concurrent record() calls
        # for the same candidate could each load the same on-disk state and
        # each save their own single entry, silently losing one run's
        # entry to the other's. A short sleep is injected inside the now-
        # locked region (between load and save) to reliably widen the race
        # window: the second thread must block on fcntl.flock() until the
        # first releases, so by the time it loads, it sees the first
        # thread's already-appended entry.
        import threading
        import time

        tl = CandidateTimeline(tmp_path)
        original_load = tl._load

        def slow_load(candidate_id: str):
            data = original_load(candidate_id)
            time.sleep(0.05)
            return data

        monkeypatch.setattr(tl, "_load", slow_load)

        def worker(fpp: float) -> None:
            tl.record(_row(scores={"false_positive_probability": fpp}))

        threads = [
            threading.Thread(target=worker, args=(0.1,)),
            threading.Thread(target=worker, args=(0.2,)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        entries = tl.entries("TIC1_01")
        assert len(entries) == 2
        assert {e.fpp for e in entries} == {0.1, 0.2}
