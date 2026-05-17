"""Tests for Skills.toi_watcher."""
from __future__ import annotations

from pathlib import Path

import pytest
from Skills.toi_watcher import (
    TOIChange,
    TOIWatchResult,
    format_watch_result,
    watch_toi_list,
)


def _toi_fn(*rows):
    def fn():
        return list(rows)
    return fn


def _row(toi="1.01", tic_id=1, disposition="PC"):
    return {"toi": toi, "tic_id": tic_id, "disposition": disposition}


class TestWatchTOIList:
    def test_returns_watch_result(self, tmp_path: Path) -> None:
        r = watch_toi_list(tmp_path / "snap.json", toi_table_fn=_toi_fn())
        assert isinstance(r, TOIWatchResult)

    def test_first_run_all_new(self, tmp_path: Path) -> None:
        r = watch_toi_list(
            tmp_path / "snap.json",
            toi_table_fn=_toi_fn(_row("1.01"), _row("2.01", tic_id=2)),
        )
        assert len(r.new_tois) == 2

    def test_second_run_no_changes(self, tmp_path: Path) -> None:
        snap = tmp_path / "snap.json"
        fn = _toi_fn(_row("1.01"))
        watch_toi_list(snap, toi_table_fn=fn)
        r2 = watch_toi_list(snap, toi_table_fn=fn)
        assert len(r2.new_tois) == 0
        assert len(r2.changed_tois) == 0

    def test_detects_new_toi(self, tmp_path: Path) -> None:
        snap = tmp_path / "snap.json"
        watch_toi_list(snap, toi_table_fn=_toi_fn(_row("1.01")))
        r2 = watch_toi_list(
            snap,
            toi_table_fn=_toi_fn(_row("1.01"), _row("3.01", tic_id=3)),
        )
        assert len(r2.new_tois) == 1
        assert r2.new_tois[0].toi == "3.01"

    def test_detects_removed_toi(self, tmp_path: Path) -> None:
        snap = tmp_path / "snap.json"
        watch_toi_list(snap, toi_table_fn=_toi_fn(_row("1.01"), _row("2.01", tic_id=2)))
        r2 = watch_toi_list(snap, toi_table_fn=_toi_fn(_row("1.01")))
        assert len(r2.removed_tois) == 1

    def test_detects_disposition_change(self, tmp_path: Path) -> None:
        snap = tmp_path / "snap.json"
        watch_toi_list(snap, toi_table_fn=_toi_fn(_row("1.01", disposition="PC")))
        r2 = watch_toi_list(
            snap,
            toi_table_fn=_toi_fn(_row("1.01", disposition="CP")),
        )
        assert any(c.field == "disposition" for c in r2.changed_tois)

    def test_snapshot_created(self, tmp_path: Path) -> None:
        snap = tmp_path / "snap.json"
        watch_toi_list(snap, toi_table_fn=_toi_fn())
        assert snap.exists()

    def test_snapshot_age_none_first_run(self, tmp_path: Path) -> None:
        r = watch_toi_list(tmp_path / "snap.json", toi_table_fn=_toi_fn())
        assert r.snapshot_age_hours is None

    def test_snapshot_age_set_second_run(self, tmp_path: Path) -> None:
        snap = tmp_path / "snap.json"
        watch_toi_list(snap, toi_table_fn=_toi_fn())
        r2 = watch_toi_list(snap, toi_table_fn=_toi_fn())
        assert r2.snapshot_age_hours is not None
        assert r2.snapshot_age_hours >= 0.0

    def test_generated_at_is_string(self, tmp_path: Path) -> None:
        r = watch_toi_list(tmp_path / "snap.json", toi_table_fn=_toi_fn())
        assert isinstance(r.generated_at, str)

    def test_empty_toi_fn(self, tmp_path: Path) -> None:
        r = watch_toi_list(tmp_path / "snap.json", toi_table_fn=_toi_fn())
        assert len(r.new_tois) == 0


class TestFormatWatchResult:
    def test_returns_string(self, tmp_path: Path) -> None:
        r = watch_toi_list(tmp_path / "snap.json", toi_table_fn=_toi_fn())
        assert isinstance(format_watch_result(r), str)

    def test_contains_header(self, tmp_path: Path) -> None:
        r = watch_toi_list(tmp_path / "snap.json", toi_table_fn=_toi_fn())
        assert "TOI Watch" in format_watch_result(r)
