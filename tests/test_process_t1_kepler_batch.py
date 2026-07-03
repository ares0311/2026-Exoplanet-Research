"""Tests for Skills/process_t1_kepler_batch.py."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Skills"))

from process_t1_kepler_batch import (  # noqa: E402
    BatchSummary,
    T1KeplerProcessingStore,
    TargetResult,
    _clear_directory,
    _directory_size_bytes,
    _format_eta,
    format_batch_summary,
    group_rows_by_target,
    load_manifest_rows,
    make_default_lc_fetcher,
    process_target,
    run_batch,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _manifest_row(
    *,
    target_id: int = 10797460,
    source_row_id: str = "K00752.01",
    period_days: float = 9.48803557,
    epoch_bkjd: float = 170.53875,
    label: int = 1,
    split: str = "train",
) -> dict:
    return {
        "duration_hours": 2.9575,
        "epoch_bkjd": epoch_bkjd,
        "group_key": f"kepler:kic:{target_id}",
        "label": label,
        "label_name": "CONFIRMED" if label == 1 else "FALSE POSITIVE",
        "lightcurve_search": {
            "author": "Kepler", "exptime": 1800, "mission": "Kepler",
            "target": f"KIC {target_id}",
        },
        "manifest_version": "t1-1-kepler-manifest-v1",
        "mission": "Kepler",
        "period_days": period_days,
        "source": "nasa_exoplanet_archive_dr25_koi",
        "source_row_id": source_row_id,
        "source_table": "cumulative",
        "split": split,
        "target_id": target_id,
        "target_name": f"KIC {target_id}",
    }


def _write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _flat_lc_fetcher(n_points: int = 2000):
    """Return a fetcher yielding n_points of flux with a periodic dip."""
    times = [float(i) * 0.0208333 for i in range(n_points)]  # ~30-min cadence
    flux = [1.0 - (0.01 if abs(i % 200 - 100) < 5 else 0.0) for i in range(n_points)]

    def _fetch(target_id: int):
        return times, flux

    return _fetch


# ---------------------------------------------------------------------------
# load_manifest_rows / group_rows_by_target
# ---------------------------------------------------------------------------


class TestLoadManifestRows:
    def test_parses_jsonl(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.jsonl"
        _write_manifest(path, [_manifest_row(), _manifest_row(source_row_id="K00752.02")])
        rows = load_manifest_rows(path)
        assert len(rows) == 2
        assert rows[0]["target_id"] == 10797460

    def test_skips_blank_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.jsonl"
        path.write_text(json.dumps(_manifest_row()) + "\n\n\n")
        rows = load_manifest_rows(path)
        assert len(rows) == 1


class TestGroupRowsByTarget:
    def test_groups_multiple_rows_per_target(self) -> None:
        rows = [
            _manifest_row(target_id=1, source_row_id="a"),
            _manifest_row(target_id=1, source_row_id="b"),
            _manifest_row(target_id=2, source_row_id="c"),
        ]
        groups = group_rows_by_target(rows)
        assert set(groups.keys()) == {1, 2}
        assert len(groups[1]) == 2
        assert len(groups[2]) == 1

    def test_sorted_by_target_id(self) -> None:
        rows = [_manifest_row(target_id=30), _manifest_row(target_id=10)]
        groups = group_rows_by_target(rows)
        assert list(groups.keys()) == [10, 30]


# ---------------------------------------------------------------------------
# T1KeplerProcessingStore
# ---------------------------------------------------------------------------


class TestT1KeplerProcessingStore:
    def test_new_store_has_no_done_targets(self, tmp_path: Path) -> None:
        store = T1KeplerProcessingStore(tmp_path / "progress.sqlite3")
        assert store.done_target_ids() == set()

    def test_mark_active_then_done(self, tmp_path: Path) -> None:
        store = T1KeplerProcessingStore(tmp_path / "progress.sqlite3")
        store.mark_active(123, 2)
        assert 123 in store.active_target_ids()
        assert 123 not in store.done_target_ids()

        store.mark_done(123, n_written=2, n_failed=0, flag="OK")
        assert 123 in store.done_target_ids()
        assert 123 not in store.active_target_ids()

    def test_mark_active_is_idempotent(self, tmp_path: Path) -> None:
        store = T1KeplerProcessingStore(tmp_path / "progress.sqlite3")
        store.mark_active(1, 5)
        store.mark_active(1, 5)  # must not raise (upsert)
        assert store.active_target_ids() == {1}

    def test_summary_counts(self, tmp_path: Path) -> None:
        store = T1KeplerProcessingStore(tmp_path / "progress.sqlite3")
        store.mark_active(1, 2)
        store.mark_done(1, n_written=2, n_failed=0, flag="OK")
        store.mark_active(2, 1)
        summary = store.summary()
        assert summary["n_targets"] == 2
        assert summary["n_done"] == 1
        assert summary["n_active"] == 1
        assert summary["n_written"] == 2

    def test_reopening_store_preserves_state(self, tmp_path: Path) -> None:
        db_path = tmp_path / "progress.sqlite3"
        store1 = T1KeplerProcessingStore(db_path)
        store1.mark_active(1, 1)
        store1.mark_done(1, n_written=1, n_failed=0, flag="OK")

        store2 = T1KeplerProcessingStore(db_path)
        assert store2.done_target_ids() == {1}


# ---------------------------------------------------------------------------
# process_target
# ---------------------------------------------------------------------------


class TestProcessTarget:
    def test_ok_flag_writes_one_record_per_row(self) -> None:
        rows = [_manifest_row(source_row_id="a"), _manifest_row(source_row_id="b", period_days=5.0)]
        result = process_target(10797460, rows, lc_fetcher=_flat_lc_fetcher(), n_bins=51)
        assert isinstance(result, TargetResult)
        assert result.flag == "OK"
        assert len(result.records) == 2
        assert result.n_failed == 0
        assert all(len(r["flux"]) == 51 for r in result.records)

    def test_none_fetch_result_flags_no_data_or_no_lightkurve(self) -> None:
        rows = [_manifest_row()]
        result = process_target(1, rows, lc_fetcher=lambda _tid: None)
        assert result.flag in {"NO_LIGHTKURVE", "NO_DATA"}
        assert result.n_failed == 1
        assert result.records == ()

    def test_exception_flags_error(self) -> None:
        def _raise(_tid: int):
            raise RuntimeError("connection refused")

        rows = [_manifest_row()]
        result = process_target(1, rows, lc_fetcher=_raise)
        assert result.flag.startswith("ERROR")
        assert result.n_failed == 1

    def test_too_few_points_marks_row_failed_not_ok(self) -> None:
        rows = [_manifest_row()]
        result = process_target(
            1, rows, lc_fetcher=lambda _tid: ([0.0, 1.0], [1.0, 1.0]), n_bins=201
        )
        assert result.n_failed == 1
        assert result.records == ()

    def test_invalid_period_marks_row_failed(self) -> None:
        rows = [_manifest_row(period_days=0.0)]
        result = process_target(1, rows, lc_fetcher=_flat_lc_fetcher(), n_bins=51)
        assert result.n_failed == 1
        assert result.records == ()

    def test_record_carries_manifest_fields(self) -> None:
        rows = [_manifest_row(source_row_id="K00752.01", split="test", label=0)]
        result = process_target(10797460, rows, lc_fetcher=_flat_lc_fetcher(), n_bins=51)
        record = result.records[0]
        assert record["source_row_id"] == "K00752.01"
        assert record["split"] == "test"
        assert record["label"] == 0
        assert record["target_id"] == 10797460

    def test_one_fetch_serves_multiple_rows_same_target(self) -> None:
        calls: list[int] = []

        def _counting_fetcher(target_id: int):
            calls.append(target_id)
            return _flat_lc_fetcher()(target_id)

        rows = [_manifest_row(source_row_id="a"), _manifest_row(source_row_id="b", period_days=5.0)]
        process_target(10797460, rows, lc_fetcher=_counting_fetcher, n_bins=51)
        assert calls == [10797460]  # fetched once, not once per manifest row


# ---------------------------------------------------------------------------
# directory helpers
# ---------------------------------------------------------------------------


class TestDirectoryHelpers:
    def test_clear_directory_removes_files_and_subdirs(self, tmp_path: Path) -> None:
        (tmp_path / "a.fits").write_text("x")
        sub = tmp_path / "mastDownload"
        sub.mkdir()
        (sub / "b.fits").write_text("y")

        _clear_directory(tmp_path)

        assert tmp_path.exists()
        assert list(tmp_path.iterdir()) == []

    def test_clear_directory_missing_dir_does_not_raise(self, tmp_path: Path) -> None:
        _clear_directory(tmp_path / "does_not_exist")

    def test_directory_size_bytes(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_bytes(b"x" * 100)
        assert _directory_size_bytes(tmp_path) == 100

    def test_directory_size_bytes_missing_dir_is_zero(self, tmp_path: Path) -> None:
        assert _directory_size_bytes(tmp_path / "nope") == 0


# ---------------------------------------------------------------------------
# run_batch
# ---------------------------------------------------------------------------


class TestRunBatch:
    def _manifest_with_targets(self, tmp_path: Path, n_targets: int = 3) -> Path:
        rows = [
            _manifest_row(target_id=1000 + i, source_row_id=f"row{i}") for i in range(n_targets)
        ]
        path = tmp_path / "manifest.jsonl"
        _write_manifest(path, rows)
        return path

    def test_writes_one_snippet_per_target(self, tmp_path: Path) -> None:
        manifest_path = self._manifest_with_targets(tmp_path, 3)
        summary = run_batch(
            manifest_path=manifest_path,
            output_path=tmp_path / "out" / "snippets.jsonl",
            db_path=tmp_path / "progress.sqlite3",
            raw_dir=tmp_path / "raw",
            max_targets=None,
            n_bins=51,
            lc_fetcher=_flat_lc_fetcher(),
        )
        assert isinstance(summary, BatchSummary)
        assert summary.n_targets_total == 3
        assert summary.n_targets_processed_this_run == 3
        assert summary.n_snippets_written == 3

        lines = (tmp_path / "out" / "snippets.jsonl").read_text().splitlines()
        assert len(lines) == 3

    def test_max_targets_bounds_this_run(self, tmp_path: Path) -> None:
        manifest_path = self._manifest_with_targets(tmp_path, 5)
        summary = run_batch(
            manifest_path=manifest_path,
            output_path=tmp_path / "out" / "snippets.jsonl",
            db_path=tmp_path / "progress.sqlite3",
            raw_dir=tmp_path / "raw",
            max_targets=2,
            n_bins=51,
            lc_fetcher=_flat_lc_fetcher(),
        )
        assert summary.n_targets_total == 5
        assert summary.n_targets_processed_this_run == 2

    def test_resume_skips_already_done_targets(self, tmp_path: Path) -> None:
        manifest_path = self._manifest_with_targets(tmp_path, 3)
        output_path = tmp_path / "out" / "snippets.jsonl"
        db_path = tmp_path / "progress.sqlite3"

        run_batch(
            manifest_path=manifest_path, output_path=output_path, db_path=db_path,
            raw_dir=tmp_path / "raw", max_targets=2, n_bins=51,
            lc_fetcher=_flat_lc_fetcher(),
        )
        summary2 = run_batch(
            manifest_path=manifest_path, output_path=output_path, db_path=db_path,
            raw_dir=tmp_path / "raw", max_targets=None, n_bins=51,
            lc_fetcher=_flat_lc_fetcher(),
        )
        assert summary2.n_targets_skipped_done == 2
        assert summary2.n_targets_processed_this_run == 1

        lines = output_path.read_text().splitlines()
        assert len(lines) == 3  # no duplicates across the two runs

    def test_multiple_rows_per_target_all_written(self, tmp_path: Path) -> None:
        rows = [
            _manifest_row(target_id=1, source_row_id="a"),
            _manifest_row(target_id=1, source_row_id="b", period_days=5.0),
        ]
        manifest_path = tmp_path / "manifest.jsonl"
        _write_manifest(manifest_path, rows)
        summary = run_batch(
            manifest_path=manifest_path,
            output_path=tmp_path / "out" / "snippets.jsonl",
            db_path=tmp_path / "progress.sqlite3",
            raw_dir=tmp_path / "raw",
            max_targets=None,
            n_bins=51,
            lc_fetcher=_flat_lc_fetcher(),
        )
        assert summary.n_targets_total == 1
        assert summary.n_snippets_written == 2

    def test_progress_fn_called(self, tmp_path: Path) -> None:
        manifest_path = self._manifest_with_targets(tmp_path, 2)
        messages: list[str] = []
        run_batch(
            manifest_path=manifest_path,
            output_path=tmp_path / "out" / "snippets.jsonl",
            db_path=tmp_path / "progress.sqlite3",
            raw_dir=tmp_path / "raw",
            max_targets=None,
            n_bins=51,
            lc_fetcher=_flat_lc_fetcher(),
            progress_fn=messages.append,
        )
        assert len(messages) > 0
        assert any("KIC" in m for m in messages)

    def test_progress_includes_eta_for_multi_target_run(self, tmp_path: Path) -> None:
        manifest_path = self._manifest_with_targets(tmp_path, 3)
        messages: list[str] = []
        run_batch(
            manifest_path=manifest_path,
            output_path=tmp_path / "out" / "snippets.jsonl",
            db_path=tmp_path / "progress.sqlite3",
            raw_dir=tmp_path / "raw",
            max_targets=None,
            n_bins=51,
            lc_fetcher=_flat_lc_fetcher(),
            progress_fn=messages.append,
        )
        completion_messages = [m for m in messages if m.startswith("  ->")]
        assert len(completion_messages) == 3
        assert all("ETA=" in m for m in completion_messages)

    def test_failed_target_marked_done_not_retried(self, tmp_path: Path) -> None:
        manifest_path = self._manifest_with_targets(tmp_path, 1)
        db_path = tmp_path / "progress.sqlite3"
        output_path = tmp_path / "out" / "snippets.jsonl"

        run_batch(
            manifest_path=manifest_path, output_path=output_path, db_path=db_path,
            raw_dir=tmp_path / "raw", max_targets=None, n_bins=51,
            lc_fetcher=lambda _tid: None,  # NO_DATA/NO_LIGHTKURVE
        )
        store = T1KeplerProcessingStore(db_path)
        assert store.done_target_ids() == {1000}

        # Second run must not retry the failed-but-done target.
        calls: list[int] = []
        run_batch(
            manifest_path=manifest_path, output_path=output_path, db_path=db_path,
            raw_dir=tmp_path / "raw", max_targets=None, n_bins=51,
            lc_fetcher=lambda tid: (calls.append(tid), None)[1],
        )
        assert calls == []


# ---------------------------------------------------------------------------
# run_batch with workers > 1
# ---------------------------------------------------------------------------


class TestRunBatchConcurrency:
    def _manifest_with_targets(self, tmp_path: Path, n_targets: int) -> Path:
        rows = [
            _manifest_row(target_id=2000 + i, source_row_id=f"c{i}") for i in range(n_targets)
        ]
        path = tmp_path / "manifest.jsonl"
        _write_manifest(path, rows)
        return path

    def test_all_targets_processed_with_multiple_workers(self, tmp_path: Path) -> None:
        manifest_path = self._manifest_with_targets(tmp_path, 12)
        summary = run_batch(
            manifest_path=manifest_path,
            output_path=tmp_path / "out" / "snippets.jsonl",
            db_path=tmp_path / "progress.sqlite3",
            raw_dir=tmp_path / "raw",
            max_targets=None,
            n_bins=51,
            workers=4,
            request_delay=0.0,
            lc_fetcher=_flat_lc_fetcher(),
        )
        assert summary.n_targets_processed_this_run == 12
        assert summary.n_snippets_written == 12
        lines = (tmp_path / "out" / "snippets.jsonl").read_text().splitlines()
        assert len(lines) == 12
        # No duplicate target_ids in the output.
        target_ids = [json.loads(line)["target_id"] for line in lines]
        assert len(target_ids) == len(set(target_ids))

    def test_resume_works_with_multiple_workers(self, tmp_path: Path) -> None:
        manifest_path = self._manifest_with_targets(tmp_path, 10)
        output_path = tmp_path / "out" / "snippets.jsonl"
        db_path = tmp_path / "progress.sqlite3"

        run_batch(
            manifest_path=manifest_path, output_path=output_path, db_path=db_path,
            raw_dir=tmp_path / "raw", max_targets=4, n_bins=51,
            workers=3, request_delay=0.0, lc_fetcher=_flat_lc_fetcher(),
        )
        summary2 = run_batch(
            manifest_path=manifest_path, output_path=output_path, db_path=db_path,
            raw_dir=tmp_path / "raw", max_targets=None, n_bins=51,
            workers=3, request_delay=0.0, lc_fetcher=_flat_lc_fetcher(),
        )
        assert summary2.n_targets_skipped_done == 4
        assert summary2.n_targets_processed_this_run == 6
        lines = output_path.read_text().splitlines()
        assert len(lines) == 10

    def test_progress_fn_reports_every_completion(self, tmp_path: Path) -> None:
        manifest_path = self._manifest_with_targets(tmp_path, 8)
        messages: list[str] = []
        run_batch(
            manifest_path=manifest_path,
            output_path=tmp_path / "out" / "snippets.jsonl",
            db_path=tmp_path / "progress.sqlite3",
            raw_dir=tmp_path / "raw",
            max_targets=None,
            n_bins=51,
            workers=3,
            request_delay=0.0,
            lc_fetcher=_flat_lc_fetcher(),
            progress_fn=messages.append,
        )
        completion_messages = [m for m in messages if m.startswith("  ->")]
        assert len(completion_messages) == 8

    def test_workers_value_is_clamped_to_at_least_one(self, tmp_path: Path) -> None:
        manifest_path = self._manifest_with_targets(tmp_path, 2)
        # workers=0 must not raise or hang; behaves like workers=1.
        summary = run_batch(
            manifest_path=manifest_path,
            output_path=tmp_path / "out" / "snippets.jsonl",
            db_path=tmp_path / "progress.sqlite3",
            raw_dir=tmp_path / "raw",
            max_targets=None,
            n_bins=51,
            workers=0,
            lc_fetcher=_flat_lc_fetcher(),
        )
        assert summary.n_targets_processed_this_run == 2


# ---------------------------------------------------------------------------
# make_default_lc_fetcher: per-target directory isolation
# ---------------------------------------------------------------------------


class TestMakeDefaultLcFetcherIsolation:
    def test_uses_per_target_subdirectory_and_cleans_it_up(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured_download_dirs: list[str] = []

        class _FakeLc:
            time = type("T", (), {"value": [0.0, 1.0, 2.0]})()
            flux = type("F", (), {"value": [1.0, 1.0, 1.0]})()

            def normalize(self):
                return self

        class _FakeCollection:
            def __len__(self):
                return 1

            def stitch(self):
                return _FakeLc()

        class _FakeSearchResult:
            def __len__(self):
                return 1

            def download_all(self, download_dir):
                captured_download_dirs.append(download_dir)
                return _FakeCollection()

        fake_lk = type(
            "FakeLightkurve",
            (),
            {"search_lightcurve": staticmethod(lambda *a, **k: _FakeSearchResult())},
        )
        monkeypatch.setitem(sys.modules, "lightkurve", fake_lk)

        raw_dir = tmp_path / "raw"
        fetcher = make_default_lc_fetcher(raw_dir)
        result = fetcher(555)

        assert result is not None
        assert captured_download_dirs == [str(raw_dir / "target_555")]
        # Cleaned up after the fetch completes.
        assert not (raw_dir / "target_555").exists()

    def test_concurrent_targets_do_not_collide(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import threading
        import time as time_module

        seen_dirs_while_active: set[str] = set()
        lock = threading.Lock()

        class _FakeLc:
            time = type("T", (), {"value": [0.0, 1.0, 2.0]})()
            flux = type("F", (), {"value": [1.0, 1.0, 1.0]})()

            def normalize(self):
                return self

        class _FakeCollection:
            def __len__(self):
                return 1

            def stitch(self):
                return _FakeLc()

        class _FakeSearchResult:
            def __len__(self):
                return 1

            def download_all(self, download_dir):
                with lock:
                    seen_dirs_while_active.add(download_dir)
                # Hold the "download" open briefly so other threads overlap.
                time_module.sleep(0.05)
                assert Path(download_dir).exists(), "directory vanished mid-download"
                return _FakeCollection()

        fake_lk = type(
            "FakeLightkurve",
            (),
            {"search_lightcurve": staticmethod(lambda *a, **k: _FakeSearchResult())},
        )
        monkeypatch.setitem(sys.modules, "lightkurve", fake_lk)

        raw_dir = tmp_path / "raw"
        fetcher = make_default_lc_fetcher(raw_dir)

        threads = [threading.Thread(target=fetcher, args=(tid,)) for tid in (1, 2, 3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert seen_dirs_while_active == {
            str(raw_dir / "target_1"),
            str(raw_dir / "target_2"),
            str(raw_dir / "target_3"),
        }


# ---------------------------------------------------------------------------
# _format_eta
# ---------------------------------------------------------------------------


class TestFormatEta:
    def test_short_duration_in_seconds(self) -> None:
        assert _format_eta(45) == "45s"

    def test_boundary_at_90_seconds_stays_in_seconds(self) -> None:
        assert _format_eta(90) == "90s"

    def test_long_duration_in_minutes_seconds(self) -> None:
        assert _format_eta(125) == "2m05s"

    def test_hours_scale_duration(self) -> None:
        assert _format_eta(12160) == "202m40s"

    def test_infinite_is_unknown(self) -> None:
        assert _format_eta(float("inf")) == "unknown"

    def test_nan_is_unknown(self) -> None:
        assert _format_eta(float("nan")) == "unknown"

    def test_zero_is_zero_seconds(self) -> None:
        assert _format_eta(0) == "0s"


# ---------------------------------------------------------------------------
# format_batch_summary
# ---------------------------------------------------------------------------


class TestFormatBatchSummary:
    def test_contains_key_counts(self) -> None:
        summary = BatchSummary(
            n_targets_total=10, n_targets_processed_this_run=5, n_targets_skipped_done=5,
            n_snippets_written=7, n_rows_failed=1, elapsed_seconds=12.3,
            output_path="out.jsonl", db_path="db.sqlite3",
        )
        text = format_batch_summary(summary)
        assert "10" in text
        assert "out.jsonl" in text
        assert "db.sqlite3" in text
