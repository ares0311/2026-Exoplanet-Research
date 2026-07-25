"""Tests for src/exo_toolkit/hunter_history.py's manual-scan bridge."""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from exo_toolkit.hunter_history import build_manual_scan_source
from exo_toolkit.search_lifecycle import HunterStore

_STARTED = datetime(2026, 7, 24, tzinfo=UTC)
_COMPLETED = datetime(2026, 7, 24, 1, tzinfo=UTC)


def _entry(tic_id: int, status: str = "candidate_found") -> dict[str, object]:
    return {
        "target_id": f"TIC {tic_id}",
        "status": status,
        "searched_at": _COMPLETED.isoformat(),
        "mission": "TESS",
        "ranking_score": 0.5,
        "metrics": {"best_fpp": 0.1},
        "result": {"n_signals": 1},
        "error_message": "boom" if status == "failed" else None,
    }


class TestBuildManualScanSource:
    def test_missing_log_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="does not exist"):
            build_manual_scan_source(
                script="star_scanner",
                log_path=tmp_path / "missing.json",
                entries=[_entry(1)],
                started_at=_STARTED,
                completed_at=_COMPLETED,
                method_or_data="TESS QLP manual scan",
                source_root=tmp_path,
            )

    def test_empty_entries_raises(self, tmp_path: Path) -> None:
        log = tmp_path / "scan_log.json"
        log.write_text("{}", encoding="utf-8")
        with pytest.raises(ValueError, match="at least one entry"):
            build_manual_scan_source(
                script="star_scanner",
                log_path=log,
                entries=[],
                started_at=_STARTED,
                completed_at=_COMPLETED,
                method_or_data="TESS QLP manual scan",
                source_root=tmp_path,
            )

    def test_computes_relative_source_path_and_hash(self, tmp_path: Path) -> None:
        log = tmp_path / "logs" / "scan_log.json"
        log.parent.mkdir(parents=True)
        log.write_text('{"entries": {}}', encoding="utf-8")

        source = build_manual_scan_source(
            script="star_scanner",
            log_path=log,
            entries=[_entry(1)],
            started_at=_STARTED,
            completed_at=_COMPLETED,
            method_or_data="TESS QLP manual scan",
            source_root=tmp_path,
        )

        assert source["source_path"] == "logs/scan_log.json"
        assert len(source["source_sha256"]) == 64
        assert source["mode"] == "new"
        assert source["searched_by"] == "manual_cli:star_scanner"
        assert source["search_id"].startswith("manual-star_scanner-")
        assert source["entries"] == [_entry(1)]

    def test_search_id_deterministic_for_identical_log_bytes(
        self, tmp_path: Path
    ) -> None:
        log = tmp_path / "scan_log.json"
        log.write_text('{"entries": {"1": {}}}', encoding="utf-8")
        kwargs = {
            "script": "batch_scan",
            "log_path": log,
            "entries": [_entry(1)],
            "started_at": _STARTED,
            "completed_at": _COMPLETED,
            "method_or_data": "TESS batch scan",
            "source_root": tmp_path,
        }
        first = build_manual_scan_source(**kwargs)
        second = build_manual_scan_source(**kwargs)
        assert first["search_id"] == second["search_id"]
        assert first["source_sha256"] == second["source_sha256"]

    def test_different_log_bytes_produce_different_search_id(
        self, tmp_path: Path
    ) -> None:
        log = tmp_path / "scan_log.json"
        log.write_text('{"entries": {"1": {}}}', encoding="utf-8")
        first = build_manual_scan_source(
            script="batch_scan",
            log_path=log,
            entries=[_entry(1)],
            started_at=_STARTED,
            completed_at=_COMPLETED,
            method_or_data="TESS batch scan",
            source_root=tmp_path,
        )
        log.write_text('{"entries": {"1": {}, "2": {}}}', encoding="utf-8")
        second = build_manual_scan_source(
            script="batch_scan",
            log_path=log,
            entries=[_entry(1), _entry(2)],
            started_at=_STARTED,
            completed_at=_COMPLETED,
            method_or_data="TESS batch scan",
            source_root=tmp_path,
        )
        assert first["search_id"] != second["search_id"]


class TestManualScanSourceImportsIntoHunterStore:
    def test_round_trips_through_import_history_manifest(self, tmp_path: Path) -> None:
        log = tmp_path / "scan_log.json"
        log.write_text('{"entries": {"42": {}}}', encoding="utf-8")
        source = build_manual_scan_source(
            script="star_scanner",
            log_path=log,
            entries=[_entry(42)],
            started_at=_STARTED,
            completed_at=_COMPLETED,
            method_or_data="TESS QLP manual scan",
            source_root=tmp_path,
        )
        store = HunterStore(tmp_path / "hunter.sqlite3")

        summary = store.import_history_manifest(
            {"schema_version": 1, "sources": [source]},
            source_root=tmp_path,
        )

        assert summary["sources_created"] == 1
        assert "TIC 42" in store.searched_target_ids()
        history = store.target_history("TIC 42")
        assert history[0]["status"] == "candidate_found"

    def test_unchanged_log_is_idempotent_on_reimport(self, tmp_path: Path) -> None:
        log = tmp_path / "scan_log.json"
        log.write_text('{"entries": {"7": {}}}', encoding="utf-8")
        source = build_manual_scan_source(
            script="star_scanner",
            log_path=log,
            entries=[_entry(7)],
            started_at=_STARTED,
            completed_at=_COMPLETED,
            method_or_data="TESS QLP manual scan",
            source_root=tmp_path,
        )
        store = HunterStore(tmp_path / "hunter.sqlite3")
        manifest = {"schema_version": 1, "sources": [source]}

        store.import_history_manifest(manifest, source_root=tmp_path)
        second = store.import_history_manifest(manifest, source_root=tmp_path)

        assert second["sources_created"] == 0
        assert len(store.target_history("TIC 7")) == 1
