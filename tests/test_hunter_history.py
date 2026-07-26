"""Tests for src/exo_toolkit/hunter_history.py's manual-scan bridge."""
from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pytest

from exo_toolkit.hunter_history import (
    build_manual_scan_source,
    load_verified_history_manifest,
    resolve_history_source_path,
)
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


def _make_decoy_repo(tmp_path: Path) -> tuple[Path, Path]:
    """Build a decoy repo (pyproject.toml + src/) with a subdir to chdir
    into, matching the pattern already used to prove the equivalent
    file-manifest walk-up bug in test_hunter_cli.py."""
    decoy_repo = tmp_path / "decoy_repo"
    subdir = decoy_repo / "subdir"
    subdir.mkdir(parents=True)
    (decoy_repo / "pyproject.toml").write_text("[project]\nname='decoy'\n", encoding="utf-8")
    (decoy_repo / "src").mkdir()
    return decoy_repo, subdir


class TestDictManifestSourceRootWalkUp:
    """The dict-manifest branch previously resolved a bare Path.cwd() with
    no attempt to locate the real repo root -- dormant against every
    current caller (both already pass source_root explicitly) but the
    same latent trap already demonstrated for the file-manifest branch."""

    def test_resolve_history_source_path_walks_up_cwd_for_repo_root(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        decoy_repo, subdir = _make_decoy_repo(tmp_path)
        source_file = decoy_repo / "logs" / "prior.json"
        source_file.parent.mkdir(parents=True)
        source_file.write_text("evidence", encoding="utf-8")

        monkeypatch.chdir(subdir)
        resolved = resolve_history_source_path(
            {"schema_version": 1, "sources": []},
            "logs/prior.json",
        )

        # Must resolve against decoy_repo (found by walking up from cwd),
        # not against cwd itself (subdir), which has no logs/ directory.
        assert resolved == source_file.resolve()

    def test_resolve_history_source_path_explicit_source_root_overrides_walkup(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _decoy_repo, subdir = _make_decoy_repo(tmp_path)
        intended_dir = tmp_path / "intended"
        intended_dir.mkdir()
        (intended_dir / "prior.json").write_text("evidence", encoding="utf-8")

        monkeypatch.chdir(subdir)
        resolved = resolve_history_source_path(
            {"schema_version": 1, "sources": []},
            "prior.json",
            source_root=intended_dir,
        )

        assert resolved == (intended_dir / "prior.json").resolve()

    def test_load_verified_history_manifest_dict_walks_up_cwd_for_repo_root(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        decoy_repo, subdir = _make_decoy_repo(tmp_path)
        source_file = decoy_repo / "logs" / "prior.json"
        source_file.parent.mkdir(parents=True)
        source_file.write_text("evidence", encoding="utf-8")
        sha = hashlib.sha256(source_file.read_bytes()).hexdigest()
        manifest = {
            "schema_version": 1,
            "sources": [
                {
                    "search_id": "historical-new-search",
                    "source_path": "logs/prior.json",
                    "source_sha256": sha,
                }
            ],
        }

        monkeypatch.chdir(subdir)
        # Without this fix, resolution would be attempted against subdir
        # (bare Path.cwd()), which has no logs/ directory, and would fail
        # closed with a "does not exist" ValueError instead of succeeding
        # against the correctly-discovered decoy_repo root.
        result = load_verified_history_manifest(manifest)

        assert result == manifest

    def test_load_verified_history_manifest_dict_missing_source_fails_closed_from_wrong_cwd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A plain, non-repo cwd (no pyproject.toml/src/ anywhere above it)
        # must still fail closed rather than silently succeed.
        plain_dir = tmp_path / "plain"
        plain_dir.mkdir()
        manifest = {
            "schema_version": 1,
            "sources": [
                {
                    "search_id": "s1",
                    "source_path": "logs/prior.json",
                    "source_sha256": "a" * 64,
                }
            ],
        }
        monkeypatch.chdir(plain_dir)
        with pytest.raises(ValueError, match="does not exist"):
            load_verified_history_manifest(manifest)
