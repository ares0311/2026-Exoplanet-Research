"""Cross-project Hunter history federation: fail-closed novelty eligibility.

Covers the five required regression cases -- absent export, malformed
export, wrong schema version, one degraded source, and a positive case that
permits selection -- plus the publish path and IDENT-04 persistence.

Every test builds its own exports on disk. Nothing here reads a real sibling
repository, so the suite passes whether or not the siblings are checked out.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from exo_toolkit.hunter_cross_project_history import (
    CROSS_PROJECT_DECISION_STATES,
    CROSS_PROJECT_HISTORY_SCHEMA_VERSION,
    CROSS_PROJECT_ROOT_NAMES,
    SCANNER_STATUS_TO_CROSS_PROJECT,
    cross_project_alias_counts,
    cross_project_evidence_by_alias,
    cross_project_history_federation_validity,
    cross_project_history_validity,
    export_cross_project_history,
    load_cross_project_history_export,
    own_history_export_path,
    sibling_history_export_path,
    write_cross_project_history_export,
)
from exo_toolkit.hunter_models import HunterCandidate, PriorSearch
from exo_toolkit.search_lifecycle import HunterStore


def _write_export(
    repo_root: Path,
    *,
    source_file: str = "logs/run.json",
    source_bytes: bytes | None = b'{"run": 1}',
    schema_version: int = CROSS_PROJECT_HISTORY_SCHEMA_VERSION,
    statuses: tuple[str, ...] = ("candidate_found",),
    extra_sources: int = 0,
    sha_override: str | None = None,
) -> Path:
    """Build a self-consistent export inside ``repo_root/data_selection``."""
    origin = repo_root / source_file
    origin.parent.mkdir(parents=True, exist_ok=True)
    if source_bytes is not None:
        origin.write_bytes(source_bytes)
    digest = sha_override or hashlib.sha256(source_bytes or b"").hexdigest()

    def _source(index: int, path: str, sha: str) -> dict[str, Any]:
        return {
            "search_id": f"probe-{index}",
            "mode": "new",
            "started_at": "2026-01-01T00:00:00+00:00",
            "completed_at": "2026-01-02T00:00:00+00:00",
            "searched_by": "Probe-Hunter",
            "source_project": "probe project",
            "method_or_data": "probe",
            "source_path": path,
            "source_sha256": sha,
            "provenance_uri": f"local-artifact:{path}#sha256={sha}",
            "entries": [
                {
                    "target_id": f"TIC {100 + offset}",
                    "canonical_id": f"TIC {100 + offset}",
                    "mission": "TESS",
                    "status": status,
                    "searched_at": "2026-01-01T12:00:00+00:00",
                }
                for offset, status in enumerate(statuses)
            ],
        }

    sources = [_source(0, source_file, digest)]
    for index in range(1, extra_sources + 1):
        extra_path = f"logs/extra_{index}.json"
        extra_origin = repo_root / extra_path
        extra_origin.parent.mkdir(parents=True, exist_ok=True)
        payload = f'{{"extra": {index}}}'.encode()
        extra_origin.write_bytes(payload)
        sources.append(
            _source(index, extra_path, hashlib.sha256(payload).hexdigest())
        )

    export = repo_root / "data_selection" / "hunter_prior_search_history_v1.json"
    export.parent.mkdir(parents=True, exist_ok=True)
    export.write_text(
        json.dumps(
            {
                "schema_version": schema_version,
                "manifest_id": "hunter-prior-search-history-v1",
                "description": "probe",
                "sources": sources,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return export


# ---------------------------------------------------------------- positive


def test_intact_export_is_valid_and_permits_selection(tmp_path: Path) -> None:
    export = _write_export(tmp_path / "repo")
    state, detail, payload = cross_project_history_validity(export)
    assert state == "valid"
    assert state in CROSS_PROJECT_DECISION_STATES
    assert payload is not None
    assert "valid" in detail


def test_validity_is_stamped_per_source_and_per_entry(tmp_path: Path) -> None:
    """There is deliberately no top-level validity field."""
    export = _write_export(
        tmp_path / "repo", statuses=("candidate_found", "no_data", "weird_status")
    )
    payload = load_cross_project_history_export(export)
    assert "validity_state" not in payload
    source = payload["sources"][0]
    assert source["validity_state"] == "valid"
    entry_states = [entry["validity_state"] for entry in source["entries"]]
    # completed -> inherits source; invalid status -> invalid; unrecognized -> unknown
    assert entry_states == ["valid", "invalid", "unknown"]


def test_only_decision_grade_entries_count_as_prior_search_evidence(
    tmp_path: Path,
) -> None:
    export = _write_export(
        tmp_path / "repo", statuses=("candidate_found", "no_data", "failed")
    )
    payload = load_cross_project_history_export(export)
    counts = cross_project_alias_counts(payload)
    # Only the completed search is evidence; no_data/failed searched nothing.
    assert dict(counts) == {"TIC100": 1}
    evidence = cross_project_evidence_by_alias(payload)
    assert list(evidence) == ["TIC100"]
    assert evidence["TIC100"][0]["validity_state"] == "valid"


# ---------------------------------------------------------------- negative


def test_absent_export_is_unknown_never_valid(tmp_path: Path) -> None:
    state, detail, payload = cross_project_history_validity(tmp_path / "nope.json")
    assert state == "unknown"
    assert state not in CROSS_PROJECT_DECISION_STATES
    assert payload is None
    assert "absent" in detail


def test_malformed_export_is_invalid(tmp_path: Path) -> None:
    bad = tmp_path / "data_selection" / "hunter_prior_search_history_v1.json"
    bad.parent.mkdir(parents=True)
    bad.write_text("{ not json", encoding="utf-8")
    state, _detail, payload = cross_project_history_validity(bad)
    assert state == "invalid"
    assert payload is None


def test_wrong_schema_version_is_invalid(tmp_path: Path) -> None:
    export = _write_export(tmp_path / "repo", schema_version=99)
    state, detail, _payload = cross_project_history_validity(export)
    assert state == "invalid"
    assert "schema_version" in detail


def test_changed_origin_file_is_refresh_required(tmp_path: Path) -> None:
    """A present-but-modified source must not read as current."""
    repo = tmp_path / "repo"
    export = _write_export(repo)
    (repo / "logs" / "run.json").write_bytes(b'{"run": 2}')  # drifted
    state, _detail, _payload = cross_project_history_validity(export)
    assert state == "invalid"  # load raises on refresh-required sources


def test_absent_origin_file_is_stale_but_usable(tmp_path: Path) -> None:
    """An operator-copied export keeps working but is never called current."""
    repo = tmp_path / "repo"
    export = _write_export(repo)
    (repo / "logs" / "run.json").unlink()
    state, _detail, _payload = cross_project_history_validity(export)
    assert state == "stale-but-usable"
    assert state in CROSS_PROJECT_DECISION_STATES


def test_one_degraded_source_degrades_the_whole_export(tmp_path: Path) -> None:
    """An export is only as trustworthy as its weakest source."""
    repo = tmp_path / "repo"
    export = _write_export(repo, extra_sources=2)
    assert cross_project_history_validity(export)[0] == "valid"
    # Remove one of three origin files: that source becomes stale-but-usable.
    (repo / "logs" / "extra_1.json").unlink()
    state, detail, _payload = cross_project_history_validity(export)
    assert state == "stale-but-usable"
    assert "stale-but-usable" in detail


def test_unversioned_export_is_not_decision_grade(tmp_path: Path) -> None:
    """Absence of a label is not evidence."""
    export = tmp_path / "data_selection" / "hunter_prior_search_history_v1.json"
    export.parent.mkdir(parents=True)
    export.write_text(json.dumps({"sources": []}), encoding="utf-8")
    assert cross_project_history_validity(export)[0] == "invalid"


# ------------------------------------------------------------- federation


def _federated(monkeypatch: pytest.MonkeyPatch, roots: dict[str, Path]) -> None:
    """Point sibling discovery at throwaway roots without touching real repos."""
    monkeypatch.setattr(
        "exo_toolkit.hunter_cross_project_history.sibling_history_export_path",
        lambda project: roots[project] / "data_selection"
        / "hunter_prior_search_history_v1.json",
    )


def test_federation_requires_all_three_projects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    own = _write_export(tmp_path / "own")
    roots = {name: tmp_path / name for name in CROSS_PROJECT_ROOT_NAMES}
    for root in roots.values():
        _write_export(root)
    _federated(monkeypatch, roots)

    state, _detail, per_project = cross_project_history_federation_validity(own)
    assert set(per_project) == {"exo_hunter", *CROSS_PROJECT_ROOT_NAMES}
    assert state == "valid"


def test_missing_sibling_fails_the_federation_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    own = _write_export(tmp_path / "own")
    roots = {name: tmp_path / name for name in CROSS_PROJECT_ROOT_NAMES}
    _write_export(roots["techno_hunter"])
    # neo_hunter deliberately never publishes.
    _federated(monkeypatch, roots)

    state, detail, per_project = cross_project_history_federation_validity(own)
    assert per_project["neo_hunter"][0] == "unknown"
    assert state == "unknown"
    assert state not in CROSS_PROJECT_DECISION_STATES
    assert "neo_hunter=unknown" in detail


def test_degraded_sibling_degrades_the_federation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    own = _write_export(tmp_path / "own")
    roots = {name: tmp_path / name for name in CROSS_PROJECT_ROOT_NAMES}
    for root in roots.values():
        _write_export(root)
    (roots["techno_hunter"] / "logs" / "run.json").unlink()
    _federated(monkeypatch, roots)

    state, _detail, per_project = cross_project_history_federation_validity(own)
    assert per_project["techno_hunter"][0] == "stale-but-usable"
    assert state == "stale-but-usable"


def test_sibling_discovery_is_repo_relative_and_never_hardcoded() -> None:
    """WS-03: discovery is computed from this repo's own location."""
    repo_root = own_history_export_path().parents[1]
    for project, root_name in CROSS_PROJECT_ROOT_NAMES.items():
        path = sibling_history_export_path(project)
        assert path == (
            repo_root.parent / root_name / "data_selection"
            / "hunter_prior_search_history_v1.json"
        )
        # Never inside this repository -- consumption is read-only and external.
        assert repo_root not in path.parents
    with pytest.raises(ValueError, match="unknown sibling project"):
        sibling_history_export_path("not_a_project")


# ------------------------------------------------------------------- gate


def test_new_gate_refuses_without_decision_grade_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from exo_toolkit import hunter_cli

    own = _write_export(tmp_path / "own")
    roots = {name: tmp_path / name for name in CROSS_PROJECT_ROOT_NAMES}
    _write_export(roots["techno_hunter"])
    _federated(monkeypatch, roots)

    with pytest.raises(RuntimeError) as excinfo:
        hunter_cli._require_decision_grade_history(own)
    message = str(excinfo.value)
    # Actionable: names the blocking project and what to do about it.
    assert "neo_hunter" in message
    assert "data_selection/hunter_prior_search_history_v1.json" in message
    assert "fails closed" in message


def test_new_gate_permits_selection_when_all_three_are_decision_grade(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from exo_toolkit import hunter_cli

    own = _write_export(tmp_path / "own")
    roots = {name: tmp_path / name for name in CROSS_PROJECT_ROOT_NAMES}
    for root in roots.values():
        _write_export(root)
    _federated(monkeypatch, roots)

    state, detail = hunter_cli._require_decision_grade_history(own)
    assert state == "valid"
    assert "exo_hunter=valid" in detail


# ---------------------------------------------------------------- IDENT-04


def test_frozen_targets_record_history_validity_and_source(tmp_path: Path) -> None:
    store = HunterStore(tmp_path / "hunter.sqlite3")
    store.initialize()
    candidate = HunterCandidate(
        target_id="TIC 424242",
        canonical_id="TIC 424242",
        aliases=("424242",),
        source="test catalog",
        source_provenance={"search_category": "new"},
        estimated_download_gb=0.001,
        ranking_score=1.0,
        selection_reason="deterministic test rank",
        metrics={"priority": 1.0},
    )
    search = store.create_search(
        [candidate],
        requested_target_count=1,
        mode="new",
        selector_version="test_v1",
        config={
            "cross_project_history_validity": "stale-but-usable",
            "cross_project_history_source": "exo_hunter=stale-but-usable; ...",
        },
    )
    with sqlite3.connect(tmp_path / "hunter.sqlite3") as connection:
        row = connection.execute(
            "SELECT cross_project_history_validity, cross_project_history_source "
            "FROM search_manifest_targets WHERE search_id=?",
            (search["search_id"],),
        ).fetchone()
    assert row[0] == "stale-but-usable"
    assert row[1].startswith("exo_hunter=stale-but-usable")


def test_follow_up_search_records_no_novelty_claim(tmp_path: Path) -> None:
    """Follow-up mode makes no novelty claim, so it stores no evidence."""
    store = HunterStore(tmp_path / "hunter.sqlite3")
    store.initialize()
    candidate = HunterCandidate(
        target_id="TIC 515151",
        canonical_id="TIC 515151",
        aliases=("515151",),
        source="test catalog",
        source_provenance={"search_category": "follow-up"},
        estimated_download_gb=0.001,
        ranking_score=1.0,
        selection_reason="deterministic test rank",
        metrics={"priority": 1.0},
        prior_searches=(
            PriorSearch(
                searched_by="EXO-Hunter",
                searched_at=datetime(2026, 1, 1, tzinfo=UTC),
                source_project="2026 Exoplanet Research",
                method_or_data="TESS QLP",
                result="candidate_found",
                provenance_uri="local-artifact:logs/run.json#sha256=deadbeef",
            ),
        ),
    )
    search = store.create_search(
        [candidate],
        requested_target_count=1,
        mode="follow-up",
        selector_version="test_v1",
        config={},
    )
    with sqlite3.connect(tmp_path / "hunter.sqlite3") as connection:
        row = connection.execute(
            "SELECT cross_project_history_validity, cross_project_history_source "
            "FROM search_manifest_targets WHERE search_id=?",
            (search["search_id"],),
        ).fetchone()
    assert row == (None, None)


# ---------------------------------------------------------------- publish


def _scan_log(path: Path, records: dict[str, dict[str, Any]], updated: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"last_updated": updated, "entries": records}), encoding="utf-8"
    )


def test_publish_builds_a_self_consistent_valid_export(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _scan_log(
        repo / "logs" / "discovery_run_001.json",
        {
            "5": {
                "tic_id": 5,
                "scanned_at": "2026-01-01T00:00:00+00:00",
                "status": "candidate_found",
                "n_signals": 2,
                "priority_score": 0.5,
            },
            "3": {
                "tic_id": 3,
                "scanned_at": "2026-01-01T00:00:01+00:00",
                "status": "scanned_clear",
                "n_signals": 0,
                "priority_score": 0.2,
            },
        },
        "2026-01-02T00:00:00+00:00",
    )
    summary = write_cross_project_history_export(repo_root=repo)
    assert summary["ok"] is True
    assert summary["entry_count"] == 2
    assert summary["unique_target_count"] == 2

    export = repo / "data_selection" / "hunter_prior_search_history_v1.json"
    assert export.is_file()
    # The published export must validate as decision-grade in its own repo.
    state, _detail, payload = cross_project_history_validity(export)
    assert state == "valid"
    assert payload is not None
    assert payload["schema_version"] == 1
    entries = payload["sources"][0]["entries"]
    assert [entry["target_id"] for entry in entries] == ["TIC 3", "TIC 5"]
    # scanned_clear is a real completed search and must remain evidence.
    assert dict(cross_project_alias_counts(payload)) == {"TIC3": 1, "TIC5": 1}


def test_publish_translates_scanner_vocabulary(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _scan_log(
        repo / "logs" / "discovery_run_001.json",
        {
            str(index): {
                "tic_id": index,
                "scanned_at": "2026-01-01T00:00:00+00:00",
                "status": raw,
                "n_signals": 0,
            }
            for index, raw in enumerate(SCANNER_STATUS_TO_CROSS_PROJECT, start=1)
        },
        "2026-01-02T00:00:00+00:00",
    )
    payload = export_cross_project_history(repo_root=repo)
    entries = payload["sources"][0]["entries"]
    for entry in entries:
        raw = entry["result"]["scanner_status"]
        # Shared vocabulary out, this project's own vocabulary preserved.
        assert entry["status"] == SCANNER_STATUS_TO_CROSS_PROJECT[raw]


def test_publish_refuses_an_unmapped_scanner_status(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _scan_log(
        repo / "logs" / "discovery_run_001.json",
        {
            "1": {
                "tic_id": 1,
                "scanned_at": "2026-01-01T00:00:00+00:00",
                "status": "brand_new_status",
                "n_signals": 0,
            }
        },
        "2026-01-02T00:00:00+00:00",
    )
    with pytest.raises(ValueError, match="Unmapped scanner status"):
        export_cross_project_history(repo_root=repo)


def test_publish_refuses_when_there_is_nothing_real_to_export(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="nothing real to export"):
        export_cross_project_history(repo_root=tmp_path / "empty")


def test_committed_export_is_reproducible_from_its_own_logs() -> None:
    """The published export must be regenerable from the real scan logs.

    Guards against the export drifting into hand-maintained state: a field
    that publishing cannot reproduce is a field no sibling can trust.
    """
    committed = json.loads(own_history_export_path().read_text(encoding="utf-8"))
    rebuilt = export_cross_project_history()
    assert rebuilt == committed
