"""Contract tests for the deterministic installed-CLI Hunter acceptance."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

from Skills.run_hunter_production_acceptance import run_acceptance

from exo_toolkit.hunter_validity import validate_hunter_database
from exo_toolkit.search_lifecycle import HUNTER_SCHEMA_VERSION


def test_installed_cli_clean_state_acceptance(tmp_path: Path) -> None:
    work_dir = tmp_path / "work"
    evidence_path = tmp_path / "evidence.json"
    snapshot_path = tmp_path / "snapshot.sqlite3.gz"

    evidence = run_acceptance(
        work_dir=work_dir,
        evidence_out=evidence_path,
        snapshot_out=snapshot_path,
    )

    assert evidence_path.is_file()
    assert snapshot_path.is_file()
    assert all(evidence["assertion_results"].values())
    assert evidence["commands"][1]["returncode"] == 2
    assert evidence["commands"][2]["returncode"] == 0
    assert evidence["request"]["new"]["selected"][0] == "TIC 999999"


def test_committed_v16_evidence_snapshot_is_independently_verifiable(
    tmp_path: Path,
) -> None:
    evidence_path = Path(
        "artifacts/manifests/hunter_live_acceptance_v16.json"
    )
    snapshot_path = Path(
        "artifacts/evidence/hunter_production_snapshot_v16.sqlite3.gz"
    )
    if not evidence_path.is_file() or not snapshot_path.is_file():
        return

    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert all(evidence["assertion_results"].values())
    assert hashlib.sha256(snapshot_path.read_bytes()).hexdigest() == (
        evidence["snapshot"]["compressed_sha256"]
    )
    database_path = tmp_path / "snapshot.sqlite3"
    database_bytes = gzip.decompress(snapshot_path.read_bytes())
    database_path.write_bytes(database_bytes)
    assert hashlib.sha256(database_bytes).hexdigest() == (
        evidence["snapshot"]["database_sha256"]
    )
    validation = validate_hunter_database(
        database_path,
        schema_version=HUNTER_SCHEMA_VERSION,
        history_manifest=Path(evidence["validity_report"]["history_manifest"]),
        history_source_root=Path(
            evidence["validity_report"]["history_source_root"]
        ),
    )
    assert validation["ok"] is True
