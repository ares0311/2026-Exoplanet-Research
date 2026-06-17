"""Regression tests for git-add-safe local artifact policy."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _is_ignored(path: str) -> bool:
    result = subprocess.run(
        ["git", "check-ignore", "-q", "--no-index", "--", path],
        cwd=REPO_ROOT,
        check=False,
    )
    return result.returncode == 0


def test_git_add_dot_ignores_local_artifact_classes() -> None:
    ignored_paths = [
        ".venv314/bin/python",
        ".venv-py313-backup/bin/python",
        "data/kepler_snippets.jsonl",
        "data/tess_snippets_v2.jsonl",
        "data/kepler_cnn_splits/train.json",
        "data/tess_cnn_splits/manifest.json",
        "data/processed/tess_cnn_seed42/train.json",
        "checkpoints/cnn_kepler_pretrain/best.pt",
        "checkpoints/cnn_tess_finetuned/training.log",
        "models/cnn/best.pt",
        "models/cnn_tess_finetuned/best.pt",
        "logs/background_search.sqlite3",
        "logs/background-search.cron.log",
        "reports/background/draft.md",
        "reports/background/draft.html",
        "reports/calibration.png",
        "reports/tier2_status.json",
        "uv.lock",
    ]

    for path in ignored_paths:
        assert _is_ignored(path), path


def test_git_add_dot_keeps_committed_sentinels_and_approved_models_visible() -> None:
    visible_paths = [
        "logs/README.md",
        "logs/.gitignore",
        "reports/README.md",
        "reports/background/.gitignore",
        "models/xgboost_koi.json",
        "models/xgboost_koi.xgb.json",
        "models/xgboost_toi.json",
        "models/xgboost_toi.xgb.json",
        "docs/LOCAL_ARTIFACT_LEDGER.md",
        "artifacts/manifests/local_artifacts.json",
    ]

    for path in visible_paths:
        assert not _is_ignored(path), path


def test_local_artifact_manifest_matches_human_ledger() -> None:
    ledger = (REPO_ROOT / "docs" / "LOCAL_ARTIFACT_LEDGER.md").read_text(encoding="utf-8")
    manifest = json.loads(
        (REPO_ROOT / "artifacts" / "manifests" / "local_artifacts.json").read_text(
            encoding="utf-8"
        )
    )

    assert manifest["policy"]["git_add_dot_must_be_safe"] is True
    assert manifest["policy"]["github_visible_ledger_required"] is True
    assert manifest["production_gap"] == "T1-1: Production Tier 2 CNN Checkpoint"

    artifact_paths = {artifact["path"] for artifact in manifest["artifacts"]}
    required_paths = {
        "data/tess_snippets_v2.jsonl",
        "data/kepler_snippets.jsonl",
        "data/kepler_cnn_splits/",
        "data/tess_cnn_splits/",
        "checkpoints/cnn_kepler_pretrain/",
        "checkpoints/cnn_tess_finetuned/",
        "models/cnn*/",
        "logs/*.sqlite*",
        "reports/",
    }
    assert required_paths <= artifact_paths

    for path in required_paths:
        assert path in ledger

    for artifact in manifest["artifacts"]:
        assert artifact["git_policy"]
        assert artifact["status"]
        assert artifact["next_action"]
        assert artifact["validation_gate"]
