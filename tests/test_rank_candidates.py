"""Tests for Skills/rank_candidates.py."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Skills.rank_candidates import (  # noqa: E402
    _cli,
    _scan_log_to_rows,
    compute_rank_score,
    load_candidates,
    rank_candidates,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row(
    *,
    candidate_id: str = "TIC0-001",
    target_id: str = "TIC 0",
    fpp: float = 0.20,
    dc: float = 0.80,
    novelty: float = 0.70,
    prov: float = 1.00,
    pathway: str = "planet_hunters_discussion",
    period_days: float = 10.0,
    snr: float = 15.0,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "target_id": target_id,
        "period_days": period_days,
        "snr": snr,
        "provenance_score": prov,
        "pathway": pathway,
        "scores": {
            "false_positive_probability": fpp,
            "detection_confidence": dc,
            "novelty_score": novelty,
        },
        "posterior": {},
    }


# ---------------------------------------------------------------------------
# compute_rank_score
# ---------------------------------------------------------------------------


class TestComputeRankScore:
    def test_best_possible_score(self) -> None:
        row = _row(fpp=0.0, dc=1.0, novelty=1.0, prov=1.0, pathway="tfop_ready")
        assert compute_rank_score(row) == 1.0

    def test_worst_possible_score(self) -> None:
        row = _row(fpp=1.0, dc=0.0, novelty=0.0, prov=0.0, pathway="github_only_reproducibility")
        assert compute_rank_score(row) == 0.0

    def test_tfop_pathway_adds_bonus(self) -> None:
        base_row = _row(pathway="github_only_reproducibility")
        tfop_row = _row(pathway="tfop_ready")
        assert compute_rank_score(tfop_row) > compute_rank_score(base_row)

    def test_score_clipped_to_one(self) -> None:
        row = _row(fpp=0.0, dc=1.0, novelty=1.0, prov=1.0, pathway="tfop_ready")
        assert compute_rank_score(row) <= 1.0

    def test_missing_scores_defaults_to_zero(self) -> None:
        row: dict[str, Any] = {"candidate_id": "x", "pathway": "github_only_reproducibility"}
        score = compute_rank_score(row)
        assert 0.0 <= score <= 1.0

    def test_lower_fpp_gives_higher_score(self) -> None:
        high_fpp = _row(fpp=0.8)
        low_fpp = _row(fpp=0.1)
        assert compute_rank_score(low_fpp) > compute_rank_score(high_fpp)


# ---------------------------------------------------------------------------
# load_candidates
# ---------------------------------------------------------------------------


class TestLoadCandidates:
    def test_single_dict_file(self, tmp_path: Path) -> None:
        f = tmp_path / "c.json"
        f.write_text(json.dumps(_row()))
        rows = load_candidates([f])
        assert len(rows) == 1

    def test_list_file(self, tmp_path: Path) -> None:
        f = tmp_path / "c.json"
        f.write_text(json.dumps([_row(), _row(candidate_id="TIC0-002")]))
        rows = load_candidates([f])
        assert len(rows) == 2

    def test_source_file_key_added(self, tmp_path: Path) -> None:
        f = tmp_path / "c.json"
        f.write_text(json.dumps(_row()))
        rows = load_candidates([f])
        assert "_source_file" in rows[0]
        assert str(f) == rows[0]["_source_file"]


# ---------------------------------------------------------------------------
# rank_candidates
# ---------------------------------------------------------------------------


class TestRankCandidates:
    def test_sorted_by_rank_score_descending(self) -> None:
        rows = [
            _row(candidate_id="A", fpp=0.8, dc=0.2),
            _row(candidate_id="B", fpp=0.1, dc=0.9),
            _row(candidate_id="C", fpp=0.4, dc=0.5),
        ]
        ranked = rank_candidates(rows)
        scores = [r["rank_score"] for r in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_top_n_limits_results(self) -> None:
        rows = [_row(candidate_id=str(i)) for i in range(10)]
        ranked = rank_candidates(rows, top_n=3)
        assert len(ranked) == 3

    def test_rank_score_key_added(self) -> None:
        ranked = rank_candidates([_row()])
        assert "rank_score" in ranked[0]


# ---------------------------------------------------------------------------
# ScanLog format support
# ---------------------------------------------------------------------------

def _scan_log(entries: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "last_updated": "2026-06-27T12:00:00Z",
        "entries": {
            str(e["tic_id"]): e for e in entries
        },
    }


def _scan_entry(
    tic_id: int = 11111,
    status: str = "candidate_found",
    best_fpp: float | None = 0.08,
    best_pathway: str | None = "planet_hunters_discussion",
    best_period_days: float | None = 5.0,
) -> dict[str, Any]:
    return {
        "tic_id": tic_id,
        "scanned_at": "2026-06-27T12:00:00Z",
        "status": status,
        "n_signals": 1,
        "best_period_days": best_period_days,
        "best_fpp": best_fpp,
        "best_pathway": best_pathway,
        "priority_score": 0.7,
        "error_message": None,
    }


class TestScanLogToRows:
    def test_candidate_found_entry_included(self) -> None:
        log = _scan_log([_scan_entry(tic_id=100, status="candidate_found")])
        rows = _scan_log_to_rows(log)
        assert len(rows) == 1
        assert rows[0]["tic_id"] == 100

    def test_non_candidate_entries_excluded(self) -> None:
        log = _scan_log([
            _scan_entry(tic_id=1, status="candidate_found"),
            _scan_entry(tic_id=2, status="scanned_clear"),
            _scan_entry(tic_id=3, status="no_data"),
            _scan_entry(tic_id=4, status="error"),
        ])
        rows = _scan_log_to_rows(log)
        assert len(rows) == 1
        assert rows[0]["tic_id"] == 1

    def test_fpp_mapped_to_scores(self) -> None:
        log = _scan_log([_scan_entry(best_fpp=0.07)])
        rows = _scan_log_to_rows(log)
        assert rows[0]["scores"]["false_positive_probability"] == 0.07

    def test_pathway_mapped(self) -> None:
        log = _scan_log([_scan_entry(best_pathway="tfop_ready")])
        rows = _scan_log_to_rows(log)
        assert rows[0]["pathway"] == "tfop_ready"

    def test_period_days_mapped(self) -> None:
        log = _scan_log([_scan_entry(best_period_days=12.3)])
        rows = _scan_log_to_rows(log)
        assert rows[0]["period_days"] == 12.3

    def test_target_id_and_candidate_id_set(self) -> None:
        log = _scan_log([_scan_entry(tic_id=99999)])
        rows = _scan_log_to_rows(log)
        assert "99999" in rows[0]["target_id"]
        assert "99999" in rows[0]["candidate_id"]

    def test_none_fpp_defaults_to_one(self) -> None:
        log = _scan_log([_scan_entry(best_fpp=None)])
        rows = _scan_log_to_rows(log)
        assert rows[0]["scores"]["false_positive_probability"] == 1.0

    def test_empty_entries(self) -> None:
        log: dict[str, Any] = {"last_updated": "2026-06-27T12:00:00Z", "entries": {}}
        rows = _scan_log_to_rows(log)
        assert rows == []


class TestLoadCandidatesScanLog:
    def test_scan_log_file_detected(self, tmp_path: Path) -> None:
        f = tmp_path / "scan.json"
        log = _scan_log([_scan_entry(tic_id=555)])
        f.write_text(json.dumps(log))
        rows = load_candidates([f])
        assert len(rows) == 1
        assert rows[0]["tic_id"] == 555

    def test_non_candidates_filtered(self, tmp_path: Path) -> None:
        f = tmp_path / "scan.json"
        log = _scan_log([
            _scan_entry(tic_id=1, status="candidate_found"),
            _scan_entry(tic_id=2, status="scanned_clear"),
        ])
        f.write_text(json.dumps(log))
        rows = load_candidates([f])
        assert len(rows) == 1

    def test_source_file_key_added_for_scan_log(self, tmp_path: Path) -> None:
        f = tmp_path / "scan.json"
        f.write_text(json.dumps(_scan_log([_scan_entry()])))
        rows = load_candidates([f])
        assert rows[0]["_source_file"] == str(f)

    def test_scan_log_ranks_correctly(self, tmp_path: Path) -> None:
        f = tmp_path / "scan.json"
        log = _scan_log([
            _scan_entry(tic_id=1, best_fpp=0.05, best_pathway="tfop_ready"),
            _scan_entry(tic_id=2, best_fpp=0.30, best_pathway="planet_hunters_discussion"),
        ])
        f.write_text(json.dumps(log))
        rows = load_candidates([f])
        ranked = rank_candidates(rows)
        assert ranked[0]["tic_id"] == 1


class TestRankCandidatesCli:
    def test_missing_file_returns_operator_message(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        code = _cli([str(tmp_path / "missing_scan.json")])

        captured = capsys.readouterr()
        assert code == 2
        assert "does not exist" in captured.err
        assert "let Skills/star_scanner.py finish successfully" in captured.err
        assert "Traceback" not in captured.err
