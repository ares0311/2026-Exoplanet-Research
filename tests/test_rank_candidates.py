"""Tests for Skills/rank_candidates.py."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Skills.rank_candidates import (  # noqa: E402
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
