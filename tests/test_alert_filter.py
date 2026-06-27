"""Tests for Skills/alert_filter.py."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Skills.alert_filter import apply_filters, filter_candidates  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row(
    fpp: float = 0.10,
    pathway: str = "tfop_ready",
    n_signals: int = 2,
    rank_score: float = 0.80,
    snr: float = 12.0,
    **kwargs: object,
) -> dict:
    base = {
        "candidate_id": "TIC1_01",
        "scores": {"false_positive_probability": fpp},
        "pathway": pathway,
        "n_signals": n_signals,
        "rank_score": rank_score,
        "snr": snr,
    }
    base.update(kwargs)
    return base


# ---------------------------------------------------------------------------
# filter_candidates
# ---------------------------------------------------------------------------


class TestFilterCandidates:
    def test_no_criteria_returns_all(self) -> None:
        rows = [_row(), _row(candidate_id="TIC1_02")]
        assert len(filter_candidates(rows)) == 2

    def test_fpp_max_excludes_high_fpp(self) -> None:
        rows = [_row(fpp=0.05), _row(fpp=0.50)]
        result = filter_candidates(rows, fpp_max=0.20)
        assert len(result) == 1
        assert result[0]["scores"]["false_positive_probability"] == pytest.approx(0.05)

    def test_pathway_filter(self) -> None:
        rows = [
            _row(pathway="tfop_ready"),
            _row(pathway="planet_hunters_discussion"),
            _row(pathway="tfop_ready"),
        ]
        result = filter_candidates(rows, pathway="tfop_ready")
        assert len(result) == 2

    def test_min_signals_filter(self) -> None:
        rows = [_row(n_signals=1), _row(n_signals=2), _row(n_signals=3)]
        result = filter_candidates(rows, min_signals=2)
        assert len(result) == 2

    def test_min_rank_score_filter(self) -> None:
        rows = [_row(rank_score=0.5), _row(rank_score=0.8), _row(rank_score=0.9)]
        result = filter_candidates(rows, min_rank_score=0.75)
        assert len(result) == 2

    def test_min_snr_filter(self) -> None:
        rows = [_row(snr=5.0), _row(snr=10.0), _row(snr=15.0)]
        result = filter_candidates(rows, min_snr=8.0)
        assert len(result) == 2

    def test_combined_criteria_are_and_logic(self) -> None:
        rows = [
            _row(fpp=0.05, pathway="tfop_ready"),
            _row(fpp=0.05, pathway="planet_hunters_discussion"),
            _row(fpp=0.50, pathway="tfop_ready"),
        ]
        result = filter_candidates(rows, fpp_max=0.20, pathway="tfop_ready")
        assert len(result) == 1

    def test_empty_input_returns_empty(self) -> None:
        assert filter_candidates([], fpp_max=0.5) == []

    def test_fpp_from_best_fpp_key(self) -> None:
        # star_scanner uses "best_fpp" key
        rows = [{"best_fpp": 0.05, "pathway": "tfop_ready"}]
        result = filter_candidates(rows, fpp_max=0.20)
        assert len(result) == 1

    def test_pathway_from_best_pathway_key(self) -> None:
        rows = [{"best_fpp": 0.05, "best_pathway": "tfop_ready"}]
        result = filter_candidates(rows, pathway="tfop_ready")
        assert len(result) == 1


# ---------------------------------------------------------------------------
# apply_filters
# ---------------------------------------------------------------------------


class TestApplyFilters:
    def test_loads_and_filters_file(self, tmp_path: Path) -> None:
        rows = [_row(fpp=0.05), _row(fpp=0.90)]
        f = tmp_path / "results.json"
        f.write_text(json.dumps(rows))
        result = apply_filters(f, fpp_max=0.20)
        assert len(result) == 1

    def test_writes_output_when_path_given(self, tmp_path: Path) -> None:
        f = tmp_path / "input.json"
        out = tmp_path / "filtered.json"
        f.write_text(json.dumps([_row()]))
        apply_filters(f, output_path=out)
        assert out.exists()
        assert json.loads(out.read_text()) == [_row()]

    def test_filters_star_scanner_scan_log(self, tmp_path: Path) -> None:
        f = tmp_path / "discovery_run.json"
        f.write_text(
            json.dumps(
                {
                    "started_at": "2026-06-27T10:00:00Z",
                    "entries": {
                        "1001": {
                            "tic_id": 1001,
                            "status": "candidate_found",
                            "n_signals": 1,
                            "best_fpp": 0.10,
                            "best_pathway": "planet_hunters_discussion",
                            "best_period_days": 12.34,
                            "priority_score": 0.77,
                            "scanned_at": "2026-06-27T10:01:00Z",
                        },
                        "1002": {
                            "tic_id": 1002,
                            "status": "candidate_found",
                            "n_signals": 1,
                            "best_fpp": 0.40,
                            "best_pathway": "github_only_reproducibility",
                            "best_period_days": 3.21,
                        },
                        "1003": {
                            "tic_id": 1003,
                            "status": "scanned_clear",
                            "n_signals": 0,
                        },
                    },
                }
            )
        )

        result = apply_filters(f, fpp_max=0.15)

        assert len(result) == 1
        assert result[0]["tic_id"] == 1001
        assert result[0]["false_positive_probability"] == pytest.approx(0.10)
        assert result[0]["period_days"] == pytest.approx(12.34)

    def test_writes_filtered_star_scanner_scan_log(self, tmp_path: Path) -> None:
        f = tmp_path / "discovery_run.json"
        out = tmp_path / "filtered.json"
        f.write_text(
            json.dumps(
                {
                    "entries": {
                        "1001": {
                            "tic_id": 1001,
                            "status": "candidate_found",
                            "n_signals": 2,
                            "best_fpp": 0.08,
                            "best_pathway": "tfop_ready",
                            "best_period_days": 4.56,
                        }
                    }
                }
            )
        )

        apply_filters(f, output_path=out, fpp_max=0.15, pathway="tfop_ready")

        written = json.loads(out.read_text())
        assert len(written) == 1
        assert written[0]["candidate_id"] == "TIC 1001"
        assert written[0]["scores"]["false_positive_probability"] == pytest.approx(0.08)
