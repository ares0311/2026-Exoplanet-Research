"""Tests for Skills.multi_target_report."""
from __future__ import annotations

import json
from pathlib import Path

from Skills.multi_target_report import build_multi_target_report, write_multi_target_report


def _row(
    cid: str = "TIC1-001",
    target: str = "TIC 1",
    period: float = 10.0,
    fpp: float = 0.10,
    pathway: str = "tfop_ready",
    rank: float = 0.85,
) -> dict:
    return {
        "candidate_id": cid,
        "target_id": target,
        "period_days": period,
        "best_fpp": fpp,
        "best_pathway": pathway,
        "rank_score": rank,
    }


class TestBuildMultiTargetReport:
    def test_empty_rows_returns_placeholder(self) -> None:
        result = build_multi_target_report([])
        assert "No candidates" in result

    def test_report_contains_title(self) -> None:
        result = build_multi_target_report([_row()], title="My Report")
        assert "My Report" in result

    def test_report_contains_candidate_id(self) -> None:
        result = build_multi_target_report([_row(cid="TIC99-001")])
        assert "TIC99-001" in result

    def test_report_contains_summary_table(self) -> None:
        result = build_multi_target_report([_row()])
        assert "Summary" in result

    def test_candidates_sorted_by_fpp_ascending(self) -> None:
        rows = [_row(cid="B", fpp=0.90), _row(cid="A", fpp=0.05)]
        result = build_multi_target_report(rows)
        assert result.index("A") < result.index("B")

    def test_total_candidate_count_in_overview(self) -> None:
        rows = [_row(cid="X1"), _row(cid="X2"), _row(cid="X3")]
        result = build_multi_target_report(rows)
        assert "3" in result

    def test_pathway_breakdown_in_overview(self) -> None:
        rows = [_row(pathway="tfop_ready"), _row(cid="X2", pathway="github_only_reproducibility")]
        result = build_multi_target_report(rows)
        assert "tfop_ready" in result
        assert "github_only_reproducibility" in result

    def test_missing_period_and_fpp_does_not_crash(self) -> None:
        row = {"candidate_id": "X", "target_id": "TIC 0",
               "best_pathway": "github_only_reproducibility"}
        result = build_multi_target_report([row])
        assert "X" in result

    def test_default_title_used_when_not_provided(self) -> None:
        result = build_multi_target_report([_row()])
        assert "Multi-Target" in result

    def test_timeline_section_included_when_path_given(self, tmp_path: Path) -> None:
        timeline_path = tmp_path / "timeline.json"
        timeline_data = {
            "entries": {
                "TIC1-001": [
                    {"run_at": "2026-05-01T00:00:00", "fpp": 0.10,
                     "pathway": "tfop_ready", "scorer": "bayesian"}
                ]
            }
        }
        timeline_path.write_text(json.dumps(timeline_data))
        result = build_multi_target_report([_row()], timeline_path=timeline_path)
        assert "Score History" in result

    def test_no_timeline_section_when_path_is_none(self) -> None:
        result = build_multi_target_report([_row()])
        assert "Score History" not in result

    def test_multiple_pathways_counted_separately(self) -> None:
        rows = [
            _row(cid="A", pathway="tfop_ready"),
            _row(cid="B", pathway="tfop_ready"),
            _row(cid="C", pathway="planet_hunters_discussion"),
        ]
        result = build_multi_target_report(rows)
        assert "tfop_ready" in result


class TestWriteMultiTargetReport:
    def test_writes_file(self, tmp_path: Path) -> None:
        out = tmp_path / "report.md"
        write_multi_target_report([_row()], out)
        assert out.exists()
        assert len(out.read_text()) > 0
