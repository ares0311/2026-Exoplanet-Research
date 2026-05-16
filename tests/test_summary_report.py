"""Tests for Skills/summary_report.py."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Skills.summary_report import build_report, load_results, write_report  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row(status: str = "candidate_found", **kwargs: object) -> dict:
    base: dict = {
        "tic_id": 100,
        "status": status,
        "n_signals": 1,
        "best_period_days": 5.0,
        "best_fpp": 0.05,
        "best_pathway": "tfop_ready",
    }
    base.update(kwargs)
    return base


# ---------------------------------------------------------------------------
# load_results
# ---------------------------------------------------------------------------


class TestLoadResults:
    def test_loads_list_file(self, tmp_path: Path) -> None:
        f = tmp_path / "results.json"
        f.write_text(json.dumps([_row(), _row(tic_id=200)]))
        rows = load_results([f])
        assert len(rows) == 2

    def test_loads_single_dict_file(self, tmp_path: Path) -> None:
        f = tmp_path / "result.json"
        f.write_text(json.dumps(_row()))
        rows = load_results([f])
        assert len(rows) == 1

    def test_source_file_key_added(self, tmp_path: Path) -> None:
        f = tmp_path / "results.json"
        f.write_text(json.dumps([_row()]))
        rows = load_results([f])
        assert "_source_file" in rows[0]

    def test_merges_multiple_files(self, tmp_path: Path) -> None:
        f1 = tmp_path / "a.json"
        f2 = tmp_path / "b.json"
        f1.write_text(json.dumps([_row(tic_id=1)]))
        f2.write_text(json.dumps([_row(tic_id=2)]))
        rows = load_results([f1, f2])
        assert len(rows) == 2


# ---------------------------------------------------------------------------
# build_report
# ---------------------------------------------------------------------------


class TestBuildReport:
    def test_returns_string(self) -> None:
        assert isinstance(build_report([]), str)

    def test_custom_title_appears(self) -> None:
        report = build_report([], title="My Custom Title")
        assert "My Custom Title" in report

    def test_total_count_in_report(self) -> None:
        rows = [_row(), _row(tic_id=200)]
        report = build_report(rows)
        assert "2" in report

    def test_candidate_section_present(self) -> None:
        report = build_report([_row()])
        assert "Candidates" in report

    def test_no_candidates_placeholder(self) -> None:
        report = build_report([_row(status="scanned_clear")])
        assert "No candidates found" in report

    def test_candidate_tic_id_in_report(self) -> None:
        report = build_report([_row(tic_id=150428135)])
        assert "150428135" in report

    def test_errors_section_when_errors_present(self) -> None:
        row = _row(status="error", error_message="network timeout")
        report = build_report([row])
        assert "Errors" in report
        assert "network timeout" in report

    def test_candidates_sorted_by_fpp(self) -> None:
        rows = [
            _row(tic_id=1, best_fpp=0.9),
            _row(tic_id=2, best_fpp=0.1),
        ]
        report = build_report(rows)
        idx_2 = report.index("TIC 2")
        idx_1 = report.index("TIC 1")
        assert idx_2 < idx_1


# ---------------------------------------------------------------------------
# write_report
# ---------------------------------------------------------------------------


class TestWriteReport:
    def test_creates_file(self, tmp_path: Path) -> None:
        out = tmp_path / "report.md"
        write_report([_row()], out)
        assert out.exists()

    def test_file_content_is_markdown(self, tmp_path: Path) -> None:
        out = tmp_path / "report.md"
        write_report([_row()], out)
        content = out.read_text()
        assert content.startswith("#")
