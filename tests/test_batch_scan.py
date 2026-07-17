"""Tests for Skills/batch_scan.py."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Skills.batch_scan import _cli, _write_run_report, batch_scan, read_tic_ids  # noqa: E402

# ---------------------------------------------------------------------------
# read_tic_ids
# ---------------------------------------------------------------------------


class TestReadTicIds:
    def test_plain_text_single_column(self, tmp_path: Path) -> None:
        f = tmp_path / "ids.txt"
        f.write_text("123\n456\n789\n")
        assert read_tic_ids(f) == [123, 456, 789]

    def test_comments_skipped(self, tmp_path: Path) -> None:
        f = tmp_path / "ids.txt"
        f.write_text("# header\n100\n# skip\n200\n")
        assert read_tic_ids(f) == [100, 200]

    def test_empty_lines_skipped(self, tmp_path: Path) -> None:
        f = tmp_path / "ids.txt"
        f.write_text("\n111\n\n222\n")
        assert read_tic_ids(f) == [111, 222]

    def test_csv_first_numeric_column(self, tmp_path: Path) -> None:
        f = tmp_path / "targets.csv"
        f.write_text("tic_id,name\n100,star_a\n200,star_b\n")
        assert read_tic_ids(f) == [100, 200]

    def test_csv_header_row_skipped(self, tmp_path: Path) -> None:
        f = tmp_path / "targets.csv"
        f.write_text("TIC_ID,Tmag\n300,11.2\n400,12.5\n")
        result = read_tic_ids(f)
        assert 300 in result
        assert 400 in result

    def test_nonpositive_ids_excluded(self, tmp_path: Path) -> None:
        f = tmp_path / "ids.txt"
        f.write_text("0\n-5\n100\n")
        assert read_tic_ids(f) == [100]


# ---------------------------------------------------------------------------
# batch_scan
# ---------------------------------------------------------------------------


def _make_pipeline_fn(signal_counts: dict[int, int]) -> Any:
    """Return a mock pipeline function that returns N dummy signals per TIC ID."""

    def _fn(target_id: str, mission: str, **kwargs: Any) -> list[dict[str, Any]]:
        tic_id = int(target_id.replace("TIC ", ""))
        n = signal_counts.get(tic_id, 0)
        return [{"candidate_id": f"{target_id}-{i:03d}", "snr": 10.0} for i in range(n)]

    return _fn


class TestBatchScan:
    def test_writes_output_file(self, tmp_path: Path) -> None:
        out = tmp_path / "results.json"
        batch_scan(
            [100, 200],
            output_path=out,
            run_pipeline_fn=_make_pipeline_fn({}),
        )
        assert out.exists()

    def test_output_is_list_of_entries(self, tmp_path: Path) -> None:
        out = tmp_path / "results.json"
        batch_scan([100, 200], output_path=out, run_pipeline_fn=_make_pipeline_fn({}))
        data = json.loads(out.read_text())
        assert isinstance(data, list)
        assert len(data) == 2

    def test_candidate_found_status(self, tmp_path: Path) -> None:
        out = tmp_path / "results.json"
        batch_scan([500], output_path=out, run_pipeline_fn=_make_pipeline_fn({500: 2}))
        data = json.loads(out.read_text())
        assert data[0]["status"] == "candidate_found"
        assert len(data[0]["signals"]) == 2

    def test_scanned_clear_status(self, tmp_path: Path) -> None:
        out = tmp_path / "results.json"
        batch_scan([500], output_path=out, run_pipeline_fn=_make_pipeline_fn({}))
        data = json.loads(out.read_text())
        assert data[0]["status"] == "scanned_clear"

    def test_error_status_on_exception(self, tmp_path: Path) -> None:
        def _boom(target_id: str, mission: str, **kwargs: Any) -> list[dict[str, Any]]:
            raise RuntimeError("network failure")

        out = tmp_path / "results.json"
        batch_scan([999], output_path=out, run_pipeline_fn=_boom)
        data = json.loads(out.read_text())
        assert data[0]["status"] == "error"
        assert "error" in data[0]

    def test_resume_skips_completed(self, tmp_path: Path) -> None:
        out = tmp_path / "results.json"
        # Pre-populate with TIC 100 already done
        out.write_text(json.dumps([{"tic_id": 100, "status": "scanned_clear", "signals": []}]))

        call_log: list[int] = []

        def _fn(target_id: str, mission: str, **kwargs: Any) -> list[dict[str, Any]]:
            call_log.append(int(target_id.replace("TIC ", "")))
            return []

        batch_scan([100, 200], output_path=out, resume=True, run_pipeline_fn=_fn)
        # Only TIC 200 should have been scanned
        assert call_log == [200]

    def test_resume_false_rescans_completed(self, tmp_path: Path) -> None:
        out = tmp_path / "results.json"
        out.write_text(json.dumps([{"tic_id": 100, "status": "scanned_clear", "signals": []}]))

        call_log: list[int] = []

        def _fn(target_id: str, mission: str, **kwargs: Any) -> list[dict[str, Any]]:
            call_log.append(int(target_id.replace("TIC ", "")))
            return []

        batch_scan([100, 200], output_path=out, resume=False, run_pipeline_fn=_fn)
        assert 100 in call_log

    def test_returns_all_entries(self, tmp_path: Path) -> None:
        out = tmp_path / "results.json"
        result = batch_scan(
            [10, 20, 30], output_path=out, run_pipeline_fn=_make_pipeline_fn({})
        )
        assert len(result) == 3


# ---------------------------------------------------------------------------
# Run Report (AGENTS.md Rule 7 retrofit)
# ---------------------------------------------------------------------------


class TestRunReport:
    def test_success_status_with_no_errors(self) -> None:
        results = [
            {"tic_id": 1, "status": "candidate_found"},
            {"tic_id": 2, "status": "scanned_clear"},
        ]
        with patch(
            "Skills.batch_scan.run_and_commit_report", return_value=True
        ) as commit:
            _write_run_report(
                started_at="2026-07-16T00:00:00+00:00",
                elapsed_seconds=5.0,
                results=results,
                output_path=Path("results.json"),
                git_run_fn=MagicMock(),
            )
        report, path = commit.call_args.args
        assert report.script == "batch_scan"
        assert report.status == "success"
        assert report.items_processed == 2
        assert report.items_written == 2
        assert report.items_failed == 0
        assert path.name == "batch_scan.jsonl"

    def test_partial_status_when_errors_present(self) -> None:
        results = [
            {"tic_id": 1, "status": "candidate_found"},
            {"tic_id": 2, "status": "error"},
        ]
        with patch(
            "Skills.batch_scan.run_and_commit_report", return_value=True
        ) as commit:
            _write_run_report(
                started_at="2026-07-16T00:00:00+00:00",
                elapsed_seconds=5.0,
                results=results,
                output_path=Path("results.json"),
                git_run_fn=MagicMock(),
            )
        report, _path = commit.call_args.args
        assert report.status == "partial"
        assert report.items_failed == 1
        assert report.items_written == 1

    def test_output_path_recorded(self) -> None:
        with patch(
            "Skills.batch_scan.run_and_commit_report", return_value=True
        ) as commit:
            _write_run_report(
                started_at="2026-07-16T00:00:00+00:00",
                elapsed_seconds=1.0,
                results=[],
                output_path=Path("results/candidates.json"),
                git_run_fn=MagicMock(),
            )
        report, _path = commit.call_args.args
        assert report.output_paths == ("results/candidates.json",)

    def test_git_run_fn_is_threaded_through(self) -> None:
        fake_runner = MagicMock()
        with patch(
            "Skills.batch_scan.run_and_commit_report", return_value=True
        ) as commit:
            _write_run_report(
                started_at="2026-07-16T00:00:00+00:00",
                elapsed_seconds=1.0,
                results=[],
                output_path=Path("out.json"),
                git_run_fn=fake_runner,
            )
        assert commit.call_args.kwargs["run_fn"] is fake_runner

    def test_commit_failure_warns_but_does_not_raise(self, capsys: Any) -> None:
        with patch("Skills.batch_scan.run_and_commit_report", return_value=False):
            _write_run_report(
                started_at="2026-07-16T00:00:00+00:00",
                elapsed_seconds=1.0,
                results=[],
                output_path=Path("out.json"),
                git_run_fn=MagicMock(),
            )
        assert "Warning" in capsys.readouterr().err

    def test_cli_writes_run_report_with_injected_git_runner(
        self, tmp_path: Path
    ) -> None:
        targets = tmp_path / "targets.txt"
        targets.write_text("100\n200\n")
        out = tmp_path / "results.json"
        fake_runner = MagicMock()

        with (
            patch(
                "Skills.batch_scan.batch_scan",
                return_value=[{"tic_id": 100, "status": "scanned_clear"}],
            ),
            patch(
                "Skills.batch_scan.run_and_commit_report", return_value=True
            ) as commit,
        ):
            exit_code = _cli(
                [str(targets), "--output", str(out)], git_run_fn=fake_runner
            )

        assert exit_code == 0
        commit.assert_called_once()
        assert commit.call_args.kwargs["run_fn"] is fake_runner
