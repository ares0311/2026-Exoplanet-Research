"""Tests for Skills/batch_scan.py."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Skills.batch_scan import (  # noqa: E402
    _bridge_manual_scan_to_hunter,
    _cli,
    _manual_scan_history_entry,
    _write_run_report,
    batch_scan,
    read_tic_ids,
)

from exo_toolkit.search_lifecycle import HunterStore  # noqa: E402

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

    def test_new_entries_populated_with_only_this_calls_scans(
        self, tmp_path: Path
    ) -> None:
        out = tmp_path / "results.json"
        out.write_text(json.dumps([{"tic_id": 100, "status": "scanned_clear", "signals": []}]))
        new_entries: list[dict[str, Any]] = []

        batch_scan(
            [100, 200],
            output_path=out,
            resume=True,
            run_pipeline_fn=_make_pipeline_fn({}),
            new_entries=new_entries,
        )

        assert [e["tic_id"] for e in new_entries] == [200]


# ---------------------------------------------------------------------------
# Hunter durable-history bridge
# ---------------------------------------------------------------------------


class TestManualScanHistoryEntry:
    def test_candidate_found_maps_through_unchanged(self) -> None:
        entry = _manual_scan_history_entry(
            {"target_id": "TIC 1", "status": "candidate_found", "signals": [{}]},
            mission="TESS",
        )
        assert entry["status"] == "candidate_found"
        assert entry["metrics"]["n_signals"] == 1
        assert "error_message" not in entry

    def test_scanned_clear_maps_to_no_signal(self) -> None:
        entry = _manual_scan_history_entry(
            {"target_id": "TIC 1", "status": "scanned_clear", "signals": []},
            mission="TESS",
        )
        assert entry["status"] == "no_signal"

    def test_error_maps_to_failed_with_error_message(self) -> None:
        entry = _manual_scan_history_entry(
            {
                "target_id": "TIC 1",
                "status": "error",
                "signals": [],
                "error": "Traceback...\nRuntimeError: boom",
            },
            mission="TESS",
        )
        assert entry["status"] == "failed"
        assert "boom" in entry["error_message"]


class TestBridgeManualScanToHunter:
    def test_empty_entries_is_a_noop(self, tmp_path: Path) -> None:
        assert (
            _bridge_manual_scan_to_hunter(
                log_path=tmp_path / "results.json",
                mission="TESS",
                entries=[],
                started_at=None,  # type: ignore[arg-type]
                completed_at=None,  # type: ignore[arg-type]
                hunter_db_path=tmp_path / "hunter.sqlite3",
            )
            is None
        )

    def test_real_scan_durably_recorded_in_hunter_store(self, tmp_path: Path) -> None:
        from datetime import UTC, datetime

        out = tmp_path / "results.json"
        results = batch_scan(
            [777], output_path=out, run_pipeline_fn=_make_pipeline_fn({})
        )
        hunter_db = tmp_path / "hunter.sqlite3"

        summary = _bridge_manual_scan_to_hunter(
            log_path=out,
            mission="TESS",
            entries=results,
            started_at=datetime(2026, 7, 24, tzinfo=UTC),
            completed_at=datetime.now(UTC),
            hunter_db_path=hunter_db,
        )

        assert summary is not None
        assert summary["sources_created"] == 1
        store = HunterStore(hunter_db)
        assert "TIC 777" in store.searched_target_ids()
        assert store.target_history("TIC 777")[0]["status"] == "no_signal"

    def test_missing_log_file_warns_but_returns_none(
        self, tmp_path: Path, capsys: Any
    ) -> None:
        from datetime import UTC, datetime

        result = _bridge_manual_scan_to_hunter(
            log_path=tmp_path / "does_not_exist.json",
            mission="TESS",
            entries=[{"target_id": "TIC 1", "status": "scanned_clear", "signals": []}],
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            hunter_db_path=tmp_path / "hunter.sqlite3",
        )
        assert result is None
        assert "Warning" in capsys.readouterr().err


class TestCliHunterBridgeWiring:
    def test_cli_bridges_new_scan_by_default(self, tmp_path: Path) -> None:
        targets = tmp_path / "targets.txt"
        targets.write_text("777\n")
        out = tmp_path / "results.json"
        hunter_db = tmp_path / "hunter.sqlite3"

        with (
            patch("exo_toolkit.cli.run_pipeline", side_effect=lambda *a, **k: []),
            patch("Skills.batch_scan.run_and_commit_report", return_value=True),
        ):
            exit_code = _cli(
                [str(targets), "--output", str(out), "--hunter-db", str(hunter_db)]
            )

        assert exit_code == 0
        assert hunter_db.exists()
        assert "TIC 777" in HunterStore(hunter_db).searched_target_ids()

    def test_no_hunter_bridge_flag_skips_durable_recording(self, tmp_path: Path) -> None:
        targets = tmp_path / "targets.txt"
        targets.write_text("777\n")
        out = tmp_path / "results.json"
        hunter_db = tmp_path / "hunter.sqlite3"

        with (
            patch("exo_toolkit.cli.run_pipeline", side_effect=lambda *a, **k: []),
            patch("Skills.batch_scan.run_and_commit_report", return_value=True),
        ):
            exit_code = _cli(
                [
                    str(targets),
                    "--output",
                    str(out),
                    "--hunter-db",
                    str(hunter_db),
                    "--no-hunter-bridge",
                ]
            )

        assert exit_code == 0
        assert not hunter_db.exists()


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
