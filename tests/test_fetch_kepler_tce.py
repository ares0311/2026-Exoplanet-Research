"""Tests for Skills/fetch_kepler_tce.py.

No dedicated test file existed for this script before this suite (verified:
no reference anywhere under tests/). Covers the injectable query_fn/stats
side channel added for testability plus the Run Report Policy retrofit.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
from Skills.fetch_kepler_tce import _cli, _write_run_report, fetch_koi_table


def _fake_query_fn(rows: list[dict[str, Any]]):
    # Explicit columns so an empty result still has the expected schema,
    # matching what a real empty archive query response looks like.
    frame = pd.DataFrame(rows, columns=["kepoi_name", "koi_disposition", "koi_period"])

    class _FakeTable:
        def to_pandas(self) -> pd.DataFrame:
            return frame

    def query_fn(**kwargs: Any) -> _FakeTable:
        return _FakeTable()

    return query_fn


_ROWS = [
    {"kepoi_name": "K00001.01", "koi_disposition": "CONFIRMED", "koi_period": 2.47},
    {"kepoi_name": "K00002.01", "koi_disposition": "FALSE POSITIVE", "koi_period": 3.1},
    {"kepoi_name": "K00003.01", "koi_disposition": "CONFIRMED", "koi_period": 9.9},
]


class TestFetchKoiTable:
    def test_writes_csv(self, tmp_path: Path) -> None:
        out = tmp_path / "koi.csv"
        result = fetch_koi_table(out, query_fn=_fake_query_fn(_ROWS))
        assert result == out
        assert out.exists()
        df = pd.read_csv(out)
        assert len(df) == 3

    def test_stats_populated(self, tmp_path: Path) -> None:
        out = tmp_path / "koi.csv"
        stats: dict[str, int] = {}
        fetch_koi_table(out, query_fn=_fake_query_fn(_ROWS), stats=stats)
        assert stats["written"] == 3
        assert stats["errors"] == 0
        assert stats["total"] == 3
        assert stats["confirmed"] == 2
        assert stats["false_positive"] == 1

    def test_stats_unset_when_not_requested(self, tmp_path: Path) -> None:
        out = tmp_path / "koi.csv"
        result = fetch_koi_table(out, query_fn=_fake_query_fn(_ROWS))
        assert result.exists()

    def test_empty_table(self, tmp_path: Path) -> None:
        out = tmp_path / "koi.csv"
        stats: dict[str, int] = {}
        fetch_koi_table(out, query_fn=_fake_query_fn([]), stats=stats)
        assert stats["written"] == 0
        assert stats["confirmed"] == 0
        assert stats["false_positive"] == 0

    def test_query_failure_propagates_uncaught(self, tmp_path: Path) -> None:
        def broken_query_fn(**kwargs: Any) -> Any:
            raise RuntimeError("archive unreachable")

        out = tmp_path / "koi.csv"
        try:
            fetch_koi_table(out, query_fn=broken_query_fn)
        except RuntimeError as exc:
            assert "archive unreachable" in str(exc)
        else:
            raise AssertionError("expected RuntimeError to propagate")
        assert not out.exists()


# ---------------------------------------------------------------------------
# Run Report (AGENTS.md Rule 7 retrofit)
# ---------------------------------------------------------------------------


class TestRunReport:
    def test_success_status_with_notes(self) -> None:
        with patch(
            "Skills.fetch_kepler_tce.run_and_commit_report", return_value=True
        ) as commit:
            _write_run_report(
                started_at="2026-07-17T00:00:00+00:00",
                elapsed_seconds=5.0,
                stats={
                    "written": 10, "errors": 0, "total": 10,
                    "confirmed": 6, "false_positive": 4,
                },
                output_path=Path("data/kepler_koi.csv"),
                git_run_fn=MagicMock(),
            )
        report, path = commit.call_args.args
        assert report.script == "fetch_kepler_tce"
        assert report.status == "success"
        assert report.items_processed == 10
        assert report.items_written == 10
        assert report.items_failed == 0
        assert "confirmed=6" in report.notes
        assert "false_positive=4" in report.notes
        assert path.name == "fetch_kepler_tce.jsonl"

    def test_git_run_fn_is_threaded_through(self) -> None:
        fake_runner = MagicMock()
        with patch(
            "Skills.fetch_kepler_tce.run_and_commit_report", return_value=True
        ) as commit:
            _write_run_report(
                started_at="2026-07-17T00:00:00+00:00",
                elapsed_seconds=1.0,
                stats={},
                output_path=Path("out.csv"),
                git_run_fn=fake_runner,
            )
        assert commit.call_args.kwargs["run_fn"] is fake_runner

    def test_commit_failure_warns_but_does_not_raise(self, capsys: Any) -> None:
        with patch(
            "Skills.fetch_kepler_tce.run_and_commit_report", return_value=False
        ):
            _write_run_report(
                started_at="2026-07-17T00:00:00+00:00",
                elapsed_seconds=1.0,
                stats={},
                output_path=Path("out.csv"),
                git_run_fn=MagicMock(),
            )
        assert "Warning" in capsys.readouterr().out

    def test_cli_writes_run_report_with_injected_git_runner(self, tmp_path: Path) -> None:
        out = tmp_path / "koi.csv"
        fake_runner = MagicMock()

        def fake_fetch(output_path: Any, *, stats: dict[str, int] | None = None) -> Path:
            if stats is not None:
                stats["written"] = 3
                stats["errors"] = 0
                stats["total"] = 3
                stats["confirmed"] = 2
                stats["false_positive"] = 1
            return Path(output_path)

        with (
            patch("Skills.fetch_kepler_tce.fetch_koi_table", side_effect=fake_fetch),
            patch(
                "Skills.fetch_kepler_tce.run_and_commit_report", return_value=True
            ) as commit,
        ):
            exit_code = _cli(["--output", str(out)], git_run_fn=fake_runner)

        assert exit_code == 0
        commit.assert_called_once()
        assert commit.call_args.kwargs["run_fn"] is fake_runner
        report, _path = commit.call_args.args
        assert report.items_written == 3
