"""Tests for Skills/fetch_confirmed_hosts.py."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Skills.fetch_confirmed_hosts import (  # noqa: E402
    _NEA_TAP_URL,
    _QUERY,
    fetch_confirmed_host_tic_ids,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_csv(*rows: tuple[str, ...], header: str = "tic_id") -> str:
    lines = [header] + [",".join(r) for r in rows]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_frozenset(self) -> None:
        result = fetch_confirmed_host_tic_ids(fetch_fn=lambda _: _make_csv(("111",)))
        assert isinstance(result, frozenset)

    def test_elements_are_ints(self) -> None:
        result = fetch_confirmed_host_tic_ids(fetch_fn=lambda _: _make_csv(("42",)))
        assert all(isinstance(v, int) for v in result)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


class TestParsing:
    def test_single_row(self) -> None:
        result = fetch_confirmed_host_tic_ids(
            fetch_fn=lambda _: _make_csv(("12345",))
        )
        assert 12345 in result

    def test_multiple_rows(self) -> None:
        result = fetch_confirmed_host_tic_ids(
            fetch_fn=lambda _: _make_csv(("10",), ("20",), ("30",))
        )
        assert result == frozenset({10, 20, 30})

    def test_float_encoded_id(self) -> None:
        result = fetch_confirmed_host_tic_ids(
            fetch_fn=lambda _: _make_csv(("99999.0",))
        )
        assert 99999 in result

    def test_tic_prefixed_archive_id(self) -> None:
        result = fetch_confirmed_host_tic_ids(
            fetch_fn=lambda _: _make_csv(("TIC 270790652",))
        )
        assert result == frozenset({270790652})

    def test_skips_empty_values(self) -> None:
        result = fetch_confirmed_host_tic_ids(
            fetch_fn=lambda _: _make_csv(("",), ("55",))
        )
        assert result == frozenset({55})

    def test_skips_whitespace_only(self) -> None:
        result = fetch_confirmed_host_tic_ids(
            fetch_fn=lambda _: _make_csv(("  ",), ("77",))
        )
        assert result == frozenset({77})

    def test_skips_non_numeric(self) -> None:
        result = fetch_confirmed_host_tic_ids(
            fetch_fn=lambda _: _make_csv(("abc",), ("88",))
        )
        assert result == frozenset({88})

    def test_deduplicates(self) -> None:
        result = fetch_confirmed_host_tic_ids(
            fetch_fn=lambda _: _make_csv(("100",), ("100",))
        )
        assert result == frozenset({100})

    def test_empty_csv_body(self) -> None:
        result = fetch_confirmed_host_tic_ids(
            fetch_fn=lambda _: "tic_id\n"
        )
        assert result == frozenset()


# ---------------------------------------------------------------------------
# Failure modes (fail-open)
# ---------------------------------------------------------------------------


class TestFailOpen:
    def test_network_error_returns_empty(self) -> None:
        def bad_fetch(_: str) -> str:
            raise OSError("connection refused")

        result = fetch_confirmed_host_tic_ids(fetch_fn=bad_fetch)
        assert result == frozenset()

    def test_malformed_csv_returns_empty(self) -> None:
        result = fetch_confirmed_host_tic_ids(
            fetch_fn=lambda _: "\x00\x01\x02"
        )
        assert isinstance(result, frozenset)

    def test_missing_tic_id_column_returns_empty(self) -> None:
        csv = "other_col\n1\n2\n"
        result = fetch_confirmed_host_tic_ids(fetch_fn=lambda _: csv)
        assert result == frozenset()


class TestStrictMode:
    def test_network_error_raises(self) -> None:
        def bad_fetch(_: str) -> str:
            raise OSError("connection refused")

        with pytest.raises(OSError, match="connection refused"):
            fetch_confirmed_host_tic_ids(fetch_fn=bad_fetch, strict=True)

    def test_empty_result_raises(self) -> None:
        with pytest.raises(RuntimeError, match="no TIC IDs"):
            fetch_confirmed_host_tic_ids(
                fetch_fn=lambda _: "tic_id\n",
                strict=True,
            )


# ---------------------------------------------------------------------------
# URL and query
# ---------------------------------------------------------------------------


class TestUrl:
    def test_url_contains_tap_endpoint(self) -> None:
        assert "exoplanetarchive.ipac.caltech.edu" in _NEA_TAP_URL
        assert "TAP" in _NEA_TAP_URL

    def test_query_targets_ps_table(self) -> None:
        assert "FROM ps" in _QUERY

    def test_query_filters_transiting(self) -> None:
        assert "tran_flag=1" in _QUERY

    def test_query_filters_default_flag(self) -> None:
        assert "default_flag=1" in _QUERY

    def test_fetch_fn_receives_url(self) -> None:
        received: list[str] = []

        def capture_fn(url: str) -> str:
            received.append(url)
            return "tic_id\n"

        fetch_confirmed_host_tic_ids(fetch_fn=capture_fn)
        assert len(received) == 1
        assert "exoplanetarchive" in received[0]
        assert "tic_id" in received[0]


# ---------------------------------------------------------------------------
# Run Report (AGENTS.md Rule 7 retrofit)
# ---------------------------------------------------------------------------


class TestRunReport:
    def test_success_status_with_notes(self, tmp_path) -> None:
        from unittest.mock import MagicMock, patch

        from Skills.fetch_confirmed_hosts import _write_run_report

        with patch(
            "Skills.fetch_confirmed_hosts.run_and_commit_report", return_value=True
        ) as commit:
            _write_run_report(
                started_at="2026-07-17T00:00:00+00:00",
                elapsed_seconds=5.0,
                status="success",
                n_tic_ids=3,
                output_path=tmp_path / "confirmed_host_tic_ids.json",
                notes="strict=True",
                git_run_fn=MagicMock(),
            )
        report, path = commit.call_args.args
        assert report.script == "fetch_confirmed_hosts"
        assert report.status == "success"
        assert report.items_processed == 3
        assert report.items_written == 3
        assert report.items_failed == 0
        assert "strict=True" in report.notes
        assert path.name == "fetch_confirmed_hosts.jsonl"

    def test_failed_status_on_fetch_error(self) -> None:
        from unittest.mock import MagicMock, patch

        from Skills.fetch_confirmed_hosts import _write_run_report

        with patch(
            "Skills.fetch_confirmed_hosts.run_and_commit_report", return_value=True
        ) as commit:
            _write_run_report(
                started_at="2026-07-17T00:00:00+00:00",
                elapsed_seconds=1.0,
                status="failed",
                n_tic_ids=0,
                output_path=Path("out.json"),
                notes="strict=True error=connection refused",
                git_run_fn=MagicMock(),
            )
        report, _path = commit.call_args.args
        assert report.status == "failed"
        assert report.items_written == 0
        assert report.items_failed == 1
        assert "error=connection refused" in report.notes

    def test_commit_failure_warns_but_does_not_raise(self, capsys) -> None:
        from unittest.mock import patch

        from Skills.fetch_confirmed_hosts import _write_run_report

        with patch(
            "Skills.fetch_confirmed_hosts.run_and_commit_report", return_value=False
        ):
            _write_run_report(
                started_at="2026-07-17T00:00:00+00:00",
                elapsed_seconds=1.0,
                status="success",
                n_tic_ids=0,
                output_path=Path("out.json"),
                notes="strict=True",
                git_run_fn=None,
            )
        assert "Warning" in capsys.readouterr().out

    def test_cli_writes_run_report_and_output_on_success(self, tmp_path) -> None:
        from unittest.mock import MagicMock, patch

        from Skills.fetch_confirmed_hosts import _cli

        out = tmp_path / "hosts.json"
        fake_runner = MagicMock()

        with (
            patch(
                "Skills.fetch_confirmed_hosts.fetch_confirmed_host_tic_ids",
                return_value=frozenset({30, 10, 20}),
            ),
            patch(
                "Skills.fetch_confirmed_hosts.run_and_commit_report",
                return_value=True,
            ) as commit,
        ):
            exit_code = _cli(["--output", str(out)], git_run_fn=fake_runner)

        assert exit_code == 0
        assert json.loads(out.read_text()) == [10, 20, 30]
        report, _path = commit.call_args.args
        assert report.status == "success"
        assert report.items_written == 3
        assert commit.call_args.kwargs["run_fn"] is fake_runner

    def test_cli_reports_failure_on_strict_exception(self, tmp_path) -> None:
        from unittest.mock import patch

        from Skills.fetch_confirmed_hosts import _cli

        out = tmp_path / "hosts.json"

        with (
            patch(
                "Skills.fetch_confirmed_hosts.fetch_confirmed_host_tic_ids",
                side_effect=RuntimeError("no TIC IDs"),
            ),
            patch(
                "Skills.fetch_confirmed_hosts.run_and_commit_report",
                return_value=True,
            ) as commit,
        ):
            exit_code = _cli(["--output", str(out)])

        assert exit_code == 1
        assert json.loads(out.read_text()) == []
        report, _path = commit.call_args.args
        assert report.status == "failed"
        assert "no TIC IDs" in report.notes
