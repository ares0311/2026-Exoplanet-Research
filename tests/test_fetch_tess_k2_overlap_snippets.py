"""Tests for fetch_tess_k2_overlap_snippets.py.

No dedicated test file existed for this script before this suite (verified:
only an incidental reference from test_build_t1_2_k2_calibration_manifest.py).
Coverage mirrors the sibling test_fetch_tess_kepler_overlap_snippets.py,
adapted for K2Row/epic_id naming, plus the Run Report Policy retrofit tests.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import Skills.fetch_tess_k2_overlap_snippets as k2_overlap
from Skills.fetch_tess_k2_overlap_snippets import (
    K2Row,
    _normalise,
    _phase_fold_bin,
    _write_run_report,
    build_k2_tess_snippet,
    build_k2_tess_snippets,
    fetch_k2_table,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_CONFIRMED_ROW = K2Row(
    epic_id=211311380,
    disposition="CONFIRMED",
    period_days=2.4706,
    epoch_bjd=2457100.7285,
)

_FP_ROW = K2Row(
    epic_id=211399359,
    disposition="FALSE POSITIVE",
    period_days=2.2047,
    epoch_bjd=2457065.9417,
)


def _make_lc_fetcher(n_points: int = 500):
    """Injectable fetcher returning a flat sinusoid in full BJD."""

    def fetcher(epic_id: int, period: float, epoch_bjd: float):
        time_bjd = [epoch_bjd + i * period / n_points for i in range(n_points)]
        flux = []
        for t in time_bjd:
            ph = ((t - epoch_bjd) % period) / period
            ph = ph - 1.0 if ph >= 0.5 else ph
            flux.append(0.99 if abs(ph) < 0.02 else 1.0)
        return time_bjd, flux

    return fetcher


def _no_data_fetcher(epic_id: int, period: float, epoch_bjd: float):
    return None


def _error_fetcher(epic_id: int, period: float, epoch_bjd: float):
    raise RuntimeError("connection refused")


# ---------------------------------------------------------------------------
# fetch_k2_table
# ---------------------------------------------------------------------------


class TestFetchK2Table:
    """fetch_k2_table(url=...) uses fixed column names {epic_id, disposition,
    period, t0} for the test-override branch (see the function's own `col`
    dict when `url is not None`), and always requests csv format."""

    def test_parses_valid_rows(self, tmp_path) -> None:
        csv_body = (
            "epic_id,disposition,period,t0\n"
            "211311380,CONFIRMED,2.4706,2457100.7285\n"
            "211399359,FALSE POSITIVE,2.2047,2457065.9417\n"
        )
        f = tmp_path / "k2.csv"
        f.write_text(csv_body, encoding="utf-8")
        rows = fetch_k2_table(f"file://{f}")
        assert len(rows) == 2
        assert {r.disposition for r in rows} == {"CONFIRMED", "FALSE POSITIVE"}

    def test_empty_body_returns_empty_not_raise(self, tmp_path) -> None:
        f = tmp_path / "k2.csv"
        f.write_text("epic_id,disposition,period,t0\n", encoding="utf-8")
        rows = fetch_k2_table(f"file://{f}")
        assert rows == []

    def test_renamed_column_raises_instead_of_silently_returning_empty(
        self, tmp_path
    ) -> None:
        # Regression: every row missing the expected "epic_id" key (e.g. the
        # archive renamed the column) previously hit the same
        # `except (KeyError, TypeError, ValueError): continue` as an
        # ordinary malformed value, so a fully broken response silently
        # returned [] -- indistinguishable from "zero K2 rows match."
        csv_body = (
            "epic_identifier,disposition,period,t0\n"
            "211311380,CONFIRMED,2.4706,2457100.7285\n"
            "211399359,FALSE POSITIVE,2.2047,2457065.9417\n"
        )
        f = tmp_path / "k2.csv"
        f.write_text(csv_body, encoding="utf-8")
        with pytest.raises(RuntimeError, match="missing expected column"):
            fetch_k2_table(f"file://{f}")

    def test_some_rows_malformed_others_valid_skips_only_malformed(
        self, tmp_path
    ) -> None:
        csv_body = (
            "epic_id,disposition,period,t0\n"
            "211311380,CONFIRMED,2.4706,2457100.7285\n"
            "not-a-number,CONFIRMED,2.2047,2457065.9417\n"
        )
        f = tmp_path / "k2.csv"
        f.write_text(csv_body, encoding="utf-8")
        rows = fetch_k2_table(f"file://{f}")
        assert len(rows) == 1
        assert rows[0].epic_id == 211311380


# ---------------------------------------------------------------------------
# Phase-fold / normalise helpers (shared implementation with the sibling
# Kepler-overlap module; light smoke coverage only)
# ---------------------------------------------------------------------------


class TestPhaseFoldBin:
    def test_output_length_equals_n_bins(self) -> None:
        time_bjd = [2457000.0 + float(i) for i in range(100)]
        flux = [1.0] * 100
        for n in (11, 51, 201):
            bins = _phase_fold_bin(time_bjd, flux, period=5.0, epoch=2457000.0, n_bins=n)
            assert len(bins) == n


class TestNormalise:
    def test_zero_mad_returns_zeros(self) -> None:
        assert all(v == 0.0 for v in _normalise([1.0] * 10))


# ---------------------------------------------------------------------------
# build_k2_tess_snippet
# ---------------------------------------------------------------------------


class TestBuildK2TessSnippet:
    def test_ok_flag_on_valid_data(self) -> None:
        result = build_k2_tess_snippet(
            _CONFIRMED_ROW, n_bins=201, lc_fetcher=_make_lc_fetcher()
        )
        assert result.flag == "OK"
        assert len(result.flux) == 201
        assert result.epic_id == _CONFIRMED_ROW.epic_id
        assert result.label == 1

    def test_false_positive_gives_label_zero(self) -> None:
        result = build_k2_tess_snippet(_FP_ROW, n_bins=201, lc_fetcher=_make_lc_fetcher())
        assert result.label == 0

    def test_no_data_flag_when_none_returned(self) -> None:
        result = build_k2_tess_snippet(
            _CONFIRMED_ROW, n_bins=201, lc_fetcher=_no_data_fetcher
        )
        assert result.flag in {"NO_LIGHTKURVE", "NO_DATA"}

    def test_short_flag_when_too_few_points(self) -> None:
        def fetcher(epic_id, period, epoch):
            return [2457000.0, 2457001.0], [1.0, 1.0]

        result = build_k2_tess_snippet(_CONFIRMED_ROW, n_bins=201, lc_fetcher=fetcher)
        assert result.flag == "SHORT"

    def test_error_flag_on_exception(self) -> None:
        result = build_k2_tess_snippet(
            _CONFIRMED_ROW, n_bins=201, lc_fetcher=_error_fetcher
        )
        assert result.flag.startswith("ERROR")


# ---------------------------------------------------------------------------
# build_k2_tess_snippets — batch runner
# ---------------------------------------------------------------------------


def test_batch_writes_jsonl(tmp_path: Path) -> None:
    out = tmp_path / "k2_overlap.jsonl"
    rows = [_CONFIRMED_ROW, _FP_ROW]
    n = build_k2_tess_snippets(
        rows, n_bins=201, output_path=out, lc_fetcher=_make_lc_fetcher()
    )
    assert n == 2
    lines = [json.loads(ln) for ln in out.read_text().splitlines() if ln.strip()]
    assert len(lines) == 2


def test_batch_jsonl_fields(tmp_path: Path) -> None:
    out = tmp_path / "k2_overlap.jsonl"
    build_k2_tess_snippets(
        [_CONFIRMED_ROW], n_bins=201, output_path=out, lc_fetcher=_make_lc_fetcher()
    )
    obj = json.loads(out.read_text().strip())
    keys = ("tic_id", "label", "flux", "source", "period_days", "epoch_bjd", "n_bins", "epic_id")
    for key in keys:
        assert key in obj
    assert obj["source"] == "k2_tess_overlap"


def test_batch_resume_skips_done(tmp_path: Path) -> None:
    out = tmp_path / "k2_overlap.jsonl"
    rows = [_CONFIRMED_ROW, _FP_ROW]
    n1 = build_k2_tess_snippets(
        rows, n_bins=201, output_path=out, lc_fetcher=_make_lc_fetcher()
    )
    assert n1 == 2
    n2 = build_k2_tess_snippets(
        rows, n_bins=201, output_path=out, lc_fetcher=_make_lc_fetcher()
    )
    assert n2 == 0
    lines = [ln for ln in out.read_text().splitlines() if ln.strip()]
    assert len(lines) == 2


def test_batch_max_errors_stops_early(tmp_path: Path) -> None:
    out = tmp_path / "k2_overlap.jsonl"
    rows = [_CONFIRMED_ROW] * 5
    n = build_k2_tess_snippets(
        rows, n_bins=201, output_path=out, lc_fetcher=_error_fetcher, max_errors=3,
    )
    assert n == 0


def test_batch_records_terminal_failures_in_sidecar(tmp_path: Path) -> None:
    out = tmp_path / "k2_overlap.jsonl"
    n = build_k2_tess_snippets(
        [_CONFIRMED_ROW], n_bins=201, output_path=out, lc_fetcher=_no_data_fetcher
    )
    assert n == 0
    failure_log = out.with_name(out.name + ".failures.jsonl")
    assert failure_log.exists()
    failures = [json.loads(ln) for ln in failure_log.read_text().splitlines()]
    assert len(failures) == 1
    assert failures[0]["epic_id"] == _CONFIRMED_ROW.epic_id
    assert failures[0]["flag"] in {"NO_DATA", "NO_LIGHTKURVE"}


def test_batch_stats_populated_on_success(tmp_path: Path) -> None:
    out = tmp_path / "k2_overlap.jsonl"
    rows = [_CONFIRMED_ROW, _FP_ROW]
    stats: dict[str, int] = {}
    n = build_k2_tess_snippets(
        rows, n_bins=201, output_path=out, lc_fetcher=_make_lc_fetcher(), stats=stats
    )
    assert stats["written"] == n == 2
    assert stats["errors"] == 0
    assert stats["total"] == 2


def test_batch_stats_populated_with_errors(tmp_path: Path) -> None:
    out = tmp_path / "k2_overlap.jsonl"
    stats: dict[str, int] = {}
    n = build_k2_tess_snippets(
        [_CONFIRMED_ROW],
        n_bins=201,
        output_path=out,
        lc_fetcher=_no_data_fetcher,
        stats=stats,
    )
    assert n == 0
    assert stats["written"] == 0
    assert stats["errors"] == 1
    assert stats["terminal_failures"] == 1
    assert stats["total"] == 1


def test_batch_stats_unset_when_not_requested(tmp_path: Path) -> None:
    out = tmp_path / "k2_overlap.jsonl"
    n = build_k2_tess_snippets(
        [_CONFIRMED_ROW], n_bins=201, output_path=out, lc_fetcher=_make_lc_fetcher()
    )
    assert n == 1


# ---------------------------------------------------------------------------
# Run Report (AGENTS.md Rule 7 retrofit)
# ---------------------------------------------------------------------------


class TestRunReport:
    def test_success_status_with_no_errors(self) -> None:
        with patch(
            "Skills.fetch_tess_k2_overlap_snippets.run_and_commit_report",
            return_value=True,
        ) as commit:
            _write_run_report(
                started_at="2026-07-17T00:00:00+00:00",
                elapsed_seconds=5.0,
                stats={"written": 10, "errors": 0, "terminal_failures": 0, "total": 10},
                output_path=Path("data/tess_k2_overlap_snippets.jsonl"),
                git_run_fn=MagicMock(),
            )
        report, path = commit.call_args.args
        assert report.script == "fetch_tess_k2_overlap_snippets"
        assert report.status == "success"
        assert report.items_processed == 10
        assert report.items_written == 10
        assert report.items_failed == 0
        assert path.name == "fetch_tess_k2_overlap_snippets.jsonl"

    def test_partial_status_when_errors_present(self) -> None:
        with patch(
            "Skills.fetch_tess_k2_overlap_snippets.run_and_commit_report",
            return_value=True,
        ) as commit:
            _write_run_report(
                started_at="2026-07-17T00:00:00+00:00",
                elapsed_seconds=5.0,
                stats={"written": 8, "errors": 2, "terminal_failures": 1, "total": 10},
                output_path=Path("out.jsonl"),
                git_run_fn=MagicMock(),
            )
        report, _path = commit.call_args.args
        assert report.status == "partial"
        assert report.items_failed == 2
        assert report.items_written == 8
        assert "terminal_failures=1" in report.notes

    def test_git_run_fn_is_threaded_through(self) -> None:
        fake_runner = MagicMock()
        with patch(
            "Skills.fetch_tess_k2_overlap_snippets.run_and_commit_report",
            return_value=True,
        ) as commit:
            _write_run_report(
                started_at="2026-07-17T00:00:00+00:00",
                elapsed_seconds=1.0,
                stats={},
                output_path=Path("out.jsonl"),
                git_run_fn=fake_runner,
            )
        assert commit.call_args.kwargs["run_fn"] is fake_runner

    def test_commit_failure_warns_but_does_not_raise(self, capsys: Any) -> None:
        with patch(
            "Skills.fetch_tess_k2_overlap_snippets.run_and_commit_report",
            return_value=False,
        ):
            _write_run_report(
                started_at="2026-07-17T00:00:00+00:00",
                elapsed_seconds=1.0,
                stats={},
                output_path=Path("out.jsonl"),
                git_run_fn=MagicMock(),
            )
        assert "Warning" in capsys.readouterr().out

    def test_cli_writes_run_report_with_injected_git_runner(self, tmp_path: Path) -> None:
        out = tmp_path / "k2_overlap.jsonl"
        fake_runner = MagicMock()

        def fake_build(*args: Any, **kwargs: Any) -> int:
            stats = kwargs.get("stats")
            if stats is not None:
                stats["written"] = 1
                stats["errors"] = 0
                stats["terminal_failures"] = 0
                stats["total"] = 1
            return 1

        with (
            patch(
                "Skills.fetch_tess_k2_overlap_snippets.fetch_k2_table",
                return_value=[_CONFIRMED_ROW],
            ),
            patch(
                "Skills.fetch_tess_k2_overlap_snippets.build_k2_tess_snippets",
                side_effect=fake_build,
            ),
            patch(
                "Skills.fetch_tess_k2_overlap_snippets.run_and_commit_report",
                return_value=True,
            ) as commit,
        ):
            exit_code = k2_overlap._cli(["--output", str(out)], git_run_fn=fake_runner)

        assert exit_code == 0
        commit.assert_called_once()
        assert commit.call_args.kwargs["run_fn"] is fake_runner
        report, _path = commit.call_args.args
        assert report.items_written == 1
