"""Tests for Skills/run_report.py."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Skills"))

from run_report import (  # noqa: E402
    RunReport,
    append_run_report,
    commit_and_push_report,
    format_run_report,
    report_path_for,
    run_and_commit_report,
)


def test_optional_process_lock_creates_requested_lock_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from run_report import _optional_process_lock

    lock_path = tmp_path / "report.lock"
    monkeypatch.setenv("EXO_RUN_REPORT_LOCK_PATH", str(lock_path))
    with _optional_process_lock():
        assert lock_path.exists()


def _report(**overrides) -> RunReport:
    defaults = {
        "script": "process_t1_kepler_batch",
        "status": "success",
        "started_at": "2026-07-03T00:00:00+00:00",
        "completed_at": "2026-07-03T01:00:00+00:00",
        "elapsed_seconds": 3600.0,
        "items_processed": 250,
        "items_written": 278,
        "items_failed": 0,
    }
    defaults.update(overrides)
    return RunReport(**defaults)


class _FakeRun:
    """Injectable stand-in for subprocess.run recording calls and returning canned results."""

    def __init__(
        self, results: dict[tuple[str, ...], subprocess.CompletedProcess[str]] | None = None
    ):
        self.results = results or {}
        self.calls: list[list[str]] = []

    def __call__(self, args, **_kwargs) -> subprocess.CompletedProcess[str]:
        self.calls.append(list(args))
        key = tuple(args)
        if key in self.results:
            return self.results[key]
        return subprocess.CompletedProcess(args, returncode=0, stdout="", stderr="")


class TestReportPathFor:
    def test_no_shard_uses_plain_filename(self, tmp_path: Path) -> None:
        path = report_path_for("my_script", report_dir=tmp_path)
        assert path == tmp_path / "my_script.jsonl"

    def test_shard_count_one_is_treated_as_unsharded(self, tmp_path: Path) -> None:
        path = report_path_for("my_script", shard_index=0, shard_count=1, report_dir=tmp_path)
        assert path == tmp_path / "my_script.jsonl"

    def test_sharded_paths_are_distinct(self, tmp_path: Path) -> None:
        paths = {
            report_path_for("my_script", shard_index=i, shard_count=4, report_dir=tmp_path)
            for i in range(4)
        }
        assert len(paths) == 4


class TestAppendRunReport:
    def test_writes_one_json_line(self, tmp_path: Path) -> None:
        path = tmp_path / "reports" / "script.jsonl"
        append_run_report(_report(), path)
        lines = path.read_text().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["script"] == "process_t1_kepler_batch"
        assert record["items_processed"] == 250

    def test_appends_without_truncating(self, tmp_path: Path) -> None:
        path = tmp_path / "script.jsonl"
        append_run_report(_report(items_processed=1), path)
        append_run_report(_report(items_processed=2), path)
        lines = path.read_text().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["items_processed"] == 1
        assert json.loads(lines[1])["items_processed"] == 2

    def test_overall_progress_fields_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "script.jsonl"
        append_run_report(
            _report(items_done_total=3877, items_total=6515, percent_done=59.5), path
        )
        record = json.loads(path.read_text().splitlines()[0])
        assert record["items_done_total"] == 3877
        assert record["items_total"] == 6515
        assert record["percent_done"] == 59.5

    def test_overall_progress_fields_default_to_none(self, tmp_path: Path) -> None:
        path = tmp_path / "script.jsonl"
        append_run_report(_report(), path)
        record = json.loads(path.read_text().splitlines()[0])
        assert record["items_done_total"] is None
        assert record["items_total"] is None
        assert record["percent_done"] is None


class TestFormatRunReport:
    def test_includes_key_fields(self) -> None:
        text = format_run_report(_report())
        assert "process_t1_kepler_batch" in text
        assert "250" in text
        assert "278" in text
        assert "success" in text

    def test_includes_shard_when_sharded(self) -> None:
        text = format_run_report(_report(shard_index=1, shard_count=4))
        assert "1/4" in text

    def test_omits_shard_when_not_sharded(self) -> None:
        text = format_run_report(_report())
        assert "Shard" not in text

    def test_includes_notes_when_present(self) -> None:
        text = format_run_report(_report(notes="two targets returned NO_DATA"))
        assert "two targets returned NO_DATA" in text

    def test_includes_overall_progress_when_present(self) -> None:
        text = format_run_report(
            _report(items_done_total=3877, items_total=6515, percent_done=59.5)
        )
        assert "3877/6515" in text
        assert "59.5%" in text

    def test_omits_overall_progress_when_absent(self) -> None:
        text = format_run_report(_report())
        assert "Overall progress" not in text


class TestCommitAndPushReport:
    def test_stages_only_the_given_path(self, tmp_path: Path) -> None:
        fake = _FakeRun()
        path = tmp_path / "reports" / "script.jsonl"
        ok = commit_and_push_report(path, message="test commit", run_fn=fake)
        assert ok is True
        add_call = fake.calls[0]
        assert add_call == ["git", "add", "--", str(path)]

    def test_noop_when_nothing_to_commit(self, tmp_path: Path) -> None:
        path = tmp_path / "script.jsonl"
        results = {
            ("git", "add", "--", str(path)): subprocess.CompletedProcess(
                [], returncode=0, stdout="", stderr=""
            ),
            ("git", "commit", "-m", "msg", "--", str(path)): subprocess.CompletedProcess(
                [], returncode=1, stdout="nothing to commit, working tree clean\n", stderr=""
            ),
        }
        fake = _FakeRun(results)
        ok = commit_and_push_report(path, message="msg", run_fn=fake)
        assert ok is True
        # Must not attempt to push when there was nothing to commit.
        assert not any(call[:2] == ["git", "push"] for call in fake.calls)

    def test_real_commit_failure_returns_false(self, tmp_path: Path) -> None:
        path = tmp_path / "script.jsonl"
        results = {
            ("git", "commit", "-m", "msg", "--", str(path)): subprocess.CompletedProcess(
                [], returncode=1, stdout="", stderr="fatal: unable to write commit\n"
            ),
        }
        fake = _FakeRun(results)
        ok = commit_and_push_report(path, message="msg", run_fn=fake)
        assert ok is False

    def test_add_failure_returns_false_without_committing(self, tmp_path: Path) -> None:
        path = tmp_path / "script.jsonl"
        results = {
            ("git", "add", "--", str(path)): subprocess.CompletedProcess(
                [], returncode=1, stdout="", stderr="fatal: pathspec did not match\n"
            ),
        }
        fake = _FakeRun(results)
        ok = commit_and_push_report(path, message="msg", run_fn=fake)
        assert ok is False
        assert not any(call[0:2] == ["git", "commit"] for call in fake.calls)

    def test_retries_on_push_rejection_then_succeeds(self, tmp_path: Path) -> None:
        path = tmp_path / "script.jsonl"
        push_attempts = {"n": 0}

        def _run_fn(args, **_kwargs) -> subprocess.CompletedProcess[str]:
            if args[:2] == ["git", "push"]:
                push_attempts["n"] += 1
                if push_attempts["n"] == 1:
                    return subprocess.CompletedProcess(
                        args, returncode=1, stdout="", stderr="rejected"
                    )
                return subprocess.CompletedProcess(args, returncode=0, stdout="", stderr="")
            if args[:3] == ["git", "rev-parse", "--abbrev-ref"]:
                return subprocess.CompletedProcess(args, returncode=0, stdout="main\n", stderr="")
            return subprocess.CompletedProcess(args, returncode=0, stdout="", stderr="")

        ok = commit_and_push_report(path, message="msg", max_retries=3, run_fn=_run_fn)
        assert ok is True
        assert push_attempts["n"] == 2

    def test_gives_up_after_max_retries_without_raising(self, tmp_path: Path) -> None:
        path = tmp_path / "script.jsonl"

        def _run_fn(args, **_kwargs) -> subprocess.CompletedProcess[str]:
            if args[:2] == ["git", "push"]:
                return subprocess.CompletedProcess(args, returncode=1, stdout="", stderr="rejected")
            if args[:3] == ["git", "rev-parse", "--abbrev-ref"]:
                return subprocess.CompletedProcess(args, returncode=0, stdout="main\n", stderr="")
            return subprocess.CompletedProcess(args, returncode=0, stdout="", stderr="")

        ok = commit_and_push_report(path, message="msg", max_retries=2, run_fn=_run_fn)
        assert ok is False

    def test_rebase_conflict_aborts_and_returns_false(self, tmp_path: Path) -> None:
        path = tmp_path / "script.jsonl"

        def _run_fn(args, **_kwargs) -> subprocess.CompletedProcess[str]:
            if args[:2] == ["git", "push"]:
                return subprocess.CompletedProcess(args, returncode=1, stdout="", stderr="rejected")
            if args[:3] == ["git", "rev-parse", "--abbrev-ref"]:
                return subprocess.CompletedProcess(args, returncode=0, stdout="main\n", stderr="")
            if args[:2] == ["git", "rebase"] and "--abort" not in args:
                return subprocess.CompletedProcess(args, returncode=1, stdout="", stderr="conflict")
            return subprocess.CompletedProcess(args, returncode=0, stdout="", stderr="")

        ok = commit_and_push_report(path, message="msg", max_retries=3, run_fn=_run_fn)
        assert ok is False


class TestRunAndCommitReport:
    def test_appends_then_commits(self, tmp_path: Path) -> None:
        path = tmp_path / "script.jsonl"
        fake = _FakeRun()
        ok = run_and_commit_report(_report(), path, run_fn=fake)
        assert ok is True
        assert path.exists()
        assert len(path.read_text().splitlines()) == 1

    def test_default_message_mentions_script_and_status(self, tmp_path: Path) -> None:
        path = tmp_path / "script.jsonl"
        fake = _FakeRun()
        run_and_commit_report(_report(), path, run_fn=fake)
        commit_call = next(call for call in fake.calls if call[:2] == ["git", "commit"])
        message = commit_call[3]
        assert "process_t1_kepler_batch" in message
        assert "success" in message
