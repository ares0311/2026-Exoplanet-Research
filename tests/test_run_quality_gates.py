from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from Skills.run_quality_gates import (
    TEST_SHARD_COUNT,
    TOTAL_TEST_WORKERS,
    WORKERS_PER_SHARD,
    GateOutcome,
    GateSpec,
    _git_state,
    build_gate_specs,
    main,
    partition_test_files,
    supervise_gates,
)


def _test_file(path: Path, test_count: int) -> Path:
    path.write_text("\n".join(f"def test_{index}(): pass" for index in range(test_count)))
    return path


def test_partition_assigns_every_file_exactly_once_and_balances(tmp_path: Path) -> None:
    files = [_test_file(tmp_path / f"test_{index}.py", index + 1) for index in range(12)]
    shards = partition_test_files(files)
    flattened = [path for shard in shards for path in shard]
    assert len(shards) == TEST_SHARD_COUNT == 6
    assert sorted(flattened) == sorted(files)
    assert len(flattened) == len(set(flattened))
    weights = [
        sum(index + 1 for index, path in enumerate(files) if path in shard) for shard in shards
    ]
    assert max(weights) - min(weights) <= max(range(1, 13))


def test_partition_rejects_too_few_files(tmp_path: Path) -> None:
    files = [_test_file(tmp_path / f"test_{index}.py", 1) for index in range(5)]
    with pytest.raises(ValueError, match="need at least 6"):
        partition_test_files(files)


def test_build_specs_creates_six_disjoint_six_worker_pytest_commands(tmp_path: Path) -> None:
    files = [_test_file(tmp_path / f"test_{index}.py", index + 1) for index in range(12)]
    relative_files = [Path("tests") / path.name for path in files]
    for source, relative in zip(files, relative_files, strict=True):
        destination = tmp_path / relative
        destination.parent.mkdir(exist_ok=True)
        destination.write_text(source.read_text())

    # The command builder requires paths beneath the real repository root.
    repo_files = list((Path(__file__).parent).glob("test_*.py"))[:12]
    specs = build_gate_specs(repo_files, python_executable=".venv/bin/python")
    pytest_specs = [spec for spec in specs if spec.name.startswith("pytest_shard_")]
    static_specs = [spec for spec in specs if not spec.name.startswith("pytest_shard_")]
    assert len(specs) == 10
    assert {spec.name for spec in static_specs} == {
        "ruff",
        "mypy",
        "incomplete_implementations",
        "directive_integrity",
    }
    assert len(pytest_specs) == TEST_SHARD_COUNT
    assert TOTAL_TEST_WORKERS == 36
    seen: list[str] = []
    for spec in pytest_specs:
        assert spec.command[3:6] == ("-n", str(WORKERS_PER_SHARD), "--dist=worksteal")
        assert spec.env is not None
        assert spec.env["PYTHONPATH"] == "src"
        assert spec.env["OMP_NUM_THREADS"] == "1"
        assert spec.env["VECLIB_MAXIMUM_THREADS"] == "1"
        seen.extend(argument for argument in spec.command if argument.startswith("tests/"))
    assert len(seen) == len(set(seen)) == len(repo_files)


class _FakeProcess:
    def __init__(self, returncode: int, pid: int) -> None:
        self.returncode = returncode
        self.pid = pid

    def poll(self) -> int:
        return self.returncode

    def terminate(self) -> None:
        return None

    def wait(self, timeout: float) -> int:
        return self.returncode

    def kill(self) -> None:
        return None


def test_supervisor_runs_all_specs_with_separate_logs(tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    def factory(command: list[str], **kwargs: object) -> _FakeProcess:
        calls.append({"command": command, **kwargs})
        return _FakeProcess(returncode=0 if len(calls) == 1 else 3, pid=200 + len(calls))

    specs = (
        GateSpec("first", ("python", "first.py")),
        GateSpec("second", ("python", "second.py"), {"PYTHONPATH": "src"}),
    )
    outcomes = supervise_gates(
        specs,
        tmp_path / "logs",
        heartbeat_seconds=1.0,
        popen_factory=factory,
        sleep_fn=lambda _seconds: None,
    )
    assert [outcome.returncode for outcome in outcomes] == [0, 3]
    assert len({outcome.log_path for outcome in outcomes}) == 2
    assert all(Path(outcome.log_path).exists() for outcome in outcomes)
    assert calls[1]["env"]["PYTHONPATH"] == "src"  # type: ignore[index]


def _init_committed_repo(root: Path) -> None:
    subprocess.run(("git", "init"), cwd=root, check=True, capture_output=True)
    subprocess.run(("git", "config", "user.email", "test@example.com"), cwd=root, check=True)
    subprocess.run(("git", "config", "user.name", "Test"), cwd=root, check=True)
    (root / "file.txt").write_text("content\n")
    subprocess.run(("git", "add", "file.txt"), cwd=root, check=True)
    subprocess.run(
        ("git", "commit", "-m", "initial"), cwd=root, check=True, capture_output=True
    )


class TestGitState:
    def test_clean_repo_reports_sha_and_not_dirty(self, tmp_path: Path) -> None:
        _init_committed_repo(tmp_path)

        state = _git_state(tmp_path)
        expected_sha = subprocess.run(
            ("git", "rev-parse", "HEAD"), cwd=tmp_path, capture_output=True, text=True, check=True
        ).stdout.strip()
        assert state["git_head_sha"] == expected_sha
        assert state["git_dirty"] is False

    def test_uncommitted_change_reports_dirty(self, tmp_path: Path) -> None:
        _init_committed_repo(tmp_path)
        (tmp_path / "file.txt").write_text("changed\n")

        state = _git_state(tmp_path)
        assert state["git_dirty"] is True

    def test_non_git_directory_fails_loudly_not_silently(self, tmp_path: Path) -> None:
        state = _git_state(tmp_path)
        assert state["git_head_sha"] is None
        assert state["git_dirty"] is None
        assert "git_state_error" in state


class TestMainCapturesGitStateBeforeGatesRun:
    def test_git_state_is_captured_before_supervise_gates_runs(self) -> None:
        # Regression: the summary JSON must record the tree state the gates
        # actually verified, not whatever state exists after they finish --
        # a commit landing during the run (plausible in this repo, which has
        # 15+ scripts that auto-commit Run Reports) must not be silently
        # attributed to the result. Proven here via call order, not just the
        # final summary contents, since a stale git_state_fn could otherwise
        # coincidentally produce the same value regardless of when it's called.
        call_order: list[str] = []
        created_log_dirs: list[Path] = []

        def fake_git_state(repo_root: Path) -> dict[str, object]:
            call_order.append("git_state")
            return {"git_head_sha": "before-sha", "git_dirty": False}

        def fake_supervise(
            specs: tuple[GateSpec, ...], log_dir: Path, *, heartbeat_seconds: float
        ) -> tuple[GateOutcome, ...]:
            call_order.append("supervise_gates")
            created_log_dirs.append(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            outcomes = []
            for spec in specs:
                log_path = log_dir / f"{spec.name}.log"
                log_path.write_text("ok\n")
                outcomes.append(
                    GateOutcome(
                        name=spec.name,
                        returncode=0,
                        elapsed_seconds=0.01,
                        log_path=str(log_path),
                    )
                )
            return tuple(outcomes)

        try:
            exit_code = main(
                ["--heartbeat-seconds", "1.0"],
                supervise_gates_fn=fake_supervise,
                git_state_fn=fake_git_state,
            )
            assert exit_code == 0
            assert call_order == ["git_state", "supervise_gates"]
            summary = json.loads(
                (created_log_dirs[0] / "quality_gate_summary.json").read_text()
            )
            assert summary["git_head_sha"] == "before-sha"
            assert summary["git_dirty"] is False
        finally:
            for log_dir in created_log_dirs:
                shutil.rmtree(log_dir, ignore_errors=True)
