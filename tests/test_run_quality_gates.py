from __future__ import annotations

from pathlib import Path

import pytest
from Skills.run_quality_gates import (
    TEST_SHARD_COUNT,
    TOTAL_TEST_WORKERS,
    WORKERS_PER_SHARD,
    GateSpec,
    build_gate_specs,
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
    assert len(specs) == 8
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
