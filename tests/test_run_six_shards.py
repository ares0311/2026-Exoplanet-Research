from __future__ import annotations

import json
from pathlib import Path

import pytest
from Skills.run_six_shards import (
    SHARD_COUNT,
    WORKERS_PER_SHARD,
    build_shard_commands,
    supervise_shards,
)


def test_build_commands_injects_exact_six_by_six() -> None:
    commands = build_shard_commands(
        "process_t1_kepler_batch.py",
        ["--", "--max-targets", "1000"],
        python_executable=".venv/bin/python",
    )
    assert len(commands) == SHARD_COUNT == 6
    for index, command in enumerate(commands):
        assert command[0] == ".venv/bin/python"
        assert command[-6:] == (
            "--workers",
            str(WORKERS_PER_SHARD),
            "--shard-index",
            str(index),
            "--shard-count",
            str(SHARD_COUNT),
        )
        assert "--max-targets" in command


def test_crossmatch_pilot_is_reviewed_for_six_by_six() -> None:
    commands = build_shard_commands(
        "crossmatch_tess_catalina_labels.py",
        ["--", "--max-targets", "216", "--batch-size", "6"],
        python_executable=".venv/bin/python",
    )
    assert len(commands) == 6
    assert all("--max-targets" in command for command in commands)


@pytest.mark.parametrize("flag", ["--workers", "--workers=3", "--shard-index"])
def test_build_commands_rejects_launcher_owned_flags(flag: str) -> None:
    with pytest.raises(ValueError, match="controlled by this launcher"):
        build_shard_commands("process_t1_kepler_batch.py", [flag])


def test_build_commands_rejects_unreviewed_script() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        build_shard_commands("arbitrary.py", [])


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


def test_supervisor_returns_all_outcomes_and_separate_logs(tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    def factory(command: list[str], **kwargs: object) -> _FakeProcess:
        calls.append({"command": command, **kwargs})
        return _FakeProcess(returncode=0 if len(calls) < 3 else 7, pid=100 + len(calls))

    outcomes = supervise_shards(
        [("python", "worker.py", str(index)) for index in range(3)],
        tmp_path / "logs",
        start_delay_seconds=0.0,
        heartbeat_seconds=1.0,
        popen_factory=factory,
        sleep_fn=lambda _seconds: None,
    )
    assert [outcome.returncode for outcome in outcomes] == [0, 0, 7]
    assert len({outcome.log_path for outcome in outcomes}) == 3
    assert all(Path(outcome.log_path).exists() for outcome in outcomes)
    assert all("EXO_RUN_REPORT_LOCK_PATH" in call["env"] for call in calls)  # type: ignore[operator]


def test_launcher_summary_shape_is_json_serializable(tmp_path: Path) -> None:
    commands = build_shard_commands("fetch_t1_2_k2_calibration_snippets.py", [])
    payload = {"shards": len(commands), "commands": commands}
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(payload))
    assert json.loads(path.read_text())["shards"] == 6
