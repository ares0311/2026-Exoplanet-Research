from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from exo_toolkit.ml.isolated_cnn_scorer import IsolatedCnnScorer


def test_predict_proba_uses_machine_readable_worker_protocol(tmp_path: Path) -> None:
    checkpoint = tmp_path / "best.pt"
    checkpoint.touch()
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(
            args, returncode=0, stdout='{"probability": 0.73}\n', stderr=""
        )

    scorer = IsolatedCnnScorer(checkpoint, run_fn=fake_run)
    assert scorer.predict_proba([0.0, 1.0]) == pytest.approx(0.73)
    assert json.loads(str(calls[0][1]["input"])) == {"snippet": [0.0, 1.0]}
    assert calls[0][0][-2:] == [
        "exo_toolkit.ml.cnn_inference_worker",
        str(checkpoint),
    ]


def test_predict_proba_fails_closed_on_worker_error(tmp_path: Path) -> None:
    checkpoint = tmp_path / "best.pt"
    checkpoint.touch()

    def fake_run(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, returncode=3, stdout="", stderr="unavailable")

    scorer = IsolatedCnnScorer(checkpoint, run_fn=fake_run)
    with pytest.raises(RuntimeError, match="exit 3"):
        scorer.predict_proba([0.0])


def test_training_mission_reads_sibling_config(tmp_path: Path) -> None:
    checkpoint = tmp_path / "best.pt"
    checkpoint.touch()
    (tmp_path / "config.json").write_text(
        json.dumps({"training_mission": "Kepler"}), encoding="utf-8"
    )
    assert IsolatedCnnScorer(checkpoint).training_mission == "Kepler"
