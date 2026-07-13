from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from Skills.build_candidate_context import build_candidate_context


def test_build_candidate_context_blends_sorts_and_reports(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {"label": 0, "xgb_prob": 0.2, "cnn_prob": 0.9, "bayes_prob": 0.4},
                {"label": 1, "xgb_prob": 0.8, "cnn_prob": 0.1, "bayes_prob": 0.6},
                {"label": 1, "xgb_prob": 0.5, "cnn_prob": 0.4, "bayes_prob": 0.5},
            )
        )
        + "\n",
        encoding="utf-8",
    )
    weights = tmp_path / "weights.json"
    weights.write_text(
        json.dumps({"w_xgb": 0.5, "w_cnn": 0.25, "w_bayes": 0.25}),
        encoding="utf-8",
    )
    output = tmp_path / "context.json"
    reports: list[tuple[Any, Path]] = []

    def fake_report(report: Any, path: Path) -> bool:
        reports.append((report, path))
        return True

    reference = build_candidate_context(
        predictions, weights, output, report_fn=fake_report
    )
    assert reference.scores_ascending == pytest.approx((0.425, 0.475, 0.575))
    assert reference.labels_aligned == (0, 1, 1)
    assert reference.n_positive == 2
    assert output.exists()
    assert reports[0][0].items_processed == 3
    assert reports[0][1].name == "build_candidate_context.jsonl"


def test_build_candidate_context_rejects_bad_weights(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        json.dumps({"label": 1, "xgb_prob": 0.5, "cnn_prob": 0.5, "bayes_prob": 0.5}),
        encoding="utf-8",
    )
    weights = tmp_path / "weights.json"
    weights.write_text(
        json.dumps({"w_xgb": 0.8, "w_cnn": 0.2, "w_bayes": 0.2}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="sum to one"):
        build_candidate_context(
            predictions,
            weights,
            tmp_path / "out.json",
            report_fn=lambda *_args, **_kwargs: True,
        )
