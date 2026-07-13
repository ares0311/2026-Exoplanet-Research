from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from exo_toolkit.candidate_context import (
    CandidateContextReference,
    contextualize_score,
    load_candidate_context,
)


def _reference() -> CandidateContextReference:
    return CandidateContextReference(
        context_id="context_v1",
        calibration_dataset_id="calibration_v1",
        score_name="full_ensemble_planet_probability",
        threshold_version="no_decision_threshold_v1",
        scores_ascending=(0.1, 0.4, 0.7, 0.9),
        labels_aligned=(0, 1, 0, 1),
        n_samples=4,
        n_positive=2,
        n_negative=2,
        source_predictions_path="predictions.jsonl",
        source_predictions_sha256="a" * 64,
        stacking_weights_path="weights.json",
        stacking_weights_sha256="b" * 64,
        limitations=("reference only",),
    )


def test_contextualize_score_reports_quantile_and_tail_fdr() -> None:
    context = contextualize_score(0.65, _reference())
    assert context["raw_score"] == pytest.approx(0.65)
    assert context["calibrated_score"] is None
    assert context["score_quantile"] == pytest.approx(0.5)
    assert context["false_discovery_reference_n"] == 2
    assert context["false_discovery_reference_negatives"] == 1
    assert context["false_discovery_estimate"] == pytest.approx(0.5)
    assert context["decision_threshold"] is None


def test_context_above_reference_has_no_fdr_estimate() -> None:
    context = contextualize_score(1.0, _reference())
    assert context["score_quantile"] == pytest.approx(1.0)
    assert context["false_discovery_reference_n"] == 0
    assert context["false_discovery_estimate"] is None


def test_reference_rejects_unsorted_or_inconsistent_samples() -> None:
    data = _reference().model_dump()
    data["scores_ascending"] = (0.4, 0.1, 0.7, 0.9)
    with pytest.raises(ValidationError, match="sorted"):
        CandidateContextReference.model_validate(data)
    data = _reference().model_dump()
    data["n_positive"] = 3
    with pytest.raises(ValidationError, match="n_positive"):
        CandidateContextReference.model_validate(data)


def test_reference_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "context.json"
    path.write_text(_reference().model_dump_json(), encoding="utf-8")
    assert load_candidate_context(path) == _reference()


def test_raw_score_must_be_probability() -> None:
    with pytest.raises(ValueError, match="raw_score"):
        contextualize_score(1.1, _reference())


def test_committed_candidate_context_has_exact_sources_and_no_threshold() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    reference = load_candidate_context(repo_root / "models/candidate_context_v1.json")
    assert reference.context_id == "full_ensemble_k2_empirical_context_v1"
    assert reference.calibration_dataset_id == "t1_2_k2pandc_calibration"
    assert reference.n_samples == 588
    assert reference.n_positive == 356
    assert reference.n_negative == 232
    assert reference.decision_threshold is None
    assert reference.threshold_version == "no_decision_threshold_v1"
    assert reference.source_predictions_sha256 == (
        "27694774052035da5a3ed31f2e1ef2edf6149be2fac972b75589adb261831278"
    )
    assert reference.stacking_weights_sha256 == (
        "e9f73f2d162847e5621dc61916e0f39032f18ba082f3d1cc7429eed9ad89054b"
    )
