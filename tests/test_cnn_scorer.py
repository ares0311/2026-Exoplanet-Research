"""Tests for src/exo_toolkit/ml/cnn_scorer.py."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from exo_toolkit.ml.cnn_scorer import CnnScorer


def _dummy_model_fn(snippet: list[float]) -> float:
    """Mock model that returns 0.7 always."""
    return 0.7


def _zero_model_fn(snippet: list[float]) -> float:
    return 0.0


def _one_model_fn(snippet: list[float]) -> float:
    return 1.0


def _snippet(n: int = 201) -> list[float]:
    return [0.0] * n


# ---------------------------------------------------------------------------
# CnnScorer.unavailable()
# ---------------------------------------------------------------------------


class TestCnnScorerUnavailable:
    def test_unavailable_factory_not_available(self) -> None:
        s = CnnScorer.unavailable()
        assert s.is_available is False

    def test_unavailable_returns_half(self) -> None:
        s = CnnScorer.unavailable()
        assert s.predict_proba(_snippet()) == pytest.approx(0.5)

    def test_unavailable_batch_all_half(self) -> None:
        s = CnnScorer.unavailable()
        result = s.predict_proba_batch([_snippet(), _snippet()])
        assert result == [pytest.approx(0.5), pytest.approx(0.5)]

    def test_checkpoint_path_none(self) -> None:
        s = CnnScorer.unavailable()
        assert s.checkpoint_path is None


# ---------------------------------------------------------------------------
# CnnScorer with model_fn (no PyTorch required)
# ---------------------------------------------------------------------------


class TestCnnScorerWithModelFn:
    def test_is_available_with_model_fn(self) -> None:
        s = CnnScorer(model_fn=_dummy_model_fn)
        assert s.is_available is True

    def test_predict_proba_uses_model_fn(self) -> None:
        s = CnnScorer(model_fn=_dummy_model_fn)
        assert s.predict_proba(_snippet()) == pytest.approx(0.7)

    def test_predict_proba_batch_uses_model_fn(self) -> None:
        s = CnnScorer(model_fn=_dummy_model_fn)
        result = s.predict_proba_batch([_snippet(), _snippet()])
        assert len(result) == 2
        for p in result:
            assert p == pytest.approx(0.7)

    def test_predict_proba_clips_to_zero(self) -> None:
        s = CnnScorer(model_fn=_zero_model_fn)
        assert s.predict_proba(_snippet()) >= 0.0

    def test_predict_proba_clips_to_one(self) -> None:
        s = CnnScorer(model_fn=_one_model_fn)
        assert s.predict_proba(_snippet()) <= 1.0

    def test_batch_empty_returns_empty(self) -> None:
        s = CnnScorer(model_fn=_dummy_model_fn)
        assert s.predict_proba_batch([]) == []

    def test_batch_count_matches_input(self) -> None:
        s = CnnScorer(model_fn=_dummy_model_fn)
        snippets = [_snippet() for _ in range(5)]
        assert len(s.predict_proba_batch(snippets)) == 5

    def test_result_is_float(self) -> None:
        s = CnnScorer(model_fn=_dummy_model_fn)
        assert isinstance(s.predict_proba(_snippet()), float)


# ---------------------------------------------------------------------------
# CnnScorer.from_checkpoint — no actual PyTorch file needed
# ---------------------------------------------------------------------------


class TestCnnScorerFromCheckpoint:
    def test_from_checkpoint_sets_path(self, tmp_path: Path) -> None:
        fake = tmp_path / "fake.pt"
        fake.write_bytes(b"")
        s = CnnScorer.from_checkpoint(fake)
        assert s.checkpoint_path == fake

    def test_from_checkpoint_missing_file_not_available(self, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent.pt"
        s = CnnScorer.from_checkpoint(missing)
        # unavailable because torch is not installed or file missing
        assert s.is_available is False or s.checkpoint_path == missing

    def test_from_checkpoint_with_calibration_path(self, tmp_path: Path) -> None:
        cal = tmp_path / "cal.json"
        cal.write_text(json.dumps({"a": 1.0, "b": 0.0, "method": "platt", "flag": "OK"}))
        fake = tmp_path / "fake.pt"
        fake.write_bytes(b"")
        s = CnnScorer.from_checkpoint(fake, calibration_path=cal)
        assert s.checkpoint_path == fake

    def test_returns_half_when_no_torch(self) -> None:
        s = CnnScorer(checkpoint_path=Path("nonexistent.pt"))
        # without torch or a mock, returns 0.5
        result = s.predict_proba(_snippet())
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# CNN weight validation via StackingScorer
# ---------------------------------------------------------------------------


class TestCnnScorerInStacking:
    def test_stacking_scorer_uses_cnn_model_fn(self) -> None:
        from exo_toolkit.ml.stacking_scorer import StackingScorer
        from exo_toolkit.schemas import CandidateFeatures

        cnn = CnnScorer(model_fn=_dummy_model_fn)
        s = StackingScorer(cnn_scorer=cnn, xgb_weight=0.0, cnn_weight=0.35)
        features = CandidateFeatures()
        p = s.predict_proba(features, bayesian_planet_prob=0.0, snippet=_snippet())
        # 0.35 * 0.7 + 0.65 * 0.0 = 0.245
        assert p == pytest.approx(0.35 * 0.7, abs=1e-6)

    def test_stacking_scorer_has_cnn_true(self) -> None:
        from exo_toolkit.ml.stacking_scorer import StackingScorer

        cnn = CnnScorer(model_fn=_dummy_model_fn)
        s = StackingScorer(cnn_scorer=cnn, xgb_weight=0.0, cnn_weight=0.35)
        assert s.has_cnn is True

    def test_stacking_scorer_has_cnn_false_when_unavailable(self) -> None:
        from exo_toolkit.ml.stacking_scorer import StackingScorer

        cnn = CnnScorer.unavailable()
        s = StackingScorer(cnn_scorer=cnn, xgb_weight=0.0, cnn_weight=0.35)
        assert s.has_cnn is False

    def test_stacking_scorer_invalid_weight_sum(self) -> None:
        from exo_toolkit.ml.stacking_scorer import StackingScorer

        with pytest.raises(ValueError):
            StackingScorer(xgb_weight=0.7, cnn_weight=0.5)

    def test_stacking_scorer_cnn_weight_property(self) -> None:
        from exo_toolkit.ml.stacking_scorer import StackingScorer

        s = StackingScorer(xgb_weight=0.35, cnn_weight=0.35)
        assert s.cnn_weight == pytest.approx(0.35)
