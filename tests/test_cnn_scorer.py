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

    def test_training_mission_none(self) -> None:
        s = CnnScorer.unavailable()
        assert s.training_mission is None


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

    def test_model_fn_scorer_training_mission_none_by_default(self) -> None:
        s = CnnScorer(model_fn=_dummy_model_fn)
        assert s.training_mission is None

    def test_model_fn_scorer_training_mission_explicit_override(self) -> None:
        s = CnnScorer(model_fn=_dummy_model_fn, training_mission="TESS")
        assert s.training_mission == "TESS"


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

    def test_loads_checkpoint_through_state_dict_loader(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        from types import SimpleNamespace

        import exo_toolkit.ml.cnn_scorer as cnn_scorer_module

        checkpoint = tmp_path / "best.pt"
        checkpoint.write_bytes(b"state-dict")
        loaded_model = object()
        batcher = SimpleNamespace(
            _load_torch_model=lambda path: (loaded_model, SimpleNamespace(n_bins=201))
        )
        monkeypatch.setattr(
            cnn_scorer_module,
            "import_module",
            lambda name: batcher if name == "Skills.cnn_inference_batcher" else None,
        )

        scorer = CnnScorer.from_checkpoint(checkpoint)

        assert scorer.is_available
        assert scorer._model is loaded_model

    def test_training_mission_loaded_from_checkpoint_config(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        from types import SimpleNamespace

        import exo_toolkit.ml.cnn_scorer as cnn_scorer_module

        checkpoint = tmp_path / "best.pt"
        checkpoint.write_bytes(b"state-dict")
        batcher = SimpleNamespace(
            _load_torch_model=lambda path: (
                object(),
                SimpleNamespace(n_bins=201, training_mission="Kepler"),
            )
        )
        monkeypatch.setattr(
            cnn_scorer_module,
            "import_module",
            lambda name: batcher if name == "Skills.cnn_inference_batcher" else None,
        )

        scorer = CnnScorer.from_checkpoint(checkpoint)

        assert scorer.training_mission == "Kepler"

    def test_training_mission_none_when_config_lacks_field(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """A checkpoint trained before training_mission existed must still load."""
        from types import SimpleNamespace

        import exo_toolkit.ml.cnn_scorer as cnn_scorer_module

        checkpoint = tmp_path / "best.pt"
        checkpoint.write_bytes(b"state-dict")
        batcher = SimpleNamespace(
            _load_torch_model=lambda path: (object(), SimpleNamespace(n_bins=201))
        )
        monkeypatch.setattr(
            cnn_scorer_module,
            "import_module",
            lambda name: batcher if name == "Skills.cnn_inference_batcher" else None,
        )

        scorer = CnnScorer.from_checkpoint(checkpoint)

        assert scorer.training_mission is None

    def test_from_checkpoint_training_mission_override_takes_precedence(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        from types import SimpleNamespace

        import exo_toolkit.ml.cnn_scorer as cnn_scorer_module

        checkpoint = tmp_path / "best.pt"
        checkpoint.write_bytes(b"state-dict")
        batcher = SimpleNamespace(
            _load_torch_model=lambda path: (
                object(),
                SimpleNamespace(n_bins=201, training_mission="Kepler"),
            )
        )
        monkeypatch.setattr(
            cnn_scorer_module,
            "import_module",
            lambda name: batcher if name == "Skills.cnn_inference_batcher" else None,
        )

        scorer = CnnScorer.from_checkpoint(checkpoint, training_mission="TESS")

        assert scorer.training_mission == "TESS"


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


# ---------------------------------------------------------------------------
# Regression: repo root must be reachable on sys.path for Skills.* imports
# ---------------------------------------------------------------------------
#
# A real incident (2026-07-10): invoking a Skills/*.py script directly as
# `python Skills/foo.py` makes Python auto-prepend only the script's own
# containing directory (Skills/) to sys.path -- not the repo root. Since
# CnnScorer._ensure_model()/_load() do the absolute import
# `import_module("Skills.cnn_inference_batcher")`, that import silently
# raised ModuleNotFoundError under this exact invocation shape, and the
# broad `except Exception` swallowed it -- every prediction fell back to
# the neutral 0.5 with no error, warning, or other signal. This corrupted
# an entire 588-row T1-2 K2 stacking-calibration batch (constant cnn_prob)
# before a manual sanity check caught it. See docs/PRODUCTION_READINESS.md
# T1-2 for the full incident record.


class TestEnsureRepoRootOnSysPath:
    def test_inserts_repo_root_when_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sys

        import exo_toolkit.ml.cnn_scorer as cnn_scorer_module

        repo_root = str(Path(cnn_scorer_module.__file__).resolve().parents[3])
        fake_path = [p for p in sys.path if p != repo_root]
        monkeypatch.setattr(sys, "path", fake_path)

        assert repo_root not in sys.path
        cnn_scorer_module._ensure_repo_root_on_sys_path()
        assert repo_root in sys.path

    def test_idempotent_when_already_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sys

        import exo_toolkit.ml.cnn_scorer as cnn_scorer_module

        repo_root = str(Path(cnn_scorer_module.__file__).resolve().parents[3])
        monkeypatch.setattr(sys, "path", [repo_root, *sys.path])
        before = list(sys.path)

        cnn_scorer_module._ensure_repo_root_on_sys_path()

        assert sys.path.count(repo_root) == before.count(repo_root)

    def test_ensure_model_loads_even_when_repo_root_missing_from_sys_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Reproduces the real 2026-07-10 incident and proves the fix."""
        import sys
        from types import SimpleNamespace

        import exo_toolkit.ml.cnn_scorer as cnn_scorer_module

        repo_root = str(Path(cnn_scorer_module.__file__).resolve().parents[3])
        monkeypatch.setattr(sys, "path", [p for p in sys.path if p != repo_root])
        assert repo_root not in sys.path

        checkpoint = tmp_path / "best.pt"
        checkpoint.write_bytes(b"state-dict")
        loaded_model = object()
        real_import_module = cnn_scorer_module.import_module

        def fake_import_module(name: str):
            if name == "Skills.cnn_inference_batcher":
                return SimpleNamespace(
                    _load_torch_model=lambda path: (
                        loaded_model,
                        SimpleNamespace(n_bins=201),
                    )
                )
            return real_import_module(name)

        monkeypatch.setattr(cnn_scorer_module, "import_module", fake_import_module)

        scorer = CnnScorer.from_checkpoint(checkpoint)

        assert scorer.is_available
        assert scorer._model is loaded_model


class TestEnsureModelWarnings:
    def test_warns_when_skills_package_unimportable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import exo_toolkit.ml.cnn_scorer as cnn_scorer_module

        checkpoint = tmp_path / "best.pt"
        checkpoint.write_bytes(b"state-dict")

        def raising_import(name: str):
            if name == "Skills.cnn_inference_batcher":
                raise ModuleNotFoundError("No module named 'Skills'")
            raise AssertionError(f"unexpected import: {name}")

        monkeypatch.setattr(cnn_scorer_module, "import_module", raising_import)

        with pytest.warns(RuntimeWarning, match="could not import Skills"):
            scorer = CnnScorer.from_checkpoint(checkpoint)

        assert scorer.is_available is False

    def test_warns_on_genuine_checkpoint_load_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from types import SimpleNamespace

        import exo_toolkit.ml.cnn_scorer as cnn_scorer_module

        checkpoint = tmp_path / "best.pt"
        checkpoint.write_bytes(b"state-dict")

        def raise_value_error(path):
            raise ValueError("architecture mismatch")

        batcher = SimpleNamespace(_load_torch_model=raise_value_error)
        monkeypatch.setattr(
            cnn_scorer_module,
            "import_module",
            lambda name: batcher if name == "Skills.cnn_inference_batcher" else None,
        )

        with pytest.warns(RuntimeWarning, match="failed to load checkpoint"):
            scorer = CnnScorer.from_checkpoint(checkpoint)

        assert scorer.is_available is False

    def test_silent_when_pytorch_not_installed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recwarn: pytest.WarningsRecorder
    ) -> None:
        """The documented 'PyTorch is optional' case must stay silent."""
        from types import SimpleNamespace

        import exo_toolkit.ml.cnn_scorer as cnn_scorer_module

        checkpoint = tmp_path / "best.pt"
        checkpoint.write_bytes(b"state-dict")

        def raise_import_error(path):
            raise ImportError("No module named 'torch'")

        batcher = SimpleNamespace(_load_torch_model=raise_import_error)
        monkeypatch.setattr(
            cnn_scorer_module,
            "import_module",
            lambda name: batcher if name == "Skills.cnn_inference_batcher" else None,
        )

        scorer = CnnScorer.from_checkpoint(checkpoint)

        assert scorer.is_available is False
        assert len(recwarn) == 0
