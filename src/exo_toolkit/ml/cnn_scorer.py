"""CNN scorer wrapper — same interface as XGBoostScorer.

Wraps the CNN inference pipeline (``Skills/cnn_inference_batcher.py``) behind
the same ``predict_proba`` / ``predict_proba_batch`` interface used by
``XGBoostScorer``, so that ``StackingScorer`` can blend all three tiers without
knowing which tier it is talking to.

PyTorch is optional: if it is not installed, every method returns 0.5 and
``is_available`` is ``False``.

Public API
----------
``CnnScorer``
    .predict_proba(snippet) → float
    .predict_proba_batch(snippets) → list[float]
    .from_checkpoint(path, *, calibration_path) (classmethod)
    .is_available → bool
    .checkpoint_path → Path | None
    .training_mission → str | None
"""
from __future__ import annotations

import sys
import warnings
from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from typing import Any


def _ensure_repo_root_on_sys_path() -> None:
    """Insert the repo root onto ``sys.path`` if it is not already there.

    ``Skills.cnn_inference_batcher`` / ``Skills.cnn_calibrator`` are absolute
    imports that require the repo root itself (not just ``src/``) on
    ``sys.path``. A caller script that is invoked directly as
    ``python Skills/foo.py`` only gets its own containing directory
    (``Skills/``) auto-prepended by Python -- not the repo root -- so without
    this, ``import_module("Skills....")`` silently raises
    ``ModuleNotFoundError``, which the broad ``except Exception`` below
    swallows: every CNN score then falls back to the neutral 0.5 with no
    error, warning, or other indication anything went wrong. This is not
    hypothetical: it produced flat 0.5 ``cnn_prob`` values across an entire
    588-row T1-2 K2 calibration batch, which fed a bogus CNN=0.000 stacking
    weight into an AUC-maximizing grid search before being caught by a
    manual sanity check. See ``docs/PRODUCTION_READINESS.md`` T1-2 for the
    incident record.
    """
    repo_root = str(Path(__file__).resolve().parents[3])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


class CnnScorer:
    """Wraps a trained CNN checkpoint for planet-candidate probability prediction.

    Args:
        checkpoint_path: Path to the ``.pt`` model file produced by ``train_cnn.py``.
        calibration_path: Optional path to a Platt calibration JSON written by
            ``cnn_calibrator.py``.  Applied post-softmax when present.
        model_fn: Injectable callable ``(snippet: list[float]) -> float`` for
            testing without PyTorch.  When provided, the checkpoint file is not
            loaded.

    Attributes:
        is_available: ``True`` when PyTorch or a mock ``model_fn`` is usable.
    """

    def __init__(
        self,
        checkpoint_path: Path | None = None,
        *,
        calibration_path: Path | None = None,
        model_fn: Callable[[list[float]], float] | None = None,
        training_mission: str | None = None,
    ) -> None:
        self._checkpoint_path = checkpoint_path
        self._calibration_path = calibration_path
        self._model_fn = model_fn
        self._model: Any = None
        self._calibration: Any = None
        self._available: bool = False
        self._training_mission: str | None = training_mission
        self._load()

    def _load(self) -> None:
        if self._model_fn is not None:
            self._available = True
        elif self._checkpoint_path is not None and self._checkpoint_path.exists():
            self._ensure_model()
            self._available = self._model is not None

        if self._calibration_path is not None and self._calibration_path.exists():
            try:
                _ensure_repo_root_on_sys_path()
                calibrator = import_module("Skills.cnn_calibrator")
                calibration = calibrator.load_cnn_calibration(self._calibration_path)
                if calibration.flag == "OK":
                    self._calibration = calibration
            except Exception:  # noqa: BLE001
                pass

    def _ensure_model(self) -> None:
        if self._model_fn is not None or self._model is not None:
            return
        if self._checkpoint_path is None:
            return
        _ensure_repo_root_on_sys_path()
        try:
            batcher = import_module("Skills.cnn_inference_batcher")
        except Exception as exc:  # noqa: BLE001
            # Skills is a required project module, not an optional dependency
            # like PyTorch -- unlike the ImportError case below, this should
            # never happen and always deserves a warning.
            self._model = None
            self._available = False
            warnings.warn(
                f"CnnScorer could not import Skills.cnn_inference_batcher "
                f"({type(exc).__name__}: {exc}); every prediction will silently "
                "fall back to the neutral 0.5. Check that the repo root is on "
                "sys.path.",
                RuntimeWarning,
                stacklevel=2,
            )
            return
        try:
            self._model, config = batcher._load_torch_model(self._checkpoint_path)
            if self._training_mission is None:
                self._training_mission = getattr(config, "training_mission", None)
        except ImportError:
            # PyTorch itself is not installed -- documented, expected, silent
            # (see module docstring: "PyTorch is optional").
            self._model = None
            self._available = False
        except Exception as exc:  # noqa: BLE001
            self._model = None
            self._available = False
            warnings.warn(
                f"CnnScorer failed to load checkpoint {self._checkpoint_path} "
                f"({type(exc).__name__}: {exc}); every prediction will silently "
                "fall back to the neutral 0.5.",
                RuntimeWarning,
                stacklevel=2,
            )

    def _apply_calibration(self, p: float) -> float:
        if self._calibration is None:
            return p
        try:
            _ensure_repo_root_on_sys_path()
            calibrator = import_module("Skills.cnn_calibrator")
            return float(calibrator.apply_cnn_calibration(p, self._calibration))
        except Exception:  # noqa: BLE001
            return p

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_available(self) -> bool:
        """True when this scorer can produce predictions."""
        return self._available

    @property
    def checkpoint_path(self) -> Path | None:
        """Path to the loaded checkpoint file."""
        return self._checkpoint_path

    @property
    def training_mission(self) -> str | None:
        """Mission this checkpoint was trained on (``"TESS"``/``"Kepler"``/``"K2"``/
        ``"JWST"``), or ``None`` if undeclared (e.g. a checkpoint trained before
        ``training_mission`` existed, or a ``model_fn``-based test double).

        Loaded from the sibling ``config.json``'s ``training_mission`` field the
        first time the model is loaded; an explicit constructor value always
        takes precedence.
        """
        return self._training_mission

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_proba(self, snippet: list[float]) -> float:
        """Return P(planet_candidate) for a single phase-folded snippet.

        Args:
            snippet: 1-D list of normalised flux values (length 201).

        Returns:
            Probability in [0, 1]; returns 0.5 when CNN is not available.
        """
        if not self._available:
            return 0.5
        if self._model_fn is not None:
            p = float(self._model_fn(snippet))
            return self._apply_calibration(max(0.0, min(1.0, p)))
        self._ensure_model()
        if self._model is None:
            return 0.5
        try:
            return self.predict_proba_batch([snippet])[0]
        except Exception:  # noqa: BLE001
            return 0.5

    def predict_proba_batch(self, snippets: list[list[float]]) -> list[float]:
        """Return P(planet_candidate) for a batch of snippets.

        Args:
            snippets: List of 1-D flux arrays.

        Returns:
            List of probabilities, one per snippet.  Returns [0.5, ...] when
            the CNN is not available.
        """
        if not snippets:
            return []
        if not self._available:
            return [0.5] * len(snippets)
        if self._model_fn is not None:
            return [
                self._apply_calibration(max(0.0, min(1.0, float(self._model_fn(s)))))
                for s in snippets
            ]
        self._ensure_model()
        if self._model is None:
            return [0.5] * len(snippets)
        try:
            import torch  # noqa: PLC0415

            tensor = torch.tensor(snippets, dtype=torch.float32).unsqueeze(1)
            with torch.no_grad():
                out = self._model(tensor)
                probs: list[float] = out.flatten().tolist()
            return [self._apply_calibration(max(0.0, min(1.0, float(p)))) for p in probs]
        except Exception:  # noqa: BLE001
            return [0.5] * len(snippets)

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        calibration_path: str | Path | None = None,
        training_mission: str | None = None,
    ) -> CnnScorer:
        """Load a ``CnnScorer`` from a saved checkpoint.

        Args:
            path: Path to the ``.pt`` checkpoint file.
            calibration_path: Optional path to a Platt calibration JSON.
            training_mission: Explicit mission override, taking precedence over
                whatever (if anything) the sibling ``config.json`` declares.
                Intended only for deliberately re-declaring the mission of a
                checkpoint trained before ``training_mission`` existed; do not
                use this to paper over the cross-mission scoring guard in
                ``run_pipeline`` — that has its own explicit
                ``allow_cross_mission_cnn`` escape hatch for that purpose.

        Returns:
            A :class:`CnnScorer` ready for prediction.
        """
        cal = Path(calibration_path) if calibration_path is not None else None
        return cls(Path(path), calibration_path=cal, training_mission=training_mission)

    @classmethod
    def unavailable(cls) -> CnnScorer:
        """Return a no-op scorer that always returns 0.5.

        Useful as a drop-in when no CNN checkpoint is available.
        """
        return cls(checkpoint_path=None)
