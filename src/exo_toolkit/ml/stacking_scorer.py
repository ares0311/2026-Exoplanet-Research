"""Tier-3 stacking scorer: weighted average of XGBoost + CNN + Bayesian posteriors.

Blends up to three probability sources:
- XGBoost (tabular features) — Tier 1
- CNN (phase-folded flux) — Tier 2
- Bayesian log-score model — fallback

Weights default to 0.35/0.35/0.30 (XGBoost/CNN/Bayesian) per CNN_SPEC.md.
Falls back gracefully to Bayesian-only when no models are loaded.

Public API
----------
``StackingScorer``
    .predict_proba(features, bayesian_planet_prob, snippet=None) → float
    .predict_proba_batch(features_list, bayesian_planet_probs, snippets=None) → np.ndarray
    .from_model_path(path, xgb_weight) (classmethod) — XGBoost only
    .from_model_paths(xgb_path, cnn_path, *, xgb_weight, cnn_weight) (classmethod)
    .bayesian_only() (classmethod)
    .from_weights_file(weights_path, *, xgb_scorer, cnn_scorer) (classmethod)
    .xgb_weight → float
    .cnn_weight → float
    .has_xgb → bool
    .has_cnn → bool
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from exo_toolkit.schemas import CandidateFeatures

if TYPE_CHECKING:
    from exo_toolkit.ml.cnn_scorer import CnnScorer
    from exo_toolkit.ml.xgboost_scorer import XGBoostScorer


class StackingScorer:
    """Weighted-average ensemble of XGBoost, CNN, and Bayesian planet probabilities.

    When no models are attached, prediction calls return the Bayesian probability
    unchanged — making it safe to use ``StackingScorer`` as the universal scoring
    interface regardless of which tiers are available.

    Args:
        xgb_scorer: A trained ``XGBoostScorer`` instance, or ``None``.
        cnn_scorer: A trained ``CnnScorer`` instance, or ``None``.
        xgb_weight: Weight given to the XGBoost probability in the blend.
        cnn_weight: Weight given to the CNN probability in the blend.
            Bayesian weight = ``1 - xgb_weight - cnn_weight``.

    Note:
        ``xgb_weight + cnn_weight`` must be <= 1.0.
    """

    def __init__(
        self,
        xgb_scorer: XGBoostScorer | None = None,
        cnn_scorer: CnnScorer | None = None,
        xgb_weight: float = 0.5,
        cnn_weight: float = 0.0,
    ) -> None:
        if not 0.0 <= xgb_weight <= 1.0:
            raise ValueError(f"xgb_weight must be in [0, 1], got {xgb_weight}")
        if not 0.0 <= cnn_weight <= 1.0:
            raise ValueError(f"cnn_weight must be in [0, 1], got {cnn_weight}")
        if xgb_weight + cnn_weight > 1.0:
            raise ValueError(
                f"xgb_weight + cnn_weight must be <= 1.0, got {xgb_weight + cnn_weight}"
            )
        self._xgb_scorer = xgb_scorer
        self._cnn_scorer = cnn_scorer
        self._xgb_weight = xgb_weight
        self._cnn_weight = cnn_weight

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def xgb_weight(self) -> float:
        """Weight given to the XGBoost probability in the blend."""
        return self._xgb_weight

    @property
    def cnn_weight(self) -> float:
        """Weight given to the CNN probability in the blend."""
        return self._cnn_weight

    @property
    def has_xgb(self) -> bool:
        """True if an XGBoost scorer is attached."""
        return self._xgb_scorer is not None

    @property
    def has_cnn(self) -> bool:
        """True if a CNN scorer is attached and available."""
        return self._cnn_scorer is not None and getattr(
            self._cnn_scorer, "is_available", False
        )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_proba(
        self,
        features: CandidateFeatures,
        bayesian_planet_prob: float,
        snippet: list[float] | None = None,
    ) -> float:
        """Return blended P(planet_candidate) for a single candidate.

        Args:
            features: A ``CandidateFeatures`` instance (for XGBoost).
            bayesian_planet_prob: P(planet_candidate) from the Bayesian model.
            snippet: Phase-folded 201-bin flux array for CNN scoring.

        Returns:
            Blended probability in [0, 1].
        """
        bayes = float(bayesian_planet_prob)
        total_w = 0.0
        blended = 0.0

        if self._xgb_scorer is not None:
            xgb_prob: float = self._xgb_scorer.predict_proba(features)
            blended += self._xgb_weight * xgb_prob
            total_w += self._xgb_weight

        if self.has_cnn and snippet is not None:
            cnn_prob: float = self._cnn_scorer.predict_proba(snippet)  # type: ignore[union-attr]
            blended += self._cnn_weight * cnn_prob
            total_w += self._cnn_weight

        bayes_w = max(0.0, 1.0 - total_w)
        blended += bayes_w * bayes
        return float(max(0.0, min(1.0, blended)))

    def predict_proba_batch(
        self,
        features_list: list[CandidateFeatures],
        bayesian_planet_probs: np.ndarray,
        snippets: list[list[float]] | None = None,
    ) -> np.ndarray:
        """Return blended P(planet_candidate) for a batch of candidates.

        Args:
            features_list: List of ``CandidateFeatures`` instances.
            bayesian_planet_probs: 1-D array of Bayesian P(planet) values.
            snippets: List of 201-bin flux arrays for CNN scoring.

        Returns:
            1-D ``float64`` array of blended probabilities.
        """
        bayes = np.array(bayesian_planet_probs, dtype=np.float64)
        total_w = 0.0
        blended = np.zeros_like(bayes)

        if self._xgb_scorer is not None:
            xgb_probs: np.ndarray = self._xgb_scorer.predict_proba_batch(features_list)
            blended += self._xgb_weight * xgb_probs
            total_w += self._xgb_weight

        if self.has_cnn and snippets is not None:
            cnn_probs_list = self._cnn_scorer.predict_proba_batch(snippets)  # type: ignore[union-attr]
            cnn_probs = np.array(cnn_probs_list, dtype=np.float64)
            blended += self._cnn_weight * cnn_probs
            total_w += self._cnn_weight

        bayes_w = max(0.0, 1.0 - total_w)
        blended += bayes_w * bayes
        return np.clip(blended, 0.0, 1.0).astype(np.float64)

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_model_path(
        cls,
        path: str | Path,
        xgb_weight: float = 0.5,
    ) -> StackingScorer:
        """Load a saved ``XGBoostScorer`` and wrap it in a ``StackingScorer``.

        Args:
            path: Path to the XGBoostScorer metadata JSON file.
            xgb_weight: Weight for the XGBoost probability (default: 0.5).

        Returns:
            A ``StackingScorer`` ready for prediction.
        """
        from exo_toolkit.ml.xgboost_scorer import XGBoostScorer

        xgb_scorer = XGBoostScorer.load(path)
        return cls(xgb_scorer=xgb_scorer, xgb_weight=xgb_weight, cnn_weight=0.0)

    @classmethod
    def from_model_paths(
        cls,
        xgb_path: str | Path,
        cnn_path: str | Path,
        *,
        xgb_weight: float = 0.35,
        cnn_weight: float = 0.35,
        cnn_calibration_path: str | Path | None = None,
    ) -> StackingScorer:
        """Load both XGBoost and CNN scorers.

        Args:
            xgb_path: Path to the XGBoostScorer metadata JSON.
            cnn_path: Path to the CNN ``.pt`` checkpoint.
            xgb_weight: XGBoost blend weight (default: 0.35).
            cnn_weight: CNN blend weight (default: 0.35).
            cnn_calibration_path: Optional Platt calibration JSON for the CNN.

        Returns:
            A ``StackingScorer`` with both tiers configured.
        """
        from exo_toolkit.ml.cnn_scorer import CnnScorer
        from exo_toolkit.ml.xgboost_scorer import XGBoostScorer

        xgb_scorer = XGBoostScorer.load(xgb_path)
        cnn = CnnScorer.from_checkpoint(cnn_path, calibration_path=cnn_calibration_path)
        return cls(
            xgb_scorer=xgb_scorer,
            cnn_scorer=cnn,
            xgb_weight=xgb_weight,
            cnn_weight=cnn_weight,
        )

    @classmethod
    def bayesian_only(cls) -> StackingScorer:
        """Return a scorer that passes through the Bayesian probability unchanged.

        Useful as a no-op drop-in when no trained models are available.
        """
        return cls(xgb_scorer=None, cnn_scorer=None, xgb_weight=0.5, cnn_weight=0.0)

    @classmethod
    def from_weights_file(
        cls,
        weights_path: str | Path,
        *,
        xgb_scorer: XGBoostScorer | None = None,
        cnn_scorer: CnnScorer | None = None,
    ) -> StackingScorer:
        """Load blend weights from a stacking_weights.json file.

        The JSON file must contain ``w_xgb`` and ``w_cnn`` keys, as written by
        ``Skills/calibrate_stacking_weights.py``.

        Args:
            weights_path: Path to the weights JSON file.
            xgb_scorer: Optional pre-loaded ``XGBoostScorer``.
            cnn_scorer: Optional pre-loaded ``CnnScorer``.

        Returns:
            A ``StackingScorer`` with calibrated blend weights.

        Raises:
            ValueError: If ``w_xgb`` or ``w_cnn`` keys are missing.
        """
        import json

        weights = json.loads(Path(weights_path).read_text(encoding="utf-8"))
        for key in ("w_xgb", "w_cnn"):
            if key not in weights:
                raise ValueError(f"Missing key '{key}' in {weights_path}")
        return cls(
            xgb_scorer=xgb_scorer,
            cnn_scorer=cnn_scorer,
            xgb_weight=float(weights["w_xgb"]),
            cnn_weight=float(weights["w_cnn"]),
        )
