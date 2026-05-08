"""Tier-3 stacking scorer: weighted average of XGBoost + Bayesian posteriors.

Blends the ``XGBoostScorer`` P(planet_candidate) with the Bayesian model's
P(planet_candidate) using a configurable weight.  Falls back gracefully to the
Bayesian-only probability when no XGBoost model is loaded.

Public API
----------
``StackingScorer``
    .predict_proba(features, bayesian_planet_prob) → float
    .predict_proba_batch(features_list, bayesian_planet_probs) → np.ndarray
    .from_model_path(path, xgb_weight) (classmethod)
    .bayesian_only() (classmethod)
    .xgb_weight → float
    .has_xgb → bool
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from exo_toolkit.schemas import CandidateFeatures

if TYPE_CHECKING:
    from exo_toolkit.ml.xgboost_scorer import XGBoostScorer


class StackingScorer:
    """Weighted-average ensemble of XGBoost and Bayesian planet probabilities.

    When no XGBoost scorer is attached (``has_xgb == False``), prediction
    calls return the Bayesian probability unchanged — making it safe to use
    ``StackingScorer`` as the universal scoring interface regardless of
    whether a trained XGBoost model is available.

    Args:
        xgb_scorer: A trained ``XGBoostScorer`` instance, or ``None``.
        xgb_weight: Weight given to the XGBoost probability in the blend
            (Bayesian weight = ``1 - xgb_weight``).  Must be in ``[0, 1]``.

    Example usage
    -------------
    >>> stacker = StackingScorer.from_model_path("model.json", xgb_weight=0.6)
    >>> p = stacker.predict_proba(features, bayesian_planet_prob=0.2)
    """

    def __init__(
        self,
        xgb_scorer: XGBoostScorer | None = None,
        xgb_weight: float = 0.5,
    ) -> None:
        if not 0.0 <= xgb_weight <= 1.0:
            raise ValueError(f"xgb_weight must be in [0, 1], got {xgb_weight}")
        self._xgb_scorer = xgb_scorer
        self._xgb_weight = xgb_weight

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def xgb_weight(self) -> float:
        """Weight given to the XGBoost probability in the blend."""
        return self._xgb_weight

    @property
    def has_xgb(self) -> bool:
        """True if an XGBoost scorer is attached."""
        return self._xgb_scorer is not None

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_proba(
        self,
        features: CandidateFeatures,
        bayesian_planet_prob: float,
    ) -> float:
        """Return blended P(planet_candidate) for a single candidate.

        Args:
            features: A ``CandidateFeatures`` instance.
            bayesian_planet_prob: P(planet_candidate) from the Bayesian model.

        Returns:
            Blended probability in [0, 1].  Equal to ``bayesian_planet_prob``
            when no XGBoost scorer is attached.
        """
        if self._xgb_scorer is None:
            return float(bayesian_planet_prob)
        xgb_prob: float = self._xgb_scorer.predict_proba(features)
        w = self._xgb_weight
        return float(w * xgb_prob + (1.0 - w) * bayesian_planet_prob)

    def predict_proba_batch(
        self,
        features_list: list[CandidateFeatures],
        bayesian_planet_probs: np.ndarray,
    ) -> np.ndarray:
        """Return blended P(planet_candidate) for a batch of candidates.

        Args:
            features_list: List of ``CandidateFeatures`` instances.
            bayesian_planet_probs: 1-D array of Bayesian P(planet) values,
                one per sample.

        Returns:
            1-D ``float64`` array of blended probabilities.
        """
        bayes = np.array(bayesian_planet_probs, dtype=np.float64)
        if self._xgb_scorer is None:
            return bayes
        xgb_probs: np.ndarray = self._xgb_scorer.predict_proba_batch(features_list)
        w = self._xgb_weight
        result: np.ndarray = (w * xgb_probs + (1.0 - w) * bayes).astype(np.float64)
        return result

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
        return cls(xgb_scorer=xgb_scorer, xgb_weight=xgb_weight)

    @classmethod
    def bayesian_only(cls) -> StackingScorer:
        """Return a scorer that passes through the Bayesian probability unchanged.

        Useful as a no-op drop-in when no trained XGBoost model is available.
        """
        return cls(xgb_scorer=None, xgb_weight=0.5)
