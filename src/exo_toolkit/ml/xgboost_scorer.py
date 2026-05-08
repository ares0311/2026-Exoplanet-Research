"""XGBoost Tier-1 ML scorer for exoplanet transit candidates.

Trains a binary XGBoost classifier on the 35 OptScore fields from
``CandidateFeatures`` to distinguish planet candidates from false positives.

Design decisions
----------------
- Missing features (``None``) are passed as ``np.nan``; XGBoost handles them
  natively via its built-in missing-value splitting.
- Labels: 1 = planet candidate, 0 = false positive.
- One-vs-rest: a single ``planet_candidate`` probability is returned; callers
  decide how to blend it with the Bayesian posterior.
- No sklearn dependency: uses xgboost's native Python API (DMatrix + xgb.train).
- Model is serialised to / loaded from a JSON file (XGBoost's portable format).

Public API
----------
``XGBoostScorer``
    .fit(features_list, labels, **xgb_kwargs) → self
    .predict_proba(features) → float          (P(planet_candidate))
    .save(path) / .load(path) (classmethod)
    .feature_names → list[str]

``features_to_array(features) → np.ndarray``
    Convert a ``CandidateFeatures`` instance to a 1-D float array (nan for None).
``features_list_to_matrix(features_list) → np.ndarray``
    Stack multiple instances into a 2-D matrix.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import xgboost as xgb

from exo_toolkit.schemas import CandidateFeatures

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Ordered list of the 35 OptScore field names — defines the column order of
# every feature matrix produced by this module.
FEATURE_NAMES: list[str] = [
    "snr_score",
    "log_snr_score",
    "transit_count_score",
    "depth_consistency_score",
    "duration_consistency_score",
    "duration_plausibility_score",
    "transit_shape_score",
    "data_gap_overlap_score",
    "odd_even_mismatch_score",
    "secondary_eclipse_score",
    "v_shape_score",
    "large_depth_score",
    "companion_radius_too_large_score",
    "duration_implausibility_score",
    "centroid_offset_score",
    "contamination_score",
    "nearby_bright_source_score",
    "aperture_edge_score",
    "dilution_sensitivity_score",
    "stellar_variability_score",
    "variability_periodogram_score",
    "harmonic_score",
    "flare_score",
    "quasi_periodic_score",
    "non_box_shape_score",
    "systematics_overlap_score",
    "quality_flag_score",
    "sector_boundary_score",
    "background_excursion_score",
    "single_event_score",
    "nearby_targets_common_signal_score",
    "known_object_score",
    "target_id_match_score",
    "period_match_score",
    "epoch_match_score",
]

# Verify at import time that FEATURE_NAMES matches the actual schema fields.
_SCHEMA_FIELDS = set(CandidateFeatures.model_fields.keys())
assert set(FEATURE_NAMES) <= _SCHEMA_FIELDS, (
    f"FEATURE_NAMES contains fields not in CandidateFeatures: "
    f"{set(FEATURE_NAMES) - _SCHEMA_FIELDS}"
)

# Default XGBoost booster parameters (native API).
_DEFAULT_XGB_PARAMS: dict[str, Any] = {
    "max_depth": 4,
    "eta": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "seed": 42,
    "verbosity": 0,
}
_DEFAULT_NUM_BOOST_ROUND: int = 100


# ---------------------------------------------------------------------------
# Feature conversion helpers
# ---------------------------------------------------------------------------


def features_to_array(features: CandidateFeatures) -> np.ndarray:
    """Convert a ``CandidateFeatures`` instance to a 1-D float array.

    ``None`` values (diagnostic not run) are encoded as ``np.nan`` so that
    XGBoost's native missing-value handling applies.

    Args:
        features: A ``CandidateFeatures`` instance.

    Returns:
        A 1-D ``float64`` array of length ``len(FEATURE_NAMES)``.
    """
    row = []
    for name in FEATURE_NAMES:
        val = getattr(features, name)
        row.append(float(val) if val is not None else np.nan)
    return np.array(row, dtype=np.float64)


def features_list_to_matrix(features_list: list[CandidateFeatures]) -> np.ndarray:
    """Stack a list of ``CandidateFeatures`` into a 2-D float matrix.

    Args:
        features_list: One or more ``CandidateFeatures`` instances.

    Returns:
        A ``float64`` array of shape ``(n_samples, len(FEATURE_NAMES))``.
    """
    return np.stack([features_to_array(f) for f in features_list], axis=0)


# ---------------------------------------------------------------------------
# Training result
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class TrainingResult:
    """Summary statistics from a completed training run."""

    n_samples: int
    n_positive: int
    n_negative: int
    n_features: int
    best_iteration: int
    feature_importance: dict[str, float]


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


class XGBoostScorer:
    """XGBoost binary classifier for transit candidate vs. false-positive scoring.

    Uses XGBoost's native Python API (``xgb.DMatrix`` + ``xgb.train``).
    Missing features (``None`` in ``CandidateFeatures``) are encoded as
    ``np.nan`` and handled natively by XGBoost.

    Example usage
    -------------
    >>> scorer = XGBoostScorer()
    >>> scorer.fit(train_features, train_labels)
    >>> p = scorer.predict_proba(test_features)  # P(planet candidate)
    >>> scorer.save("model.json")
    >>> loaded = XGBoostScorer.load("model.json")
    """

    def __init__(self) -> None:
        self._model: xgb.Booster | None = None
        self._training_result: TrainingResult | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def feature_names(self) -> list[str]:
        """Ordered list of feature names used by this scorer."""
        return list(FEATURE_NAMES)

    @property
    def is_trained(self) -> bool:
        """True if the model has been trained or loaded."""
        return self._model is not None

    @property
    def training_result(self) -> TrainingResult | None:
        """Training statistics, or None if not yet trained."""
        return self._training_result

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        features_list: list[CandidateFeatures],
        labels: list[int],
        *,
        eval_set: list[tuple[list[CandidateFeatures], list[int]]] | None = None,
        **xgb_kwargs: Any,
    ) -> XGBoostScorer:
        """Train the XGBoost classifier.

        Args:
            features_list: Training samples as ``CandidateFeatures`` instances.
            labels: Binary labels — 1 for planet candidate, 0 for false positive.
            eval_set: Optional validation set as ``[(features_list, labels), ...]``.
                Enables early stopping when used with ``early_stopping_rounds``.
            **xgb_kwargs: Override default XGBoost parameters.  Pass
                ``num_boost_round=N`` to change the number of trees.
                ``early_stopping_rounds=K`` enables early stopping (requires
                ``eval_set``).  All other kwargs override ``_DEFAULT_XGB_PARAMS``.

        Returns:
            self (for method chaining).

        Raises:
            ValueError: If ``features_list`` and ``labels`` have different lengths,
                or if ``labels`` contains values other than 0 or 1.
        """
        if len(features_list) != len(labels):
            raise ValueError(
                f"features_list length ({len(features_list)}) != "
                f"labels length ({len(labels)})"
            )
        if not all(lbl in (0, 1) for lbl in labels):
            raise ValueError("labels must contain only 0 and 1")
        if len(features_list) == 0:
            raise ValueError("features_list must not be empty")

        # Pop native-API-level kwargs before building booster params.
        num_boost_round: int = int(xgb_kwargs.pop("num_boost_round", _DEFAULT_NUM_BOOST_ROUND))
        early_stopping_rounds: int | None = xgb_kwargs.pop("early_stopping_rounds", None)

        X = features_list_to_matrix(features_list)
        y = np.array(labels, dtype=np.int32)

        params = {**_DEFAULT_XGB_PARAMS, **xgb_kwargs}

        dtrain = xgb.DMatrix(X, label=y, missing=np.nan, feature_names=FEATURE_NAMES)

        train_kwargs: dict[str, Any] = {}
        if eval_set is not None:
            watchlist = [
                (
                    xgb.DMatrix(
                        features_list_to_matrix(fl),
                        label=np.array(lbls, dtype=np.int32),
                        missing=np.nan,
                        feature_names=FEATURE_NAMES,
                    ),
                    f"eval_{i}",
                )
                for i, (fl, lbls) in enumerate(eval_set)
            ]
            train_kwargs["evals"] = [(dtrain, "train"), *watchlist]
            train_kwargs["verbose_eval"] = False
            if early_stopping_rounds is not None:
                train_kwargs["early_stopping_rounds"] = early_stopping_rounds

        booster = xgb.train(params, dtrain, num_boost_round=num_boost_round, **train_kwargs)
        self._model = booster

        # Feature importance (gain-based, normalised to sum to 1).
        # cast: xgboost stubs type get_score() as dict[str, float | list[float]];
        # the gain importance_type always yields scalar floats at runtime.
        raw_scores: dict[str, float] = cast(
            "dict[str, float]", booster.get_score(importance_type="gain")
        )
        raw_all: dict[str, float] = {
            name: raw_scores.get(name, 0.0) for name in FEATURE_NAMES
        }
        total = sum(raw_all.values())
        feature_importance = (
            {name: v / total for name, v in raw_all.items()}
            if total > 0
            else dict.fromkeys(FEATURE_NAMES, 0.0)
        )

        used_early_stopping = early_stopping_rounds is not None and eval_set is not None
        best_iter = (
            int(getattr(booster, "best_iteration", num_boost_round - 1))
            if used_early_stopping
            else num_boost_round - 1
        )

        self._training_result = TrainingResult(
            n_samples=len(labels),
            n_positive=int(np.sum(y)),
            n_negative=int(np.sum(1 - y)),
            n_features=len(FEATURE_NAMES),
            best_iteration=best_iter,
            feature_importance=feature_importance,
        )
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_proba(self, features: CandidateFeatures) -> float:
        """Return P(planet_candidate) for a single candidate.

        Args:
            features: A ``CandidateFeatures`` instance.

        Returns:
            Probability in [0, 1] that the signal is a planet candidate.

        Raises:
            RuntimeError: If the scorer has not been trained or loaded.
        """
        if self._model is None:
            raise RuntimeError("Scorer is not trained. Call fit() or load() first.")
        X = features_to_array(features).reshape(1, -1)
        dtest = xgb.DMatrix(X, missing=np.nan, feature_names=FEATURE_NAMES)
        proba: float = float(self._model.predict(dtest)[0])
        return proba

    def predict_proba_batch(
        self, features_list: list[CandidateFeatures]
    ) -> np.ndarray:
        """Return P(planet_candidate) for a list of candidates.

        Args:
            features_list: List of ``CandidateFeatures`` instances.

        Returns:
            1-D ``float64`` array of probabilities, one per sample.

        Raises:
            RuntimeError: If the scorer has not been trained or loaded.
        """
        if self._model is None:
            raise RuntimeError("Scorer is not trained. Call fit() or load() first.")
        X = features_list_to_matrix(features_list)
        dtest = xgb.DMatrix(X, missing=np.nan, feature_names=FEATURE_NAMES)
        return self._model.predict(dtest).astype(np.float64)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save the trained model and metadata to a JSON file.

        Args:
            path: Destination file path (will be created or overwritten).

        Raises:
            RuntimeError: If the scorer has not been trained.
        """
        if self._model is None:
            raise RuntimeError("Cannot save: scorer is not trained.")
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        model_path = p.with_suffix(".xgb.json")
        self._model.save_model(str(model_path))

        meta: dict[str, Any] = {
            "feature_names": FEATURE_NAMES,
            "model_file": model_path.name,
        }
        if self._training_result is not None:
            meta["training_result"] = dataclasses.asdict(self._training_result)

        p.write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> XGBoostScorer:
        """Load a previously saved scorer from a JSON metadata file.

        Args:
            path: Path to the metadata JSON file written by ``save()``.

        Returns:
            A fully initialised ``XGBoostScorer`` ready for prediction.

        Raises:
            FileNotFoundError: If ``path`` or the companion ``.xgb.json`` do
                not exist.
        """
        p = Path(path)
        meta = json.loads(p.read_text())
        model_path = p.parent / meta["model_file"]

        scorer = cls()
        booster = xgb.Booster()
        booster.load_model(str(model_path))
        scorer._model = booster

        if "training_result" in meta:
            tr = meta["training_result"]
            scorer._training_result = TrainingResult(**tr)

        return scorer
