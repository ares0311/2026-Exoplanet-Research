"""Calibration infrastructure for the Bayesian scoring model.

Provides reliability-based calibration (Platt scaling or isotonic regression)
that maps raw posterior probabilities to empirically-grounded calibrated
probabilities.  Also computes standard evaluation metrics: Brier score,
reliability curves, precision/recall, and confusion matrix.

Calibration is fit per-hypothesis (one-vs-rest) and the calibrated
probabilities are renormalized to sum to 1.0 before returning a valid
HypothesisPosterior.

Public API
----------
compute_metrics(labeled_candidates, *, n_bins) → CalibrationMetrics
fit_calibration(labeled_candidates, *, method, n_bins) → CalibrationResult
apply_calibration(posterior, result) → HypothesisPosterior
save_calibration(result, path) → Path
load_calibration(path) → CalibrationResult
"""
from __future__ import annotations

import dataclasses
import datetime
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from scipy.optimize import minimize  # type: ignore[import-untyped]

from exo_toolkit.schemas import HypothesisPosterior, ScoredCandidate

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

Hypothesis = Literal[
    "planet_candidate",
    "eclipsing_binary",
    "background_eclipsing_binary",
    "stellar_variability",
    "instrumental_artifact",
    "known_object",
]

_ALL_HYPOTHESES: tuple[str, ...] = (
    "planet_candidate",
    "eclipsing_binary",
    "background_eclipsing_binary",
    "stellar_variability",
    "instrumental_artifact",
    "known_object",
)

_MIN_FIT_SAMPLES = 5  # minimum per-hypothesis samples needed to fit calibration


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LabeledCandidate:
    """A scored candidate paired with its verified ground-truth hypothesis."""

    candidate: ScoredCandidate
    true_label: Hypothesis


@dataclass(frozen=True)
class ReliabilityCurve:
    """Reliability diagram data for one hypothesis (one-vs-rest).

    A well-calibrated model produces points close to the diagonal
    (mean_predicted ≈ fraction_positive in each bin).
    """

    hypothesis: str
    bin_edges: tuple[float, ...]       # length n_bins + 1
    bin_centers: tuple[float, ...]     # length n_bins
    mean_predicted: tuple[float, ...]  # mean predicted prob per non-empty bin
    fraction_positive: tuple[float, ...]  # empirical frequency per bin
    bin_counts: tuple[int, ...]        # samples per bin (0 = empty)


@dataclass(frozen=True)
class BrierScores:
    """Per-hypothesis and mean Brier scores (lower is better; 0 is perfect)."""

    planet_candidate: float
    eclipsing_binary: float
    background_eclipsing_binary: float
    stellar_variability: float
    instrumental_artifact: float
    known_object: float
    mean: float


@dataclass(frozen=True)
class PrecisionRecall:
    """Precision, recall, and F1 for one hypothesis treated as a binary classifier."""

    hypothesis: str
    precision: float | None  # None when no predictions made for this class
    recall: float | None     # None when no true positives exist
    f1: float | None
    support: int             # number of true examples in this class


@dataclass(frozen=True)
class CalibrationMetrics:
    """Aggregated calibration evaluation metrics for a labeled dataset."""

    n_samples: int
    brier_scores: BrierScores
    reliability_curves: tuple[ReliabilityCurve, ...]
    precision_recall: tuple[PrecisionRecall, ...]
    confusion_labels: tuple[str, ...]          # row/column order
    confusion_matrix: tuple[tuple[int, ...], ...]  # rows=true, cols=predicted


@dataclass(frozen=True)
class PlattParams:
    """Platt scaling parameters: calibrated = sigmoid(slope * raw + intercept)."""

    slope: float
    intercept: float


@dataclass(frozen=True)
class IsotonicKnots:
    """Isotonic regression knot points for piecewise-linear interpolation."""

    x_knots: tuple[float, ...]  # sorted raw probabilities (input)
    y_knots: tuple[float, ...]  # corresponding calibrated probabilities (output)


@dataclass(frozen=True)
class CalibrationResult:
    """A fitted calibration model and its training-set metrics."""

    method: str                              # "platt" or "isotonic"
    n_training_samples: int
    fitted_at: str                           # ISO 8601 UTC
    hypotheses: tuple[str, ...]              # canonical order
    platt_params: tuple[PlattParams, ...] | None    # one per hypothesis
    isotonic_knots: tuple[IsotonicKnots, ...] | None  # one per hypothesis
    training_metrics: CalibrationMetrics


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_metrics(
    labeled_candidates: Sequence[LabeledCandidate],
    *,
    n_bins: int = 10,
) -> CalibrationMetrics:
    """Compute calibration metrics for a labeled candidate set.

    Args:
        labeled_candidates: Sequence of scored candidates with verified labels.
        n_bins: Number of probability bins for reliability diagrams.

    Returns:
        CalibrationMetrics with Brier scores, reliability curves,
        precision/recall, and confusion matrix.

    Raises:
        ValueError: labeled_candidates is empty or n_bins < 2.
    """
    if len(labeled_candidates) == 0:
        raise ValueError("labeled_candidates must not be empty")
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2, got {n_bins}")

    prob_matrix, label_indices = _extract_arrays(labeled_candidates)

    return CalibrationMetrics(
        n_samples=len(labeled_candidates),
        brier_scores=_compute_brier_scores(prob_matrix, label_indices),
        reliability_curves=_compute_reliability_curves(
            prob_matrix, label_indices, n_bins
        ),
        precision_recall=_compute_precision_recall(prob_matrix, label_indices),
        confusion_labels=_ALL_HYPOTHESES,
        confusion_matrix=_compute_confusion_matrix(prob_matrix, label_indices),
    )


def fit_calibration(
    labeled_candidates: Sequence[LabeledCandidate],
    *,
    method: Literal["platt", "isotonic"] = "isotonic",
    n_bins: int = 10,
) -> CalibrationResult:
    """Fit a calibration model from labeled candidates.

    Each hypothesis is calibrated independently (one-vs-rest) then the
    result is renormalized.  Hypotheses with fewer than five training
    samples fall back to the identity mapping (no calibration).

    Args:
        labeled_candidates: Labeled training set.
        method: "platt" for Platt scaling (logistic sigmoid); "isotonic"
            for isotonic regression via PAVA + linear interpolation.
        n_bins: Number of bins for the training reliability diagram.

    Returns:
        CalibrationResult containing the fitted model and training metrics.

    Raises:
        ValueError: labeled_candidates is empty or method is unrecognised.
    """
    if len(labeled_candidates) == 0:
        raise ValueError("labeled_candidates must not be empty")
    if method not in ("platt", "isotonic"):
        raise ValueError(f"method must be 'platt' or 'isotonic', got {method!r}")

    prob_matrix, label_indices = _extract_arrays(labeled_candidates)

    platt_params: tuple[PlattParams, ...] | None = None
    isotonic_knots: tuple[IsotonicKnots, ...] | None = None

    if method == "platt":
        platt_params = tuple(
            _fit_platt(prob_matrix[:, k], (label_indices == k).astype(float))
            for k in range(6)
        )
    else:
        isotonic_knots = tuple(
            _fit_isotonic(prob_matrix[:, k], (label_indices == k).astype(float))
            for k in range(6)
        )

    training_metrics = CalibrationMetrics(
        n_samples=len(labeled_candidates),
        brier_scores=_compute_brier_scores(prob_matrix, label_indices),
        reliability_curves=_compute_reliability_curves(
            prob_matrix, label_indices, n_bins
        ),
        precision_recall=_compute_precision_recall(prob_matrix, label_indices),
        confusion_labels=_ALL_HYPOTHESES,
        confusion_matrix=_compute_confusion_matrix(prob_matrix, label_indices),
    )

    return CalibrationResult(
        method=method,
        n_training_samples=len(labeled_candidates),
        fitted_at=datetime.datetime.now(datetime.UTC).isoformat(),
        hypotheses=_ALL_HYPOTHESES,
        platt_params=platt_params,
        isotonic_knots=isotonic_knots,
        training_metrics=training_metrics,
    )


def apply_calibration(
    posterior: HypothesisPosterior,
    result: CalibrationResult,
) -> HypothesisPosterior:
    """Apply a fitted calibration model to a raw posterior.

    Each hypothesis probability is calibrated independently, then the
    six values are renormalized to sum to 1.0.

    Args:
        posterior: Raw HypothesisPosterior from the scoring pipeline.
        result: Fitted CalibrationResult from fit_calibration().

    Returns:
        Calibrated HypothesisPosterior with probabilities summing to 1.0.

    Raises:
        ValueError: All calibrated probabilities collapse to zero.
    """
    raw = _posterior_to_array(posterior)

    if result.method == "platt" and result.platt_params is not None:
        calibrated = np.array(
            [_apply_platt(raw[k], result.platt_params[k]) for k in range(6)]
        )
    elif result.method == "isotonic" and result.isotonic_knots is not None:
        calibrated = np.array(
            [_apply_isotonic(raw[k], result.isotonic_knots[k]) for k in range(6)]
        )
    else:
        calibrated = raw.copy()

    total = float(calibrated.sum())
    if total <= 0.0:
        raise ValueError(
            "Calibration collapsed all posterior probabilities to zero; "
            "check the calibration model."
        )

    calibrated /= total
    return _array_to_posterior(calibrated)


# ---------------------------------------------------------------------------
# Internal helpers — data extraction
# ---------------------------------------------------------------------------


def _extract_arrays(
    labeled_candidates: Sequence[LabeledCandidate],
) -> tuple[np.ndarray, np.ndarray]:
    """Return (prob_matrix, label_indices) for a labeled set.

    prob_matrix: shape (n, 6), raw posterior probabilities in _ALL_HYPOTHESES order.
    label_indices: shape (n,), integer index of the true hypothesis.
    """
    n = len(labeled_candidates)
    prob_matrix = np.empty((n, 6), dtype=float)
    label_indices = np.empty(n, dtype=int)

    for i, lc in enumerate(labeled_candidates):
        prob_matrix[i] = _posterior_to_array(lc.candidate.posterior)
        label_indices[i] = _ALL_HYPOTHESES.index(lc.true_label)

    return prob_matrix, label_indices


def _posterior_to_array(posterior: HypothesisPosterior) -> np.ndarray:
    return np.array([
        posterior.planet_candidate,
        posterior.eclipsing_binary,
        posterior.background_eclipsing_binary,
        posterior.stellar_variability,
        posterior.instrumental_artifact,
        posterior.known_object,
    ], dtype=float)


def _array_to_posterior(arr: np.ndarray) -> HypothesisPosterior:
    """Normalize arr to sum to 1.0 and construct a HypothesisPosterior."""
    arr = np.clip(arr, 0.0, 1.0)
    total = arr.sum()
    if total > 0.0:
        arr = arr / total
    # Nudge to ensure sum is exactly 1.0 within Pydantic tolerance
    arr[-1] += 1.0 - arr.sum()
    arr = np.clip(arr, 0.0, 1.0)
    return HypothesisPosterior(
        planet_candidate=float(arr[0]),
        eclipsing_binary=float(arr[1]),
        background_eclipsing_binary=float(arr[2]),
        stellar_variability=float(arr[3]),
        instrumental_artifact=float(arr[4]),
        known_object=float(arr[5]),
    )


# ---------------------------------------------------------------------------
# Internal helpers — metrics
# ---------------------------------------------------------------------------


def _compute_brier_scores(
    prob_matrix: np.ndarray,
    label_indices: np.ndarray,
) -> BrierScores:
    n = len(label_indices)
    if n == 0:
        return BrierScores(*(0.0 for _ in range(7)))

    per_hyp: list[float] = []
    for k in range(6):
        binary = (label_indices == k).astype(float)
        per_hyp.append(float(np.mean((prob_matrix[:, k] - binary) ** 2)))

    return BrierScores(
        planet_candidate=per_hyp[0],
        eclipsing_binary=per_hyp[1],
        background_eclipsing_binary=per_hyp[2],
        stellar_variability=per_hyp[3],
        instrumental_artifact=per_hyp[4],
        known_object=per_hyp[5],
        mean=float(np.mean(per_hyp)),
    )


def _compute_reliability_curves(
    prob_matrix: np.ndarray,
    label_indices: np.ndarray,
    n_bins: int,
) -> tuple[ReliabilityCurve, ...]:
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    curves: list[ReliabilityCurve] = []

    for k, hyp in enumerate(_ALL_HYPOTHESES):
        probs_k = prob_matrix[:, k]
        binary_k = (label_indices == k).astype(float)

        mean_pred = np.zeros(n_bins, dtype=float)
        frac_pos = np.zeros(n_bins, dtype=float)
        counts = np.zeros(n_bins, dtype=int)

        for b in range(n_bins):
            lo, hi = float(bin_edges[b]), float(bin_edges[b + 1])
            if b < n_bins - 1:
                mask = (probs_k >= lo) & (probs_k < hi)
            else:
                mask = (probs_k >= lo) & (probs_k <= hi)
            c = int(mask.sum())
            counts[b] = c
            if c > 0:
                mean_pred[b] = float(np.mean(probs_k[mask]))
                frac_pos[b] = float(np.mean(binary_k[mask]))
            else:
                mean_pred[b] = bin_centers[b]  # placeholder for empty bin

        curves.append(ReliabilityCurve(
            hypothesis=hyp,
            bin_edges=tuple(float(e) for e in bin_edges),
            bin_centers=tuple(float(c) for c in bin_centers),
            mean_predicted=tuple(float(p) for p in mean_pred),
            fraction_positive=tuple(float(f) for f in frac_pos),
            bin_counts=tuple(int(c) for c in counts),
        ))

    return tuple(curves)


def _compute_precision_recall(
    prob_matrix: np.ndarray,
    label_indices: np.ndarray,
) -> tuple[PrecisionRecall, ...]:
    predicted = np.argmax(prob_matrix, axis=1)
    results: list[PrecisionRecall] = []

    for k in range(6):
        tp = int(((predicted == k) & (label_indices == k)).sum())
        fp = int(((predicted == k) & (label_indices != k)).sum())
        fn = int(((predicted != k) & (label_indices == k)).sum())
        support = int((label_indices == k).sum())

        prec: float | None = float(tp / (tp + fp)) if (tp + fp) > 0 else None
        rec: float | None = float(tp / (tp + fn)) if (tp + fn) > 0 else None
        f1: float | None = None
        if prec is not None and rec is not None and (prec + rec) > 0.0:
            f1 = float(2.0 * prec * rec / (prec + rec))

        results.append(PrecisionRecall(
            hypothesis=_ALL_HYPOTHESES[k],
            precision=prec,
            recall=rec,
            f1=f1,
            support=support,
        ))

    return tuple(results)


def _compute_confusion_matrix(
    prob_matrix: np.ndarray,
    label_indices: np.ndarray,
) -> tuple[tuple[int, ...], ...]:
    predicted = np.argmax(prob_matrix, axis=1)
    cm = np.zeros((6, 6), dtype=int)
    for true, pred in zip(label_indices, predicted, strict=True):
        cm[true, pred] += 1
    return tuple(tuple(int(v) for v in row) for row in cm)


# ---------------------------------------------------------------------------
# Internal helpers — calibration fitting
# ---------------------------------------------------------------------------


def _sigmoid(x: Any) -> np.ndarray:
    return np.asarray(1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float))), dtype=float)


def _pava(y: np.ndarray) -> np.ndarray:
    """Pool Adjacent Violators Algorithm: returns non-decreasing fitted values."""
    if len(y) <= 1:
        return y.copy().astype(float)

    # Each block stored as (sum_y, count)
    blocks: list[list[float]] = [[float(v)] for v in y]

    i = 0
    while i < len(blocks) - 1:
        mean_i = sum(blocks[i]) / len(blocks[i])
        mean_next = sum(blocks[i + 1]) / len(blocks[i + 1])
        if mean_i > mean_next:
            blocks[i] = blocks[i] + blocks[i + 1]
            blocks.pop(i + 1)
            if i > 0:
                i -= 1
        else:
            i += 1

    result = np.empty(len(y), dtype=float)
    pos = 0
    for block in blocks:
        val = sum(block) / len(block)
        for _ in block:
            result[pos] = val
            pos += 1
    return result


def _fit_platt(probs: np.ndarray, labels: np.ndarray) -> PlattParams:
    """Fit Platt scaling via MLE.  Falls back to identity if data are insufficient."""
    n_pos = int(labels.sum())
    if len(probs) < _MIN_FIT_SAMPLES or n_pos == 0 or n_pos == len(labels):
        return PlattParams(slope=1.0, intercept=0.0)

    def neg_log_loss(params: np.ndarray) -> float:
        a, b = float(params[0]), float(params[1])
        p = _sigmoid(a * probs + b)
        p = np.clip(p, 1e-10, 1.0 - 1e-10)
        return float(-np.sum(labels * np.log(p) + (1.0 - labels) * np.log(1.0 - p)))

    try:
        res = minimize(
            neg_log_loss,
            x0=np.array([1.0, 0.0]),
            method="Nelder-Mead",
            options={"maxiter": 2000, "xatol": 1e-7, "fatol": 1e-7},
        )
        return PlattParams(slope=float(res.x[0]), intercept=float(res.x[1]))
    except Exception:
        return PlattParams(slope=1.0, intercept=0.0)


def _fit_isotonic(probs: np.ndarray, labels: np.ndarray) -> IsotonicKnots:
    """Fit isotonic regression (PAVA). Falls back to identity if data insufficient."""
    n_pos = int(labels.sum())
    if len(probs) < _MIN_FIT_SAMPLES or n_pos == 0 or n_pos == len(labels):
        return IsotonicKnots(x_knots=(0.0, 1.0), y_knots=(0.0, 1.0))

    order = np.argsort(probs)
    x = probs[order]
    y = labels[order].astype(float)
    y_fit = _pava(y)

    # Deduplicate: for equal x values, average fitted y
    unique_x: list[float] = []
    unique_y: list[float] = []
    i = 0
    while i < len(x):
        j = i
        while j < len(x) and x[j] == x[i]:
            j += 1
        unique_x.append(float(np.mean(x[i:j])))
        unique_y.append(float(np.mean(y_fit[i:j])))
        i = j

    # Ensure boundary coverage
    xs = np.array(unique_x)
    ys = np.array(unique_y)
    if xs[0] > 0.0:
        xs = np.concatenate([[0.0], xs])
        ys = np.concatenate([[ys[0]], ys])
    if xs[-1] < 1.0:
        xs = np.concatenate([xs, [1.0]])
        ys = np.concatenate([ys, [ys[-1]]])

    return IsotonicKnots(
        x_knots=tuple(float(v) for v in xs),
        y_knots=tuple(float(v) for v in ys),
    )


def _apply_platt(prob: float, params: PlattParams) -> float:
    return float(_sigmoid(params.slope * prob + params.intercept))


def _apply_isotonic(prob: float, knots: IsotonicKnots) -> float:
    return float(np.interp(prob, knots.x_knots, knots.y_knots))


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def save_calibration(result: CalibrationResult, path: Path | str) -> Path:
    """Serialize a CalibrationResult to JSON.

    Args:
        result: Fitted model from :func:`fit_calibration`.
        path: Destination JSON file.

    Returns:
        Path of the written file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dataclasses.asdict(result), indent=2))
    return path


def _dict_to_calibration_result(d: dict[str, Any]) -> CalibrationResult:
    platt: tuple[PlattParams, ...] | None = None
    if d.get("platt_params") is not None:
        platt = tuple(PlattParams(**p) for p in d["platt_params"])

    isotonic: tuple[IsotonicKnots, ...] | None = None
    if d.get("isotonic_knots") is not None:
        isotonic = tuple(
            IsotonicKnots(
                x_knots=tuple(k["x_knots"]),
                y_knots=tuple(k["y_knots"]),
            )
            for k in d["isotonic_knots"]
        )

    m = d["training_metrics"]
    bs = BrierScores(**m["brier_scores"])
    rcs = tuple(
        ReliabilityCurve(
            hypothesis=rc["hypothesis"],
            bin_edges=tuple(rc["bin_edges"]),
            bin_centers=tuple(rc["bin_centers"]),
            mean_predicted=tuple(rc["mean_predicted"]),
            fraction_positive=tuple(rc["fraction_positive"]),
            bin_counts=tuple(int(c) for c in rc["bin_counts"]),
        )
        for rc in m["reliability_curves"]
    )
    prs = tuple(PrecisionRecall(**pr) for pr in m["precision_recall"])
    metrics = CalibrationMetrics(
        n_samples=m["n_samples"],
        brier_scores=bs,
        reliability_curves=rcs,
        precision_recall=prs,
        confusion_labels=tuple(m["confusion_labels"]),
        confusion_matrix=tuple(tuple(int(v) for v in row) for row in m["confusion_matrix"]),
    )
    return CalibrationResult(
        method=d["method"],
        n_training_samples=d["n_training_samples"],
        fitted_at=d["fitted_at"],
        hypotheses=tuple(d["hypotheses"]),
        platt_params=platt,
        isotonic_knots=isotonic,
        training_metrics=metrics,
    )


def load_calibration(path: Path | str) -> CalibrationResult:
    """Load a CalibrationResult from a JSON file.

    Args:
        path: File written by :func:`save_calibration`.

    Returns:
        :class:`CalibrationResult`.
    """
    data = json.loads(Path(path).read_text())
    return _dict_to_calibration_result(data)
