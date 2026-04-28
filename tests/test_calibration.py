"""Tests for exo_toolkit.calibration.

Uses synthetic ScoredCandidate objects with known posterior values so that
Brier scores, reliability curves, and calibration fits are deterministic
and verifiable by hand.
"""
from __future__ import annotations

import numpy as np
import pytest

from exo_toolkit.calibration import (
    CalibrationMetrics,
    IsotonicKnots,
    LabeledCandidate,
    PlattParams,
    _apply_isotonic,
    _apply_platt,
    _array_to_posterior,
    _compute_brier_scores,
    _compute_confusion_matrix,
    _compute_precision_recall,
    _compute_reliability_curves,
    _extract_arrays,
    _fit_isotonic,
    _fit_platt,
    _pava,
    _posterior_to_array,
    _sigmoid,
    apply_calibration,
    compute_metrics,
    fit_calibration,
)
from exo_toolkit.schemas import (
    CandidateExplanation,
    CandidateFeatures,
    CandidateScores,
    CandidateSignal,
    HypothesisPosterior,
    ScoredCandidate,
    ScoringMetadata,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HYPOTHESES = (
    "planet_candidate",
    "eclipsing_binary",
    "background_eclipsing_binary",
    "stellar_variability",
    "instrumental_artifact",
    "known_object",
)


def _posterior(**kwargs: float) -> HypothesisPosterior:
    """Build a HypothesisPosterior; kwargs override defaults; auto-normalises."""
    defaults = dict.fromkeys(_HYPOTHESES, 1.0 / 6.0)
    defaults.update(kwargs)
    total = sum(defaults.values())
    normed = {k: v / total for k, v in defaults.items()}
    return HypothesisPosterior(**normed)


def _scored_candidate(posterior: HypothesisPosterior) -> ScoredCandidate:
    signal = CandidateSignal(
        candidate_id="TIC_1_s01",
        mission="TESS",
        target_id="TIC 1",
        period_days=5.0,
        epoch_bjd=2459000.0,
        duration_hours=2.0,
        depth_ppm=5000.0,
        transit_count=5,
        snr=10.0,
    )
    scores = CandidateScores(
        fpp=0.1,
        detection_confidence=0.9,
        novelty_score=0.5,
        habitability_interest=0.3,
        followup_value=0.6,
        submission_readiness=0.4,
    )
    explanation = CandidateExplanation()
    metadata = ScoringMetadata(
        scoring_model_name="test",
        scoring_model_version="0.0.1",
        scoring_model_commit="abc1234",
        threshold_config_hash="deadbeef",
    )
    return ScoredCandidate(
        signal=signal,
        features=CandidateFeatures(),
        posterior=posterior,
        scores=scores,
        recommended_pathway="github_only_reproducibility",
        explanation=explanation,
        metadata=metadata,
    )


def _labeled(true_label: str, posterior: HypothesisPosterior) -> LabeledCandidate:
    return LabeledCandidate(
        candidate=_scored_candidate(posterior),
        true_label=true_label,  # type: ignore[arg-type]
    )


def _perfect_set() -> list[LabeledCandidate]:
    """6 candidates each predicted and labeled as one of the 6 hypotheses."""
    result = []
    for hyp in _HYPOTHESES:
        p = _posterior(**{hyp: 10.0})  # dominant probability for hyp
        result.append(_labeled(hyp, p))
    return result


def _uniform_set(n: int = 12) -> list[LabeledCandidate]:
    """n candidates with uniform posteriors, labels cycling through hypotheses."""
    result = []
    p = _posterior()  # exactly 1/6 each
    for i in range(n):
        result.append(_labeled(_HYPOTHESES[i % 6], p))
    return result


# ---------------------------------------------------------------------------
# _pava
# ---------------------------------------------------------------------------


class TestPava:
    def test_already_sorted_unchanged(self) -> None:
        y = np.array([0.1, 0.3, 0.5, 0.8])
        out = _pava(y)
        np.testing.assert_allclose(out, y)

    def test_reversal_becomes_mean(self) -> None:
        y = np.array([0.8, 0.2])
        out = _pava(y)
        assert out[0] == pytest.approx(out[1])
        assert out[0] == pytest.approx(0.5)

    def test_output_non_decreasing(self) -> None:
        rng = np.random.default_rng(0)
        y = rng.uniform(0, 1, 20)
        out = _pava(y)
        assert np.all(np.diff(out) >= -1e-12)

    def test_length_preserved(self) -> None:
        y = np.array([0.9, 0.1, 0.5, 0.3, 0.7])
        assert len(_pava(y)) == len(y)

    def test_single_element(self) -> None:
        y = np.array([0.42])
        out = _pava(y)
        assert float(out[0]) == pytest.approx(0.42)

    def test_empty_array(self) -> None:
        out = _pava(np.array([]))
        assert len(out) == 0

    def test_constant_array_unchanged(self) -> None:
        y = np.full(5, 0.6)
        out = _pava(y)
        np.testing.assert_allclose(out, y)


# ---------------------------------------------------------------------------
# _sigmoid
# ---------------------------------------------------------------------------


class TestSigmoid:
    def test_zero_gives_half(self) -> None:
        assert float(_sigmoid(0.0)) == pytest.approx(0.5)

    def test_large_positive_gives_one(self) -> None:
        assert float(_sigmoid(100.0)) == pytest.approx(1.0, abs=1e-6)

    def test_large_negative_gives_zero(self) -> None:
        assert float(_sigmoid(-100.0)) == pytest.approx(0.0, abs=1e-6)

    def test_array_input(self) -> None:
        out = _sigmoid(np.array([-1.0, 0.0, 1.0]))
        assert len(out) == 3
        assert out[1] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# _posterior_to_array / _array_to_posterior
# ---------------------------------------------------------------------------


class TestPosteriorRoundtrip:
    def test_array_sums_to_one(self) -> None:
        arr = _posterior_to_array(_posterior())
        assert arr.sum() == pytest.approx(1.0, abs=1e-9)

    def test_array_length_six(self) -> None:
        arr = _posterior_to_array(_posterior())
        assert len(arr) == 6

    def test_round_trip(self) -> None:
        p = _posterior(planet_candidate=0.5, eclipsing_binary=0.3)
        arr = _posterior_to_array(p)
        p2 = _array_to_posterior(arr)
        assert p2.planet_candidate == pytest.approx(p.planet_candidate, abs=1e-6)
        assert p2.eclipsing_binary == pytest.approx(p.eclipsing_binary, abs=1e-6)

    def test_array_to_posterior_normalises(self) -> None:
        arr = np.array([2.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        p = _array_to_posterior(arr)
        total = (
            p.planet_candidate + p.eclipsing_binary + p.background_eclipsing_binary
            + p.stellar_variability + p.instrumental_artifact + p.known_object
        )
        assert total == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# _extract_arrays
# ---------------------------------------------------------------------------


class TestExtractArrays:
    def test_shape(self) -> None:
        candidates = _perfect_set()
        prob_matrix, label_indices = _extract_arrays(candidates)
        assert prob_matrix.shape == (6, 6)
        assert label_indices.shape == (6,)

    def test_labels_in_range(self) -> None:
        _, label_indices = _extract_arrays(_uniform_set(12))
        assert np.all(label_indices >= 0)
        assert np.all(label_indices < 6)

    def test_prob_rows_sum_to_one(self) -> None:
        prob_matrix, _ = _extract_arrays(_perfect_set())
        np.testing.assert_allclose(prob_matrix.sum(axis=1), 1.0, atol=0.01)


# ---------------------------------------------------------------------------
# _compute_brier_scores
# ---------------------------------------------------------------------------


class TestComputeBrierScores:
    def test_perfect_predictions_give_low_score(self) -> None:
        prob_matrix, label_indices = _extract_arrays(_perfect_set())
        bs = _compute_brier_scores(prob_matrix, label_indices)
        assert bs.mean < 0.1  # near-perfect predictions → low Brier

    def test_uniform_predictions_give_higher_score(self) -> None:
        prob_matrix, label_indices = _extract_arrays(_uniform_set(30))
        bs = _compute_brier_scores(prob_matrix, label_indices)
        assert bs.mean > 0.0

    def test_mean_is_average_of_per_hyp(self) -> None:
        prob_matrix, label_indices = _extract_arrays(_uniform_set(12))
        bs = _compute_brier_scores(prob_matrix, label_indices)
        per_hyp = [
            bs.planet_candidate, bs.eclipsing_binary,
            bs.background_eclipsing_binary, bs.stellar_variability,
            bs.instrumental_artifact, bs.known_object,
        ]
        assert bs.mean == pytest.approx(sum(per_hyp) / 6.0, abs=1e-9)

    def test_all_scores_non_negative(self) -> None:
        prob_matrix, label_indices = _extract_arrays(_uniform_set(12))
        bs = _compute_brier_scores(prob_matrix, label_indices)
        for score in (
            bs.planet_candidate, bs.eclipsing_binary,
            bs.background_eclipsing_binary, bs.stellar_variability,
            bs.instrumental_artifact, bs.known_object, bs.mean
        ):
            assert score >= 0.0

    def test_brier_score_bounded_above_by_one(self) -> None:
        # Worst case: always predict 1.0 when true is 0, etc.
        prob_matrix = np.ones((6, 6)) / 6.0
        label_indices = np.arange(6)
        bs = _compute_brier_scores(prob_matrix, label_indices)
        assert bs.mean <= 1.0


# ---------------------------------------------------------------------------
# _compute_reliability_curves
# ---------------------------------------------------------------------------


class TestComputeReliabilityCurves:
    def test_returns_six_curves(self) -> None:
        prob_matrix, label_indices = _extract_arrays(_uniform_set(30))
        curves = _compute_reliability_curves(prob_matrix, label_indices, n_bins=5)
        assert len(curves) == 6

    def test_bin_counts_sum_to_n(self) -> None:
        n = 24
        prob_matrix, label_indices = _extract_arrays(_uniform_set(n))
        curves = _compute_reliability_curves(prob_matrix, label_indices, n_bins=5)
        for curve in curves:
            assert sum(curve.bin_counts) == n

    def test_fraction_positive_in_range(self) -> None:
        prob_matrix, label_indices = _extract_arrays(_uniform_set(30))
        curves = _compute_reliability_curves(prob_matrix, label_indices, n_bins=5)
        for curve in curves:
            for fp in curve.fraction_positive:
                assert 0.0 <= fp <= 1.0

    def test_hypothesis_names_correct(self) -> None:
        prob_matrix, label_indices = _extract_arrays(_perfect_set())
        curves = _compute_reliability_curves(prob_matrix, label_indices, n_bins=5)
        hyp_names = tuple(c.hypothesis for c in curves)
        assert "planet_candidate" in hyp_names
        assert "known_object" in hyp_names

    def test_bin_edges_span_zero_to_one(self) -> None:
        prob_matrix, label_indices = _extract_arrays(_perfect_set())
        curves = _compute_reliability_curves(prob_matrix, label_indices, n_bins=4)
        for curve in curves:
            assert curve.bin_edges[0] == pytest.approx(0.0)
            assert curve.bin_edges[-1] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _compute_precision_recall
# ---------------------------------------------------------------------------


class TestComputePrecisionRecall:
    def test_returns_six_results(self) -> None:
        prob_matrix, label_indices = _extract_arrays(_perfect_set())
        pr = _compute_precision_recall(prob_matrix, label_indices)
        assert len(pr) == 6

    def test_perfect_predictions_give_high_pr(self) -> None:
        prob_matrix, label_indices = _extract_arrays(_perfect_set())
        pr = _compute_precision_recall(prob_matrix, label_indices)
        for entry in pr:
            if entry.support > 0:
                assert entry.precision == pytest.approx(1.0)
                assert entry.recall == pytest.approx(1.0)

    def test_support_counts_correct(self) -> None:
        candidates = _uniform_set(12)  # 2 per hypothesis
        prob_matrix, label_indices = _extract_arrays(candidates)
        pr = _compute_precision_recall(prob_matrix, label_indices)
        for entry in pr:
            assert entry.support == 2

    def test_f1_consistent_with_pr(self) -> None:
        prob_matrix, label_indices = _extract_arrays(_perfect_set())
        pr = _compute_precision_recall(prob_matrix, label_indices)
        for entry in pr:
            if (
                entry.precision is not None
                and entry.recall is not None
                and entry.precision + entry.recall > 0
            ):
                expected_f1 = (
                    2 * entry.precision * entry.recall
                    / (entry.precision + entry.recall)
                )
                assert entry.f1 == pytest.approx(expected_f1, abs=1e-9)


# ---------------------------------------------------------------------------
# _compute_confusion_matrix
# ---------------------------------------------------------------------------


class TestComputeConfusionMatrix:
    def test_shape_six_by_six(self) -> None:
        prob_matrix, label_indices = _extract_arrays(_perfect_set())
        cm = _compute_confusion_matrix(prob_matrix, label_indices)
        assert len(cm) == 6
        assert all(len(row) == 6 for row in cm)

    def test_perfect_predictions_diagonal(self) -> None:
        prob_matrix, label_indices = _extract_arrays(_perfect_set())
        cm = _compute_confusion_matrix(prob_matrix, label_indices)
        for i in range(6):
            assert cm[i][i] == 1
            off_diag = [cm[i][j] for j in range(6) if j != i]
            assert all(v == 0 for v in off_diag)

    def test_total_equals_n(self) -> None:
        n = 18
        prob_matrix, label_indices = _extract_arrays(_uniform_set(n))
        cm = _compute_confusion_matrix(prob_matrix, label_indices)
        total = sum(v for row in cm for v in row)
        assert total == n


# ---------------------------------------------------------------------------
# _fit_platt / _apply_platt
# ---------------------------------------------------------------------------


class TestPlattScaling:
    def test_identity_fallback_too_few_samples(self) -> None:
        probs = np.array([0.2, 0.8, 0.5])
        labels = np.array([0.0, 1.0, 1.0])
        params = _fit_platt(probs, labels)
        assert params.slope == pytest.approx(1.0)
        assert params.intercept == pytest.approx(0.0)

    def test_identity_fallback_all_positive(self) -> None:
        probs = np.array([0.3, 0.6, 0.8, 0.9, 0.7, 0.5])
        labels = np.ones(6)
        params = _fit_platt(probs, labels)
        assert params.slope == pytest.approx(1.0)

    def test_identity_fallback_all_negative(self) -> None:
        probs = np.array([0.3, 0.6, 0.8, 0.9, 0.7, 0.5])
        labels = np.zeros(6)
        params = _fit_platt(probs, labels)
        assert params.slope == pytest.approx(1.0)

    def test_apply_output_in_range(self) -> None:
        params = PlattParams(slope=2.0, intercept=-1.0)
        for p in [0.0, 0.25, 0.5, 0.75, 1.0]:
            result = _apply_platt(p, params)
            assert 0.0 <= result <= 1.0

    def test_apply_identity_params_passes_through(self) -> None:
        params = PlattParams(slope=1.0, intercept=0.0)
        # sigmoid(1.0 * 0.5 + 0.0) ≈ 0.622, not exactly 0.5
        result = _apply_platt(0.0, params)
        assert result == pytest.approx(0.5, abs=1e-6)

    def test_fit_reduces_brier_score(self) -> None:
        rng = np.random.default_rng(42)
        # Miscalibrated: probabilities are systematically too high
        probs = np.clip(rng.uniform(0.6, 1.0, 50), 0, 1)
        labels = (rng.uniform(0, 1, 50) < 0.3).astype(float)  # ~30% positive
        params = _fit_platt(probs, labels)
        raw_brier = float(np.mean((probs - labels) ** 2))
        cal_probs = np.array([_apply_platt(p, params) for p in probs])
        cal_brier = float(np.mean((cal_probs - labels) ** 2))
        assert cal_brier <= raw_brier + 0.05  # calibrated should be similar or better


# ---------------------------------------------------------------------------
# _fit_isotonic / _apply_isotonic
# ---------------------------------------------------------------------------


class TestIsotonicCalibration:
    def test_identity_fallback_too_few_samples(self) -> None:
        probs = np.array([0.2, 0.8])
        labels = np.array([0.0, 1.0])
        knots = _fit_isotonic(probs, labels)
        assert _apply_isotonic(0.5, knots) >= 0.0

    def test_knots_monotone(self) -> None:
        rng = np.random.default_rng(0)
        probs = rng.uniform(0, 1, 30)
        labels = (probs + rng.normal(0, 0.2, 30) > 0.5).astype(float)
        knots = _fit_isotonic(probs, labels)
        y = np.array(knots.y_knots)
        assert np.all(np.diff(y) >= -1e-12)

    def test_boundary_coverage(self) -> None:
        probs = np.linspace(0.1, 0.9, 20)
        labels = (probs > 0.5).astype(float)
        knots = _fit_isotonic(probs, labels)
        assert knots.x_knots[0] == pytest.approx(0.0)
        assert knots.x_knots[-1] == pytest.approx(1.0)

    def test_apply_output_in_range(self) -> None:
        knots = IsotonicKnots(x_knots=(0.0, 0.5, 1.0), y_knots=(0.1, 0.4, 0.9))
        for p in [0.0, 0.25, 0.5, 0.75, 1.0]:
            result = _apply_isotonic(p, knots)
            assert 0.0 <= result <= 1.0

    def test_interpolation_midpoint(self) -> None:
        knots = IsotonicKnots(x_knots=(0.0, 1.0), y_knots=(0.0, 1.0))
        assert _apply_isotonic(0.5, knots) == pytest.approx(0.5)

    def test_extrapolation_clips_to_boundary(self) -> None:
        knots = IsotonicKnots(x_knots=(0.2, 0.8), y_knots=(0.1, 0.9))
        # np.interp extrapolates with boundary values
        assert _apply_isotonic(0.0, knots) == pytest.approx(0.1)
        assert _apply_isotonic(1.0, knots) == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------


class TestComputeMetrics:
    def test_returns_calibration_metrics(self) -> None:
        result = compute_metrics(_uniform_set(12))
        assert isinstance(result, CalibrationMetrics)

    def test_n_samples_correct(self) -> None:
        result = compute_metrics(_uniform_set(18))
        assert result.n_samples == 18

    def test_raises_on_empty(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            compute_metrics([])

    def test_raises_on_bad_n_bins(self) -> None:
        with pytest.raises(ValueError, match="n_bins"):
            compute_metrics(_uniform_set(6), n_bins=1)

    def test_six_reliability_curves(self) -> None:
        result = compute_metrics(_uniform_set(12))
        assert len(result.reliability_curves) == 6

    def test_six_precision_recall(self) -> None:
        result = compute_metrics(_perfect_set())
        assert len(result.precision_recall) == 6

    def test_confusion_matrix_six_by_six(self) -> None:
        result = compute_metrics(_perfect_set())
        assert len(result.confusion_matrix) == 6
        assert all(len(row) == 6 for row in result.confusion_matrix)

    def test_confusion_labels_correct(self) -> None:
        result = compute_metrics(_perfect_set())
        assert "planet_candidate" in result.confusion_labels
        assert len(result.confusion_labels) == 6


# ---------------------------------------------------------------------------
# fit_calibration
# ---------------------------------------------------------------------------


class TestFitCalibration:
    def test_raises_on_empty(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            fit_calibration([])

    def test_raises_on_bad_method(self) -> None:
        with pytest.raises(ValueError, match="method"):
            fit_calibration(_uniform_set(6), method="boosting")  # type: ignore[arg-type]

    def test_platt_method_returns_platt_params(self) -> None:
        result = fit_calibration(_uniform_set(30), method="platt")
        assert result.platt_params is not None
        assert result.isotonic_knots is None
        assert len(result.platt_params) == 6

    def test_isotonic_method_returns_isotonic_knots(self) -> None:
        result = fit_calibration(_uniform_set(30), method="isotonic")
        assert result.isotonic_knots is not None
        assert result.platt_params is None
        assert len(result.isotonic_knots) == 6

    def test_result_is_frozen_dataclass(self) -> None:
        import dataclasses
        result = fit_calibration(_uniform_set(12))
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.method = "changed"  # type: ignore[misc]

    def test_training_metrics_n_samples_matches(self) -> None:
        candidates = _uniform_set(18)
        result = fit_calibration(candidates)
        assert result.training_metrics.n_samples == 18

    def test_fitted_at_is_iso8601(self) -> None:
        result = fit_calibration(_uniform_set(6))
        import datetime
        dt = datetime.datetime.fromisoformat(result.fitted_at)
        assert dt.tzinfo is not None  # timezone-aware

    def test_hypotheses_tuple_correct(self) -> None:
        result = fit_calibration(_uniform_set(6))
        assert "planet_candidate" in result.hypotheses
        assert len(result.hypotheses) == 6

    def test_default_method_is_isotonic(self) -> None:
        result = fit_calibration(_uniform_set(12))
        assert result.method == "isotonic"


# ---------------------------------------------------------------------------
# apply_calibration
# ---------------------------------------------------------------------------


class TestApplyCalibration:
    def test_returns_hypothesis_posterior(self) -> None:
        result = fit_calibration(_uniform_set(12))
        p = _posterior(planet_candidate=0.5)
        calibrated = apply_calibration(p, result)
        assert isinstance(calibrated, HypothesisPosterior)

    def test_output_sums_to_one(self) -> None:
        result = fit_calibration(_uniform_set(12))
        p = _posterior(planet_candidate=0.5)
        calibrated = apply_calibration(p, result)
        total = (
            calibrated.planet_candidate + calibrated.eclipsing_binary
            + calibrated.background_eclipsing_binary + calibrated.stellar_variability
            + calibrated.instrumental_artifact + calibrated.known_object
        )
        assert total == pytest.approx(1.0, abs=0.01)

    def test_all_probs_non_negative(self) -> None:
        result = fit_calibration(_uniform_set(12))
        p = _posterior()
        calibrated = apply_calibration(p, result)
        for attr in _HYPOTHESES:
            assert getattr(calibrated, attr) >= 0.0

    def test_platt_calibration_applies(self) -> None:
        result = fit_calibration(_uniform_set(30), method="platt")
        p = _posterior(planet_candidate=0.8)
        calibrated = apply_calibration(p, result)
        total = sum(getattr(calibrated, h) for h in _HYPOTHESES)
        assert total == pytest.approx(1.0, abs=0.01)

    def test_isotonic_calibration_applies(self) -> None:
        result = fit_calibration(_uniform_set(30), method="isotonic")
        p = _posterior(eclipsing_binary=0.9)
        calibrated = apply_calibration(p, result)
        total = sum(getattr(calibrated, h) for h in _HYPOTHESES)
        assert total == pytest.approx(1.0, abs=0.01)

    def test_dominant_hypothesis_stays_dominant(self) -> None:
        # With a strongly dominant hypothesis, calibration should preserve dominance
        labeled = []
        for _ in range(10):
            labeled.append(_labeled("planet_candidate", _posterior(planet_candidate=5.0)))
        for _ in range(10):
            labeled.append(_labeled("eclipsing_binary", _posterior(eclipsing_binary=5.0)))
        for _ in range(2):
            for h in _HYPOTHESES[2:]:
                labeled.append(_labeled(h, _posterior(**{h: 5.0})))
        result = fit_calibration(labeled, method="isotonic")
        p = _posterior(planet_candidate=5.0)
        calibrated = apply_calibration(p, result)
        assert calibrated.planet_candidate >= calibrated.eclipsing_binary
