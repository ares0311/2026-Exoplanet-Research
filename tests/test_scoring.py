"""Tests for exo_toolkit.scoring."""
import math

import pytest

from exo_toolkit.hypotheses import DEFAULT_LOG_PRIORS
from exo_toolkit.schemas import CandidateFeatures, CandidateScores, CandidateSignal
from exo_toolkit.scoring import (
    compute_posterior,
    compute_scores,
    score_candidate,
    softmax,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def base_signal() -> CandidateSignal:
    return CandidateSignal(
        candidate_id="TIC_999_s01",
        mission="TESS",
        target_id="TIC 999",
        period_days=10.0,
        epoch_bjd=2459000.0,
        duration_hours=2.5,
        depth_ppm=1000.0,
        transit_count=3,
        snr=10.0,
    )


@pytest.fixture
def empty_features() -> CandidateFeatures:
    return CandidateFeatures()


@pytest.fixture
def clean_planet_features() -> CandidateFeatures:
    return CandidateFeatures(
        log_snr_score=1.0,
        transit_count_score=1.0,
        depth_consistency_score=0.95,
        duration_consistency_score=0.90,
        duration_plausibility_score=0.90,
        transit_shape_score=0.80,
        odd_even_mismatch_score=0.02,
        secondary_eclipse_score=0.05,
        centroid_offset_score=0.05,
        contamination_score=0.05,
        systematics_overlap_score=0.05,
        stellar_variability_score=0.05,
        known_object_score=0.0,
    )


@pytest.fixture
def eb_features() -> CandidateFeatures:
    return CandidateFeatures(
        odd_even_mismatch_score=0.95,
        secondary_eclipse_score=0.90,
        v_shape_score=0.85,
        large_depth_score=0.70,
        companion_radius_too_large_score=0.75,
    )


@pytest.fixture
def known_object_features() -> CandidateFeatures:
    return CandidateFeatures(
        target_id_match_score=1.0,
        period_match_score=1.0,
        epoch_match_score=0.95,
        coordinate_match_score=0.98,
        known_object_score=0.99,
    )


# ---------------------------------------------------------------------------
# softmax
# ---------------------------------------------------------------------------


class TestSoftmax:
    def test_output_sums_to_one(self) -> None:
        scores = {"a": 1.0, "b": 2.0, "c": 0.5}
        result = softmax(scores)
        assert sum(result.values()) == pytest.approx(1.0)

    def test_highest_log_score_gets_highest_probability(self) -> None:
        scores = {"planet": 5.0, "eb": 1.0, "other": -1.0}
        result = softmax(scores)
        assert result["planet"] == max(result.values())

    def test_equal_scores_give_equal_probabilities(self) -> None:
        scores = {"a": 2.0, "b": 2.0, "c": 2.0}
        result = softmax(scores)
        assert result["a"] == pytest.approx(result["b"])
        assert result["b"] == pytest.approx(result["c"])

    def test_numerically_stable_with_large_values(self) -> None:
        scores = {"a": 1000.0, "b": 999.0}
        result = softmax(scores)
        assert sum(result.values()) == pytest.approx(1.0)
        assert not any(math.isnan(v) for v in result.values())

    def test_numerically_stable_with_negative_values(self) -> None:
        scores = {"a": -1000.0, "b": -999.0}
        result = softmax(scores)
        assert sum(result.values()) == pytest.approx(1.0)

    def test_single_entry_gives_probability_one(self) -> None:
        result = softmax({"only": 42.0})
        assert result["only"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# compute_posterior
# ---------------------------------------------------------------------------


class TestComputePosterior:
    def test_posterior_sums_to_one(
        self, empty_features: CandidateFeatures
    ) -> None:
        p = compute_posterior(empty_features)
        total = (
            p.planet_candidate
            + p.eclipsing_binary
            + p.background_eclipsing_binary
            + p.stellar_variability
            + p.instrumental_artifact
            + p.known_object
        )
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_empty_features_reflects_priors(
        self, empty_features: CandidateFeatures
    ) -> None:
        p = compute_posterior(empty_features)
        # With no evidence, hypotheses with equal priors get equal posteriors
        assert p.eclipsing_binary == pytest.approx(p.background_eclipsing_binary)
        assert p.eclipsing_binary == pytest.approx(p.stellar_variability)
        assert p.eclipsing_binary == pytest.approx(p.instrumental_artifact)
        # Planet and known_object have half the prior — they get lower posteriors
        assert p.planet_candidate < p.eclipsing_binary
        assert p.known_object < p.eclipsing_binary

    def test_planet_features_raise_planet_posterior(
        self, clean_planet_features: CandidateFeatures
    ) -> None:
        p = compute_posterior(clean_planet_features)
        assert p.planet_candidate > 0.5

    def test_eb_features_raise_eb_posterior(
        self, eb_features: CandidateFeatures
    ) -> None:
        p = compute_posterior(eb_features)
        assert p.eclipsing_binary > p.planet_candidate

    def test_known_object_features_raise_known_object_posterior(
        self, known_object_features: CandidateFeatures
    ) -> None:
        p = compute_posterior(known_object_features)
        assert p.known_object > 0.5

    def test_custom_priors_applied(
        self, empty_features: CandidateFeatures
    ) -> None:
        # Give planet_candidate a dominant prior; suppress all others
        custom = {k: math.log(0.01) for k in DEFAULT_LOG_PRIORS}
        custom["planet_candidate"] = math.log(0.95)
        p = compute_posterior(empty_features, log_priors=custom)
        assert p.planet_candidate > 0.5

    def test_all_posteriors_in_unit_interval(
        self, clean_planet_features: CandidateFeatures
    ) -> None:
        p = compute_posterior(clean_planet_features)
        for val in (
            p.planet_candidate,
            p.eclipsing_binary,
            p.background_eclipsing_binary,
            p.stellar_variability,
            p.instrumental_artifact,
            p.known_object,
        ):
            assert 0.0 <= val <= 1.0


# ---------------------------------------------------------------------------
# compute_scores
# ---------------------------------------------------------------------------


class TestComputeScores:
    def test_fpp_equals_one_minus_planet_posterior(
        self,
        base_signal: CandidateSignal,
        clean_planet_features: CandidateFeatures,
    ) -> None:
        posterior = compute_posterior(clean_planet_features)
        scores = compute_scores(base_signal, clean_planet_features, posterior)
        assert scores.false_positive_probability == pytest.approx(
            1.0 - posterior.planet_candidate, abs=1e-9
        )

    def test_detection_confidence_neutral_with_no_features(
        self,
        base_signal: CandidateSignal,
        empty_features: CandidateFeatures,
    ) -> None:
        posterior = compute_posterior(empty_features)
        scores = compute_scores(base_signal, empty_features, posterior)
        # sigmoid(0) = 0.5 — maximum uncertainty
        assert scores.detection_confidence == pytest.approx(0.5)

    def test_strong_signal_raises_detection_confidence(
        self,
        base_signal: CandidateSignal,
        clean_planet_features: CandidateFeatures,
    ) -> None:
        posterior = compute_posterior(clean_planet_features)
        scores = compute_scores(base_signal, clean_planet_features, posterior)
        assert scores.detection_confidence > 0.85

    def test_novelty_score_one_when_no_catalog_match(
        self,
        base_signal: CandidateSignal,
        empty_features: CandidateFeatures,
    ) -> None:
        posterior = compute_posterior(empty_features)
        scores = compute_scores(base_signal, empty_features, posterior)
        assert scores.novelty_score == pytest.approx(1.0)

    def test_novelty_score_near_zero_for_known_object(
        self,
        base_signal: CandidateSignal,
        known_object_features: CandidateFeatures,
    ) -> None:
        posterior = compute_posterior(known_object_features)
        scores = compute_scores(base_signal, known_object_features, posterior)
        assert scores.novelty_score < 0.05

    def test_all_scores_in_unit_interval(
        self,
        base_signal: CandidateSignal,
        clean_planet_features: CandidateFeatures,
    ) -> None:
        posterior = compute_posterior(clean_planet_features)
        scores = compute_scores(base_signal, clean_planet_features, posterior)
        for field in CandidateScores.model_fields:
            val = getattr(scores, field)
            assert 0.0 <= val <= 1.0, f"{field} = {val} outside [0, 1]"

    def test_habitability_interest_higher_for_longer_period(
        self,
        clean_planet_features: CandidateFeatures,
    ) -> None:
        short_signal = CandidateSignal(
            candidate_id="x",
            mission="TESS",
            target_id="y",
            period_days=2.0,
            epoch_bjd=0.0,
            duration_hours=1.0,
            depth_ppm=1000.0,
            transit_count=3,
            snr=10.0,
        )
        long_signal = CandidateSignal(
            candidate_id="x",
            mission="TESS",
            target_id="y",
            period_days=250.0,
            epoch_bjd=0.0,
            duration_hours=4.0,
            depth_ppm=1000.0,
            transit_count=3,
            snr=10.0,
        )
        posterior = compute_posterior(clean_planet_features)
        short_scores = compute_scores(short_signal, clean_planet_features, posterior)
        long_scores = compute_scores(long_signal, clean_planet_features, posterior)
        assert long_scores.habitability_interest > short_scores.habitability_interest


# ---------------------------------------------------------------------------
# score_candidate
# ---------------------------------------------------------------------------


class TestScoreCandidate:
    def test_returns_posterior_and_scores(
        self,
        base_signal: CandidateSignal,
        empty_features: CandidateFeatures,
    ) -> None:
        posterior, scores = score_candidate(base_signal, empty_features)
        assert isinstance(posterior, type(posterior))
        total = (
            posterior.planet_candidate
            + posterior.eclipsing_binary
            + posterior.background_eclipsing_binary
            + posterior.stellar_variability
            + posterior.instrumental_artifact
            + posterior.known_object
        )
        assert total == pytest.approx(1.0, abs=1e-9)
        assert 0.0 <= scores.false_positive_probability <= 1.0

    def test_clean_planet_gives_low_fpp(
        self,
        base_signal: CandidateSignal,
        clean_planet_features: CandidateFeatures,
    ) -> None:
        posterior, scores = score_candidate(base_signal, clean_planet_features)
        assert scores.false_positive_probability < 0.5
        assert posterior.planet_candidate > 0.5

    def test_eb_signal_gives_high_fpp(
        self,
        base_signal: CandidateSignal,
        eb_features: CandidateFeatures,
    ) -> None:
        _, scores = score_candidate(base_signal, eb_features)
        assert scores.false_positive_probability > 0.5

    def test_custom_priors_flow_through(
        self,
        base_signal: CandidateSignal,
        empty_features: CandidateFeatures,
    ) -> None:
        custom = {k: math.log(0.01) for k in DEFAULT_LOG_PRIORS}
        custom["planet_candidate"] = math.log(0.95)
        posterior, _ = score_candidate(base_signal, empty_features, log_priors=custom)
        assert posterior.planet_candidate > 0.5

    def test_known_object_novelty_near_zero(
        self,
        base_signal: CandidateSignal,
        known_object_features: CandidateFeatures,
    ) -> None:
        _, scores = score_candidate(base_signal, known_object_features)
        assert scores.novelty_score < 0.05


# ---------------------------------------------------------------------------
# Scoring invariants (Milestone 12o)
# ---------------------------------------------------------------------------


def _planet_features() -> CandidateFeatures:
    return CandidateFeatures(
        log_snr_score=0.9, transit_count_score=0.8, depth_consistency_score=0.9,
        transit_shape_score=0.8, duration_plausibility_score=0.8,
        odd_even_mismatch_score=0.0, secondary_eclipse_score=0.0,
        centroid_offset_score=0.0, contamination_score=0.0,
        systematics_overlap_score=0.0, stellar_variability_score=0.0,
    )


def _eb_features() -> CandidateFeatures:
    return CandidateFeatures(
        odd_even_mismatch_score=0.9, secondary_eclipse_score=0.9,
        v_shape_score=0.8, large_depth_score=0.8,
        companion_radius_too_large_score=0.8,
    )


def _instrumental_features() -> CandidateFeatures:
    return CandidateFeatures(
        systematics_overlap_score=0.9, quality_flag_score=0.9,
        sector_boundary_score=0.8, depth_scatter_chi2_score=0.9,
        transit_timing_variation_score=0.8,
    )


def _make_signal() -> CandidateSignal:
    return CandidateSignal(
        candidate_id="TIC_1_s01", mission="TESS", target_id="TIC 1",
        period_days=10.0, epoch_bjd=2459000.0, duration_hours=2.5,
        depth_ppm=1000.0, transit_count=3, snr=10.0,
    )


class TestScoringInvariants:
    def test_planet_features_planet_wins(self) -> None:
        posterior, _ = score_candidate(_make_signal(), _planet_features())
        vals = {
            "planet": posterior.planet_candidate,
            "eb": posterior.eclipsing_binary,
            "beb": posterior.background_eclipsing_binary,
            "sv": posterior.stellar_variability,
            "ia": posterior.instrumental_artifact,
            "ko": posterior.known_object,
        }
        assert posterior.planet_candidate == max(vals.values())

    def test_eb_features_eb_wins(self) -> None:
        posterior, _ = score_candidate(_make_signal(), _eb_features())
        assert posterior.eclipsing_binary == max(
            posterior.planet_candidate, posterior.eclipsing_binary,
            posterior.background_eclipsing_binary, posterior.stellar_variability,
            posterior.instrumental_artifact, posterior.known_object,
        )

    def test_instrumental_features_instrumental_wins(self) -> None:
        posterior, _ = score_candidate(_make_signal(), _instrumental_features())
        assert posterior.instrumental_artifact == max(
            posterior.planet_candidate, posterior.eclipsing_binary,
            posterior.background_eclipsing_binary, posterior.stellar_variability,
            posterior.instrumental_artifact, posterior.known_object,
        )

    def test_empty_features_planet_below_threshold(self) -> None:
        posterior, _ = score_candidate(_make_signal(), CandidateFeatures())
        assert posterior.planet_candidate < 0.30

    @pytest.mark.parametrize("features", [
        _planet_features(),
        _eb_features(),
        _instrumental_features(),
        CandidateFeatures(),
        CandidateFeatures(log_snr_score=0.5),
    ])
    def test_posteriors_sum_to_one(self, features: CandidateFeatures) -> None:
        posterior, _ = score_candidate(_make_signal(), features)
        total = (
            posterior.planet_candidate + posterior.eclipsing_binary
            + posterior.background_eclipsing_binary + posterior.stellar_variability
            + posterior.instrumental_artifact + posterior.known_object
        )
        assert total == pytest.approx(1.0, abs=0.01)

    def test_clean_planet_gives_low_fpp(self) -> None:
        _, scores = score_candidate(_make_signal(), _planet_features())
        assert scores.false_positive_probability < 0.5

    def test_eb_features_give_high_fpp(self) -> None:
        _, scores = score_candidate(_make_signal(), _eb_features())
        assert scores.false_positive_probability > 0.5

    def test_custom_log_priors_shift_posteriors(self) -> None:
        default_posterior, _ = score_candidate(_make_signal(), CandidateFeatures())
        strong_planet_prior = {k: math.log(0.01) for k in DEFAULT_LOG_PRIORS}
        strong_planet_prior["planet_candidate"] = math.log(0.95)
        strong_posterior, _ = score_candidate(
            _make_signal(), CandidateFeatures(), log_priors=strong_planet_prior,
        )
        assert strong_posterior.planet_candidate > default_posterior.planet_candidate


class TestHypothesisWeightSensitivity:
    def test_score_candidate_returns_tuple(self) -> None:
        result = score_candidate(_make_signal(), CandidateFeatures())
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_detection_confidence_higher_for_clean_features(self) -> None:
        _, noisy = score_candidate(_make_signal(), CandidateFeatures())
        _, clean = score_candidate(_make_signal(), _planet_features())
        assert clean.detection_confidence > noisy.detection_confidence

    def test_novelty_score_is_high_with_no_catalog_match(self) -> None:
        _, scores = score_candidate(_make_signal(), CandidateFeatures(known_object_score=None))
        assert scores.novelty_score > 0.5

    def test_novelty_score_lower_with_catalog_match(self) -> None:
        _, no_match = score_candidate(_make_signal(), CandidateFeatures())
        _, match = score_candidate(
            _make_signal(),
            CandidateFeatures(known_object_score=1.0, target_id_match_score=1.0),
        )
        assert match.novelty_score < no_match.novelty_score

    def test_fpp_equals_one_minus_planet_posterior(self) -> None:
        posterior, scores = score_candidate(_make_signal(), _planet_features())
        assert abs(scores.false_positive_probability - (1.0 - posterior.planet_candidate)) < 0.01

    def test_higher_fp_features_lower_detection_confidence(self) -> None:
        _, clean_scores = score_candidate(_make_signal(), _planet_features())
        _, fp_scores = score_candidate(_make_signal(), _eb_features())
        assert clean_scores.detection_confidence > fp_scores.detection_confidence

    def test_custom_priors_override_defaults(self) -> None:
        low_planet_prior = dict(DEFAULT_LOG_PRIORS)
        low_planet_prior["planet_candidate"] = math.log(0.01)
        posterior, _ = score_candidate(
            _make_signal(), CandidateFeatures(), log_priors=low_planet_prior
        )
        assert posterior.planet_candidate < 0.15

    def test_all_score_fields_in_unit_interval(self) -> None:
        _, scores = score_candidate(_make_signal(), _planet_features())
        for val in [
            scores.false_positive_probability, scores.detection_confidence,
            scores.novelty_score, scores.habitability_interest,
            scores.followup_value, scores.submission_readiness,
        ]:
            assert 0.0 <= val <= 1.0
