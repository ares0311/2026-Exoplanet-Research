"""Tests for exo_toolkit.hypotheses."""
import math

import pytest

from exo_toolkit.hypotheses import (
    DEFAULT_LOG_PRIORS,
    compute_log_scores,
    log_score_background_eb,
    log_score_eclipsing_binary,
    log_score_instrumental,
    log_score_known_object,
    log_score_planet,
    log_score_stellar_variability,
)
from exo_toolkit.schemas import CandidateFeatures

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def empty_features() -> CandidateFeatures:
    """All feature scores absent — posterior should reduce to priors."""
    return CandidateFeatures()


@pytest.fixture
def clean_planet_features() -> CandidateFeatures:
    """Feature profile consistent with a genuine transiting planet."""
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
    )


@pytest.fixture
def eb_features() -> CandidateFeatures:
    """Feature profile consistent with an on-target eclipsing binary."""
    return CandidateFeatures(
        log_snr_score=0.90,
        odd_even_mismatch_score=0.95,
        secondary_eclipse_score=0.90,
        v_shape_score=0.85,
        large_depth_score=0.70,
        companion_radius_too_large_score=0.75,
        duration_implausibility_score=0.60,
    )


@pytest.fixture
def background_eb_features() -> CandidateFeatures:
    """Feature profile consistent with a background eclipsing binary."""
    return CandidateFeatures(
        centroid_offset_score=0.90,
        contamination_score=0.80,
        nearby_bright_source_score=0.70,
        aperture_edge_score=0.50,
        dilution_sensitivity_score=0.80,
    )


@pytest.fixture
def variability_features() -> CandidateFeatures:
    """Feature profile consistent with stellar variability."""
    return CandidateFeatures(
        variability_periodogram_score=0.85,
        harmonic_score=0.75,
        flare_score=0.60,
        quasi_periodic_score=0.70,
        non_box_shape_score=0.80,
    )


@pytest.fixture
def instrumental_features() -> CandidateFeatures:
    """Feature profile consistent with an instrumental artifact."""
    return CandidateFeatures(
        systematics_overlap_score=0.90,
        quality_flag_score=0.85,
        sector_boundary_score=0.70,
        background_excursion_score=0.75,
        single_event_score=1.0,
        nearby_targets_common_signal_score=0.65,
    )


@pytest.fixture
def known_object_features() -> CandidateFeatures:
    """Feature profile consistent with a catalog match."""
    return CandidateFeatures(
        target_id_match_score=1.0,
        period_match_score=1.0,
        epoch_match_score=0.95,
        coordinate_match_score=0.98,
    )


# ---------------------------------------------------------------------------
# DEFAULT_LOG_PRIORS
# ---------------------------------------------------------------------------


class TestDefaultLogPriors:
    def test_all_six_hypotheses_present(self) -> None:
        expected = {
            "planet_candidate",
            "eclipsing_binary",
            "background_eclipsing_binary",
            "stellar_variability",
            "instrumental_artifact",
            "known_object",
        }
        assert set(DEFAULT_LOG_PRIORS.keys()) == expected

    def test_priors_sum_to_one_in_probability_space(self) -> None:
        total = sum(math.exp(v) for v in DEFAULT_LOG_PRIORS.values())
        assert total == pytest.approx(1.0)

    def test_planet_and_known_object_prior_lower(self) -> None:
        # Both are set to 0.10; all others are 0.20
        assert DEFAULT_LOG_PRIORS["planet_candidate"] < DEFAULT_LOG_PRIORS["eclipsing_binary"]
        assert DEFAULT_LOG_PRIORS["known_object"] < DEFAULT_LOG_PRIORS["eclipsing_binary"]


# ---------------------------------------------------------------------------
# log_score_planet
# ---------------------------------------------------------------------------


class TestLogScorePlanet:
    def test_empty_features_equals_log_prior(
        self, empty_features: CandidateFeatures
    ) -> None:
        score = log_score_planet(empty_features)
        assert score == pytest.approx(DEFAULT_LOG_PRIORS["planet_candidate"])

    def test_positive_features_increase_score(
        self, empty_features: CandidateFeatures
    ) -> None:
        baseline = log_score_planet(empty_features)
        boosted = log_score_planet(
            CandidateFeatures(log_snr_score=1.0, transit_count_score=1.0)
        )
        assert boosted > baseline

    def test_negative_features_decrease_score(
        self, empty_features: CandidateFeatures
    ) -> None:
        baseline = log_score_planet(empty_features)
        penalised = log_score_planet(
            CandidateFeatures(odd_even_mismatch_score=1.0, secondary_eclipse_score=1.0)
        )
        assert penalised < baseline

    def test_clean_planet_signal_high_score(
        self, clean_planet_features: CandidateFeatures
    ) -> None:
        assert log_score_planet(clean_planet_features) > 0.0

    def test_custom_prior_applied(self, empty_features: CandidateFeatures) -> None:
        flat_prior = math.log(1 / 6)
        score = log_score_planet(empty_features, log_prior=flat_prior)
        assert score == pytest.approx(flat_prior)


# ---------------------------------------------------------------------------
# log_score_eclipsing_binary
# ---------------------------------------------------------------------------


class TestLogScoreEclipsingBinary:
    def test_empty_features_equals_log_prior(
        self, empty_features: CandidateFeatures
    ) -> None:
        score = log_score_eclipsing_binary(empty_features)
        assert score == pytest.approx(DEFAULT_LOG_PRIORS["eclipsing_binary"])

    def test_eb_indicators_increase_score(
        self, eb_features: CandidateFeatures
    ) -> None:
        baseline = log_score_eclipsing_binary(CandidateFeatures())
        assert log_score_eclipsing_binary(eb_features) > baseline

    def test_custom_prior_applied(self, empty_features: CandidateFeatures) -> None:
        custom = math.log(0.05)
        assert log_score_eclipsing_binary(empty_features, log_prior=custom) == pytest.approx(custom)


# ---------------------------------------------------------------------------
# log_score_background_eb
# ---------------------------------------------------------------------------


class TestLogScoreBackgroundEB:
    def test_empty_features_equals_log_prior(
        self, empty_features: CandidateFeatures
    ) -> None:
        assert log_score_background_eb(empty_features) == pytest.approx(
            DEFAULT_LOG_PRIORS["background_eclipsing_binary"]
        )

    def test_centroid_and_contamination_raise_score(
        self, background_eb_features: CandidateFeatures
    ) -> None:
        baseline = log_score_background_eb(CandidateFeatures())
        assert log_score_background_eb(background_eb_features) > baseline


# ---------------------------------------------------------------------------
# log_score_stellar_variability
# ---------------------------------------------------------------------------


class TestLogScoreStellarVariability:
    def test_empty_features_equals_log_prior(
        self, empty_features: CandidateFeatures
    ) -> None:
        assert log_score_stellar_variability(empty_features) == pytest.approx(
            DEFAULT_LOG_PRIORS["stellar_variability"]
        )

    def test_variability_indicators_raise_score(
        self, variability_features: CandidateFeatures
    ) -> None:
        baseline = log_score_stellar_variability(CandidateFeatures())
        assert log_score_stellar_variability(variability_features) > baseline


# ---------------------------------------------------------------------------
# log_score_instrumental
# ---------------------------------------------------------------------------


class TestLogScoreInstrumental:
    def test_empty_features_equals_log_prior(
        self, empty_features: CandidateFeatures
    ) -> None:
        assert log_score_instrumental(empty_features) == pytest.approx(
            DEFAULT_LOG_PRIORS["instrumental_artifact"]
        )

    def test_systematics_raise_score(
        self, instrumental_features: CandidateFeatures
    ) -> None:
        baseline = log_score_instrumental(CandidateFeatures())
        assert log_score_instrumental(instrumental_features) > baseline


# ---------------------------------------------------------------------------
# log_score_known_object
# ---------------------------------------------------------------------------


class TestLogScoreKnownObject:
    def test_empty_features_equals_log_prior(
        self, empty_features: CandidateFeatures
    ) -> None:
        assert log_score_known_object(empty_features) == pytest.approx(
            DEFAULT_LOG_PRIORS["known_object"]
        )

    def test_catalog_match_raises_score_dramatically(
        self, known_object_features: CandidateFeatures
    ) -> None:
        baseline = log_score_known_object(CandidateFeatures())
        assert log_score_known_object(known_object_features) > baseline + 5.0


# ---------------------------------------------------------------------------
# compute_log_scores
# ---------------------------------------------------------------------------


class TestComputeLogScores:
    def test_returns_all_six_keys(self, empty_features: CandidateFeatures) -> None:
        scores = compute_log_scores(empty_features)
        expected = {
            "planet_candidate",
            "eclipsing_binary",
            "background_eclipsing_binary",
            "stellar_variability",
            "instrumental_artifact",
            "known_object",
        }
        assert set(scores.keys()) == expected

    def test_empty_features_equals_priors(
        self, empty_features: CandidateFeatures
    ) -> None:
        scores = compute_log_scores(empty_features)
        for key, log_prior in DEFAULT_LOG_PRIORS.items():
            assert scores[key] == pytest.approx(log_prior), f"mismatch for {key}"

    def test_custom_prior_overrides_default(
        self, empty_features: CandidateFeatures
    ) -> None:
        custom_prior = math.log(0.50)
        scores = compute_log_scores(
            empty_features,
            log_priors={"planet_candidate": custom_prior},
        )
        assert scores["planet_candidate"] == pytest.approx(custom_prior)
        # Other hypotheses unchanged
        assert scores["eclipsing_binary"] == pytest.approx(
            DEFAULT_LOG_PRIORS["eclipsing_binary"]
        )

    def test_planet_wins_on_clean_planet_signal(
        self, clean_planet_features: CandidateFeatures
    ) -> None:
        scores = compute_log_scores(clean_planet_features)
        assert scores["planet_candidate"] == max(scores.values())

    def test_eb_wins_on_eclipsing_binary_signal(
        self, eb_features: CandidateFeatures
    ) -> None:
        scores = compute_log_scores(eb_features)
        assert scores["eclipsing_binary"] == max(scores.values())

    def test_background_eb_wins_on_centroid_signal(
        self, background_eb_features: CandidateFeatures
    ) -> None:
        scores = compute_log_scores(background_eb_features)
        assert scores["background_eclipsing_binary"] == max(scores.values())

    def test_variability_wins_on_variability_signal(
        self, variability_features: CandidateFeatures
    ) -> None:
        scores = compute_log_scores(variability_features)
        assert scores["stellar_variability"] == max(scores.values())

    def test_instrumental_wins_on_systematics_signal(
        self, instrumental_features: CandidateFeatures
    ) -> None:
        scores = compute_log_scores(instrumental_features)
        assert scores["instrumental_artifact"] == max(scores.values())

    def test_known_object_wins_on_catalog_match(
        self, known_object_features: CandidateFeatures
    ) -> None:
        scores = compute_log_scores(known_object_features)
        assert scores["known_object"] == max(scores.values())


# ---------------------------------------------------------------------------
# depth_scatter_chi2_score wiring
# ---------------------------------------------------------------------------


class TestDepthScatterChi2Wiring:
    def test_high_chi2_lowers_planet_score(self) -> None:
        base = CandidateFeatures()
        scattered = CandidateFeatures(depth_scatter_chi2_score=1.0)
        assert log_score_planet(scattered) < log_score_planet(base)

    def test_high_chi2_raises_instrumental_score(self) -> None:
        base = CandidateFeatures()
        scattered = CandidateFeatures(depth_scatter_chi2_score=1.0)
        assert log_score_instrumental(scattered) > log_score_instrumental(base)

    def test_zero_chi2_neutral_for_planet(self) -> None:
        base = CandidateFeatures()
        clean = CandidateFeatures(depth_scatter_chi2_score=0.0)
        assert log_score_planet(clean) == pytest.approx(log_score_planet(base))

    def test_none_chi2_neutral(self) -> None:
        base = CandidateFeatures()
        null_chi2 = CandidateFeatures(depth_scatter_chi2_score=None)
        assert log_score_planet(null_chi2) == pytest.approx(log_score_planet(base))
        assert log_score_instrumental(null_chi2) == pytest.approx(
            log_score_instrumental(base)
        )

    def test_scattered_depths_boost_instrumental_in_full_scores(self) -> None:
        clean = CandidateFeatures(depth_scatter_chi2_score=0.0)
        scattered = CandidateFeatures(depth_scatter_chi2_score=1.0)
        scores_clean = compute_log_scores(clean)
        scores_scattered = compute_log_scores(scattered)
        assert scores_scattered["instrumental_artifact"] > scores_clean["instrumental_artifact"]
        assert scores_scattered["planet_candidate"] < scores_clean["planet_candidate"]


class TestTransitTimingVariationWiring:
    def test_high_ttv_lowers_planet_score(self) -> None:
        base = CandidateFeatures()
        erratic = CandidateFeatures(transit_timing_variation_score=1.0)
        assert log_score_planet(erratic) < log_score_planet(base)

    def test_high_ttv_raises_instrumental_score(self) -> None:
        base = CandidateFeatures()
        erratic = CandidateFeatures(transit_timing_variation_score=1.0)
        assert log_score_instrumental(erratic) > log_score_instrumental(base)

    def test_none_ttv_is_neutral(self) -> None:
        base = CandidateFeatures()
        null_ttv = CandidateFeatures(transit_timing_variation_score=None)
        assert log_score_planet(null_ttv) == pytest.approx(log_score_planet(base))
        assert log_score_instrumental(null_ttv) == pytest.approx(log_score_instrumental(base))


# ---------------------------------------------------------------------------
# New feature wiring tests (Milestones 12a-12e)
# ---------------------------------------------------------------------------


class TestOutOfTransitScatterWiring:
    def test_high_oot_lowers_planet_score(self) -> None:
        base = CandidateFeatures()
        scattered = CandidateFeatures(out_of_transit_scatter_score=1.0)
        assert log_score_planet(scattered) < log_score_planet(base)

    def test_high_oot_raises_instrumental_score(self) -> None:
        base = CandidateFeatures()
        scattered = CandidateFeatures(out_of_transit_scatter_score=1.0)
        assert log_score_instrumental(scattered) > log_score_instrumental(base)


class TestMultiSectorDepthWiring:
    def test_consistent_sector_depths_raises_planet_score(self) -> None:
        base = CandidateFeatures()
        consistent = CandidateFeatures(multi_sector_depth_consistency_score=1.0)
        assert log_score_planet(consistent) > log_score_planet(base)

    def test_consistent_sector_depths_lowers_instrumental_score(self) -> None:
        base = CandidateFeatures()
        consistent = CandidateFeatures(multi_sector_depth_consistency_score=1.0)
        assert log_score_instrumental(consistent) < log_score_instrumental(base)


class TestStellarDensityWiring:
    def test_consistent_density_raises_planet_score(self) -> None:
        base = CandidateFeatures()
        consistent = CandidateFeatures(stellar_density_consistency_score=1.0)
        assert log_score_planet(consistent) > log_score_planet(base)

    def test_consistent_density_lowers_eb_score(self) -> None:
        from exo_toolkit.hypotheses import log_score_eclipsing_binary
        base = CandidateFeatures()
        consistent = CandidateFeatures(stellar_density_consistency_score=1.0)
        assert log_score_eclipsing_binary(consistent) < log_score_eclipsing_binary(base)


class TestCentroidMotionWiring:
    def test_high_centroid_motion_lowers_planet_score(self) -> None:
        base = CandidateFeatures()
        motion = CandidateFeatures(centroid_motion_score=1.0)
        assert log_score_planet(motion) < log_score_planet(base)

    def test_high_centroid_motion_raises_beb_score(self) -> None:
        from exo_toolkit.hypotheses import log_score_background_eb
        base = CandidateFeatures()
        motion = CandidateFeatures(centroid_motion_score=1.0)
        assert log_score_background_eb(motion) > log_score_background_eb(base)


class TestLimbDarkeningWiring:
    def test_plausible_ld_raises_planet_score(self) -> None:
        base = CandidateFeatures()
        plausible = CandidateFeatures(limb_darkening_plausibility_score=1.0)
        assert log_score_planet(plausible) > log_score_planet(base)

    def test_plausible_ld_lowers_eb_score(self) -> None:
        from exo_toolkit.hypotheses import log_score_eclipsing_binary
        base = CandidateFeatures()
        plausible = CandidateFeatures(limb_darkening_plausibility_score=1.0)
        assert log_score_eclipsing_binary(plausible) < log_score_eclipsing_binary(base)
