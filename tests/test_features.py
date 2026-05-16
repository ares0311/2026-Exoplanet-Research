"""Tests for exo_toolkit.features."""
import pytest

from exo_toolkit.features import (
    RawDiagnostics,
    aperture_edge_score,
    background_excursion_score,
    centroid_offset_score,
    companion_radius_too_large_score,
    contamination_score,
    coordinate_match_score,
    data_gap_overlap_score,
    depth_consistency_score,
    dilution_sensitivity_score,
    duration_consistency_score,
    duration_implausibility_score,
    duration_plausibility_score,
    epoch_match_score,
    extract_features,
    flare_score,
    harmonic_score,
    known_object_score,
    large_depth_score,
    log_snr_score,
    nearby_bright_source_score,
    nearby_targets_common_signal_score,
    non_box_shape_score,
    odd_even_mismatch_score,
    period_match_score,
    quality_flag_score,
    quasi_periodic_score,
    secondary_eclipse_score,
    sector_boundary_score,
    single_event_score,
    snr_score,
    stellar_variability_score,
    systematics_overlap_score,
    target_id_match_score,
    transit_count_score,
    depth_scatter_chi2_score,
    transit_timing_variation_score,
    transit_shape_score,
    v_shape_score,
    variability_periodogram_score,
)
from exo_toolkit.schemas import CandidateFeatures, CandidateSignal

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def base_signal() -> CandidateSignal:
    return CandidateSignal(
        candidate_id="TIC_111_s01",
        mission="TESS",
        target_id="TIC 111",
        period_days=10.0,
        epoch_bjd=2459000.0,
        duration_hours=2.5,
        depth_ppm=1000.0,
        transit_count=3,
        snr=10.0,
    )


@pytest.fixture
def empty_diagnostics() -> RawDiagnostics:
    return RawDiagnostics()


# ---------------------------------------------------------------------------
# snr_score
# ---------------------------------------------------------------------------


class TestSnrScore:
    def test_at_lower_threshold(self) -> None:
        assert snr_score(5.0) == pytest.approx(0.0)

    def test_at_upper_threshold(self) -> None:
        assert snr_score(12.0) == pytest.approx(1.0)

    def test_midpoint(self) -> None:
        assert 0.0 < snr_score(8.5) < 1.0

    def test_clips_below_zero(self) -> None:
        assert snr_score(0.0) == pytest.approx(0.0)

    def test_clips_above_one(self) -> None:
        assert snr_score(100.0) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# log_snr_score
# ---------------------------------------------------------------------------


class TestLogSnrScore:
    def test_at_snr_one(self) -> None:
        assert log_snr_score(1.0) == pytest.approx(0.0)

    def test_at_snr_twelve(self) -> None:
        assert log_snr_score(12.0) == pytest.approx(1.0)

    def test_clips_below_zero(self) -> None:
        assert log_snr_score(0.0) == pytest.approx(0.0)

    def test_clips_above_one(self) -> None:
        assert log_snr_score(100.0) == pytest.approx(1.0)

    def test_monotonically_increasing(self) -> None:
        scores = [log_snr_score(s) for s in [1, 3, 6, 9, 12]]
        assert all(a <= b for a, b in zip(scores, scores[1:], strict=False))


# ---------------------------------------------------------------------------
# transit_count_score
# ---------------------------------------------------------------------------


class TestTransitCountScore:
    def test_single_transit(self) -> None:
        assert transit_count_score(1) == pytest.approx(0.25)

    def test_two_transits(self) -> None:
        assert transit_count_score(2) == pytest.approx(0.70)

    def test_three_transits(self) -> None:
        assert transit_count_score(3) == pytest.approx(1.00)

    def test_many_transits(self) -> None:
        assert transit_count_score(20) == pytest.approx(1.00)


# ---------------------------------------------------------------------------
# depth_consistency_score
# ---------------------------------------------------------------------------


class TestDepthConsistencyScore:
    def test_identical_depths_score_one(self) -> None:
        depths = (1000.0, 1000.0, 1000.0)
        errors = (10.0, 10.0, 10.0)
        assert depth_consistency_score(depths, errors) == pytest.approx(1.0)

    def test_single_transit_returns_none(self) -> None:
        assert depth_consistency_score((1000.0,), (10.0,)) is None

    def test_highly_variable_depths_score_near_zero(self) -> None:
        # MAD/median ≈ 0.5, well above default threshold of 0.30
        depths = (500.0, 1000.0, 1500.0)
        errors = (50.0, 50.0, 50.0)
        score = depth_consistency_score(depths, errors)
        assert score is not None
        assert score == pytest.approx(0.0)

    def test_mildly_variable_depths_between_zero_and_one(self) -> None:
        depths = (980.0, 1000.0, 1020.0)
        errors = (10.0, 10.0, 10.0)
        score = depth_consistency_score(depths, errors)
        assert score is not None
        assert 0.0 < score < 1.0


# ---------------------------------------------------------------------------
# duration_consistency_score
# ---------------------------------------------------------------------------


class TestDurationConsistencyScore:
    def test_identical_durations_score_one(self) -> None:
        durations = (2.5, 2.5, 2.5)
        errors = (0.1, 0.1, 0.1)
        assert duration_consistency_score(durations, errors) == pytest.approx(1.0)

    def test_single_transit_returns_none(self) -> None:
        assert duration_consistency_score((2.5,), (0.1,)) is None


# ---------------------------------------------------------------------------
# duration_plausibility_score
# ---------------------------------------------------------------------------


class TestDurationPlausibilityScore:
    def test_plausible_duration_scores_high(self) -> None:
        # Solar-type star, 10-day period: T_max ≈ 3.4 h; 2.5 h is central
        score = duration_plausibility_score(2.5, 10.0)
        assert score > 0.7

    def test_implausibly_long_duration_scores_low(self) -> None:
        score = duration_plausibility_score(48.0, 10.0)
        assert score < 0.3

    def test_zero_duration_scores_zero(self) -> None:
        assert duration_plausibility_score(0.0, 10.0) == pytest.approx(0.0)

    def test_score_is_clipped(self) -> None:
        score = duration_plausibility_score(2.5, 10.0)
        assert 0.0 <= score <= 1.0

    def test_longer_period_allows_longer_duration(self) -> None:
        short_p = duration_plausibility_score(5.0, 10.0)
        long_p = duration_plausibility_score(5.0, 100.0)
        # 5 h is more plausible for a 100-day orbit than a 10-day orbit
        assert long_p >= short_p


# ---------------------------------------------------------------------------
# odd_even_mismatch_score
# ---------------------------------------------------------------------------


class TestOddEvenMismatchScore:
    def test_no_mismatch_scores_zero(self) -> None:
        assert odd_even_mismatch_score(1000.0, 10.0, 1000.0, 10.0) == pytest.approx(0.0)

    def test_five_sigma_mismatch_scores_one(self) -> None:
        # |1000 - 900| / sqrt(10² + 10²) ≈ 7.07 sigma → clipped to 1
        assert odd_even_mismatch_score(1000.0, 10.0, 900.0, 10.0) == pytest.approx(1.0)

    def test_zero_error_returns_zero(self) -> None:
        assert odd_even_mismatch_score(1000.0, 0.0, 900.0, 0.0) == pytest.approx(0.0)

    def test_partial_mismatch_between_zero_and_one(self) -> None:
        # 2.5-sigma mismatch → score ≈ 0.5
        score = odd_even_mismatch_score(1000.0, 20.0, 950.0, 20.0)
        assert 0.0 < score < 1.0


# ---------------------------------------------------------------------------
# secondary_eclipse_score
# ---------------------------------------------------------------------------


class TestSecondaryEclipseScore:
    def test_no_signal_scores_zero(self) -> None:
        assert secondary_eclipse_score(0.0) == pytest.approx(0.0)

    def test_snr_seven_scores_one(self) -> None:
        assert secondary_eclipse_score(7.0) == pytest.approx(1.0)

    def test_clips_above_one(self) -> None:
        assert secondary_eclipse_score(20.0) == pytest.approx(1.0)

    def test_intermediate_snr(self) -> None:
        score = secondary_eclipse_score(3.5)
        assert score == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# transit_shape_score / v_shape_score / non_box_shape_score
# ---------------------------------------------------------------------------


class TestShapeScores:
    def test_flat_bottom_transit_shape_one(self) -> None:
        assert transit_shape_score(1.0) == pytest.approx(1.0)

    def test_v_shape_transit_shape_zero(self) -> None:
        assert transit_shape_score(0.0) == pytest.approx(0.0)

    def test_v_shape_v_score_one(self) -> None:
        assert v_shape_score(0.0) == pytest.approx(1.0)

    def test_flat_bottom_v_score_zero(self) -> None:
        assert v_shape_score(1.0) == pytest.approx(0.0)

    def test_non_box_is_inverse_of_transit_shape(self) -> None:
        for frac in (0.0, 0.3, 0.7, 1.0):
            assert non_box_shape_score(frac) == pytest.approx(1.0 - frac)

    def test_shape_and_v_sum_to_one(self) -> None:
        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            assert transit_shape_score(frac) + v_shape_score(frac) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# large_depth_score / companion_radius_too_large_score
# ---------------------------------------------------------------------------


class TestDepthScores:
    def test_shallow_depth_large_score_zero(self) -> None:
        assert large_depth_score(1000.0) == pytest.approx(0.0)

    def test_deep_eb_depth_large_score_one(self) -> None:
        assert large_depth_score(100_000.0) == pytest.approx(1.0)

    def test_large_depth_clips(self) -> None:
        assert large_depth_score(200_000.0) == pytest.approx(1.0)

    def test_small_companion_radius_zero(self) -> None:
        # depth 100 ppm → R_companion ≈ 0.01 R_Sun → well below 0.15 threshold
        assert companion_radius_too_large_score(100.0) == pytest.approx(0.0)

    def test_large_companion_radius_one(self) -> None:
        # depth 250_000 ppm → R_companion ≈ 0.5 R_Sun for 1 R_Sun star
        assert companion_radius_too_large_score(250_000.0) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# duration_implausibility_score
# ---------------------------------------------------------------------------


class TestDurationImplausibilityScore:
    def test_plausible_duration_implausibility_low(self) -> None:
        score = duration_implausibility_score(2.5, 10.0)
        assert score < 0.3

    def test_implausible_duration_implausibility_high(self) -> None:
        score = duration_implausibility_score(48.0, 10.0)
        assert score > 0.7

    def test_sums_with_plausibility_to_one(self) -> None:
        for dur in (1.0, 2.5, 10.0, 30.0):
            p = duration_plausibility_score(dur, 10.0)
            ip = duration_implausibility_score(dur, 10.0)
            assert p + ip == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# contamination / centroid / aperture / dilution
# ---------------------------------------------------------------------------


class TestContaminationScores:
    def test_no_contamination_zero(self) -> None:
        assert contamination_score(0.0) == pytest.approx(0.0)

    def test_full_contamination_one(self) -> None:
        assert contamination_score(1.0) == pytest.approx(1.0)

    def test_centroid_no_offset_zero(self) -> None:
        assert centroid_offset_score(0.0) == pytest.approx(0.0)

    def test_centroid_five_sigma_one(self) -> None:
        assert centroid_offset_score(5.0) == pytest.approx(1.0)

    def test_aperture_edge_proximate_score(self) -> None:
        assert aperture_edge_score(0.0) == pytest.approx(0.0)
        assert aperture_edge_score(1.0) == pytest.approx(1.0)

    def test_dilution_equals_contamination(self) -> None:
        for ratio in (0.0, 0.3, 0.7, 1.0):
            assert dilution_sensitivity_score(ratio) == pytest.approx(
                contamination_score(ratio)
            )


# ---------------------------------------------------------------------------
# nearby_bright_source_score
# ---------------------------------------------------------------------------


class TestNearbyBrightSourceScore:
    def test_no_sources_zero(self) -> None:
        assert nearby_bright_source_score(0, None) == pytest.approx(0.0)

    def test_three_sources_no_mag_one(self) -> None:
        assert nearby_bright_source_score(3, None) == pytest.approx(1.0)

    def test_bright_nearby_source_raises_score(self) -> None:
        # Δmag = 0 → maximally contaminating
        score_bright = nearby_bright_source_score(1, 0.0)
        score_faint = nearby_bright_source_score(1, 5.0)
        assert score_bright > score_faint


# ---------------------------------------------------------------------------
# stellar_variability_score
# ---------------------------------------------------------------------------


class TestStellarVariabilityScore:
    def test_all_none_returns_none(self) -> None:
        assert stellar_variability_score(None, None, None, None) is None

    def test_single_input_uses_it(self) -> None:
        score = stellar_variability_score(0.5, None, None, None)
        assert score is not None
        assert score == pytest.approx(variability_periodogram_score(0.5))

    def test_multiple_inputs_averaged(self) -> None:
        score = stellar_variability_score(0.5, 0.0, None, None)
        assert score is not None
        expected = (variability_periodogram_score(0.5) + harmonic_score(0.0)) / 2
        assert score == pytest.approx(expected)

    def test_result_clipped_to_unit_interval(self) -> None:
        # All inputs at maximum
        score = stellar_variability_score(1.0, 1.0, 10.0, 1.0)
        assert score is not None
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# systematics_overlap_score
# ---------------------------------------------------------------------------


class TestSystematicsOverlapScore:
    def test_all_none_returns_none(self) -> None:
        assert systematics_overlap_score(None, None, None) is None

    def test_single_input_propagates(self) -> None:
        score = systematics_overlap_score(0.8, None, None)
        assert score == pytest.approx(quality_flag_score(0.8))

    def test_takes_maximum_of_components(self) -> None:
        # quality_flag_fraction=0.2, background_excursion=10σ (→ 1.0)
        score = systematics_overlap_score(0.2, None, 10.0)
        assert score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# known_object_score
# ---------------------------------------------------------------------------


class TestKnownObjectScore:
    def test_all_none_returns_none(self) -> None:
        assert known_object_score(None, None, None, None) is None

    def test_perfect_match_scores_near_one(self) -> None:
        score = known_object_score(True, 0.0, 0.0, 0.0)
        assert score is not None
        assert score > 0.9

    def test_no_match_scores_near_zero(self) -> None:
        score = known_object_score(False, 10.0, 10.0, 60.0)
        assert score is not None
        assert score == pytest.approx(0.0)

    def test_partial_match_between_zero_and_one(self) -> None:
        score = known_object_score(None, 1.5, None, 10.0)
        assert score is not None
        assert 0.0 < score < 1.0

    def test_target_id_match_score_binary(self) -> None:
        assert target_id_match_score(True) == pytest.approx(1.0)
        assert target_id_match_score(False) == pytest.approx(0.0)

    def test_period_match_score_zero_sigma_is_one(self) -> None:
        assert period_match_score(0.0) == pytest.approx(1.0)

    def test_period_match_score_three_sigma_is_zero(self) -> None:
        assert period_match_score(3.0) == pytest.approx(0.0)

    def test_coordinate_match_zero_arcsec_is_one(self) -> None:
        assert coordinate_match_score(0.0) == pytest.approx(1.0)

    def test_coordinate_match_thirty_arcsec_is_zero(self) -> None:
        assert coordinate_match_score(30.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# individual instrumental scores
# ---------------------------------------------------------------------------


class TestInstrumentalScores:
    def test_quality_flag_score(self) -> None:
        assert quality_flag_score(0.0) == pytest.approx(0.0)
        assert quality_flag_score(1.0) == pytest.approx(1.0)

    def test_sector_boundary_score(self) -> None:
        assert sector_boundary_score(0.5) == pytest.approx(0.5)

    def test_background_excursion_five_sigma(self) -> None:
        assert background_excursion_score(5.0) == pytest.approx(1.0)

    def test_single_event_score_true_for_one_transit(self) -> None:
        assert single_event_score(1) == pytest.approx(1.0)

    def test_single_event_score_false_for_multiple(self) -> None:
        assert single_event_score(2) == pytest.approx(0.0)
        assert single_event_score(10) == pytest.approx(0.0)

    def test_nearby_targets_common_signal(self) -> None:
        assert nearby_targets_common_signal_score(0.0) == pytest.approx(0.0)
        assert nearby_targets_common_signal_score(1.0) == pytest.approx(1.0)

    def test_data_gap_overlap(self) -> None:
        assert data_gap_overlap_score(0.5) == pytest.approx(0.5)

    def test_epoch_match_score(self) -> None:
        assert epoch_match_score(0.0) == pytest.approx(1.0)
        assert epoch_match_score(3.0) == pytest.approx(0.0)

    def test_variability_periodogram_score(self) -> None:
        assert variability_periodogram_score(0.5) == pytest.approx(1.0)
        assert variability_periodogram_score(0.0) == pytest.approx(0.0)

    def test_flare_score(self) -> None:
        assert flare_score(0.0) == pytest.approx(0.0)
        assert flare_score(2.0) == pytest.approx(1.0)

    def test_quasi_periodic_score(self) -> None:
        assert quasi_periodic_score(0.7) == pytest.approx(0.7)

    def test_harmonic_score(self) -> None:
        assert harmonic_score(0.0) == pytest.approx(0.0)
        assert harmonic_score(0.5) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# extract_features
# ---------------------------------------------------------------------------


class TestExtractFeatures:
    def test_minimal_signal_empty_diagnostics(
        self,
        base_signal: CandidateSignal,
        empty_diagnostics: RawDiagnostics,
    ) -> None:
        f = extract_features(base_signal, empty_diagnostics)
        # Always-computed scores are present
        assert f.snr_score is not None
        assert f.log_snr_score is not None
        assert f.transit_count_score is not None
        assert f.duration_plausibility_score is not None
        assert f.large_depth_score is not None
        assert f.companion_radius_too_large_score is not None
        assert f.duration_implausibility_score is not None
        assert f.single_event_score is not None

    def test_missing_diagnostics_yield_none(
        self,
        base_signal: CandidateSignal,
        empty_diagnostics: RawDiagnostics,
    ) -> None:
        f = extract_features(base_signal, empty_diagnostics)
        assert f.depth_consistency_score is None
        assert f.odd_even_mismatch_score is None
        assert f.secondary_eclipse_score is None
        assert f.contamination_score is None
        assert f.centroid_offset_score is None
        assert f.stellar_variability_score is None
        assert f.systematics_overlap_score is None
        assert f.known_object_score is None

    def test_full_diagnostics_populate_all_fields(
        self, base_signal: CandidateSignal
    ) -> None:
        diag = RawDiagnostics(
            individual_depths=(1000.0, 1010.0, 990.0),
            individual_depth_errors=(20.0, 20.0, 20.0),
            individual_durations=(2.5, 2.4, 2.6),
            individual_duration_errors=(0.1, 0.1, 0.1),
            depth_odd_ppm=1000.0,
            err_odd_ppm=15.0,
            depth_even_ppm=990.0,
            err_even_ppm=15.0,
            secondary_snr=1.5,
            ingress_egress_fraction=0.7,
            stellar_radius_rsun=1.0,
            stellar_mass_msun=1.0,
            contamination_ratio=0.10,
            centroid_offset_sigma=0.8,
            nearby_bright_source_count=1,
            nearby_source_magnitude_diff=3.0,
            aperture_edge_proximity=0.2,
            quality_flag_fraction=0.05,
            sector_boundary_fraction=0.0,
            background_excursion_sigma=1.0,
            data_gap_fraction=0.02,
            nearby_targets_common_signal=0.05,
            ls_power_at_period=0.1,
            ls_power_at_harmonics=0.05,
            flare_rate_per_day=0.1,
            quasi_periodic_strength=0.2,
            target_id_matched=False,
            period_match_sigma=5.0,
            epoch_match_sigma=5.0,
            coordinate_match_arcsec=45.0,
        )
        f = extract_features(base_signal, diag)

        assert f.depth_consistency_score is not None
        assert f.duration_consistency_score is not None
        assert f.odd_even_mismatch_score is not None
        assert f.secondary_eclipse_score is not None
        assert f.transit_shape_score is not None
        assert f.v_shape_score is not None
        assert f.contamination_score is not None
        assert f.centroid_offset_score is not None
        assert f.stellar_variability_score is not None
        assert f.systematics_overlap_score is not None
        assert f.known_object_score is not None

    def test_all_scores_in_unit_interval(self, base_signal: CandidateSignal) -> None:
        diag = RawDiagnostics(
            individual_depths=(1000.0, 1000.0, 1000.0),
            individual_depth_errors=(10.0, 10.0, 10.0),
            secondary_snr=3.0,
            ingress_egress_fraction=0.8,
            contamination_ratio=0.15,
            centroid_offset_sigma=1.0,
            quality_flag_fraction=0.1,
            ls_power_at_period=0.2,
        )
        f = extract_features(base_signal, diag)
        for field_name in CandidateFeatures.model_fields:
            val = getattr(f, field_name)
            if val is not None:
                assert 0.0 <= val <= 1.0, f"{field_name} = {val} is outside [0, 1]"

    def test_known_tess_planet_profile(self) -> None:
        """High-SNR, multi-transit, clean signal should produce a sensible feature set."""
        signal = CandidateSignal(
            candidate_id="TIC_TOI_test",
            mission="TESS",
            target_id="TIC 999",
            period_days=5.0,
            epoch_bjd=2458600.0,
            duration_hours=1.8,
            depth_ppm=2000.0,
            transit_count=5,
            snr=15.0,
        )
        diag = RawDiagnostics(
            individual_depths=(2000.0, 1990.0, 2010.0, 2005.0, 1995.0),
            individual_depth_errors=(50.0,) * 5,
            depth_odd_ppm=2002.0,
            err_odd_ppm=40.0,
            depth_even_ppm=1998.0,
            err_even_ppm=40.0,
            secondary_snr=0.5,
            ingress_egress_fraction=0.75,
            contamination_ratio=0.05,
            centroid_offset_sigma=0.3,
        )
        f = extract_features(signal, diag)

        assert f.snr_score is not None and f.snr_score == pytest.approx(1.0)
        assert f.transit_count_score == pytest.approx(1.0)
        assert f.depth_consistency_score is not None and f.depth_consistency_score > 0.8
        assert f.odd_even_mismatch_score is not None and f.odd_even_mismatch_score < 0.2
        assert f.secondary_eclipse_score is not None and f.secondary_eclipse_score < 0.2
        assert f.transit_shape_score is not None and f.transit_shape_score > 0.5


# ---------------------------------------------------------------------------
# depth_scatter_chi2_score
# ---------------------------------------------------------------------------


class TestDepthScatterChi2Score:
    def test_consistent_depths_near_zero(self) -> None:
        # depths match within errors → chi2_reduced ≈ 0 → score near 0
        depths = (1000.0, 1000.0, 1000.0)
        errors = (50.0, 50.0, 50.0)
        s = depth_scatter_chi2_score(depths, errors)
        assert s is not None and s == pytest.approx(0.0)

    def test_highly_scattered_depths_near_one(self) -> None:
        # huge scatter relative to errors → chi2_reduced >> threshold → score 1
        depths = (100.0, 2000.0, 100.0, 2000.0)
        errors = (1.0, 1.0, 1.0, 1.0)
        s = depth_scatter_chi2_score(depths, errors)
        assert s is not None and s == pytest.approx(1.0)

    def test_none_when_single_transit(self) -> None:
        assert depth_scatter_chi2_score((1000.0,), (50.0,)) is None

    def test_none_when_empty(self) -> None:
        assert depth_scatter_chi2_score((), ()) is None

    def test_none_when_zero_error(self) -> None:
        assert depth_scatter_chi2_score((1000.0, 1000.0), (0.0, 50.0)) is None

    def test_moderate_scatter_in_range(self) -> None:
        # chi2_reduced ≈ 0.25 / 3.0 ≈ 0.08 → in (0, 1)
        depths = (1000.0, 1050.0, 950.0)
        errors = (100.0, 100.0, 100.0)
        s = depth_scatter_chi2_score(depths, errors)
        assert s is not None and 0.0 < s < 1.0

    def test_custom_chi2_threshold(self) -> None:
        depths = (1000.0, 1000.0, 1000.0)
        errors = (50.0, 50.0, 50.0)
        # threshold doesn't matter when scatter is zero
        s = depth_scatter_chi2_score(depths, errors, chi2_threshold=1.0)
        assert s is not None and s == pytest.approx(0.0)

    def test_extract_features_propagates_chi2(self) -> None:
        """extract_features should populate depth_scatter_chi2_score."""
        from exo_toolkit.schemas import CandidateSignal  # noqa: PLC0415

        sig = CandidateSignal(
            candidate_id="x",
            mission="TESS",
            target_id="TIC 1",
            period_days=5.0,
            epoch_bjd=2458600.0,
            duration_hours=2.0,
            depth_ppm=1000.0,
            transit_count=3,
            snr=10.0,
        )
        diag = RawDiagnostics(
            individual_depths=(1000.0, 1000.0, 1000.0),
            individual_depth_errors=(50.0, 50.0, 50.0),
        )
        f = extract_features(sig, diag)
        assert f.depth_scatter_chi2_score is not None
        assert f.depth_scatter_chi2_score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# transit_timing_variation_score
# ---------------------------------------------------------------------------


class TestTransitTimingVariationScore:
    _PERIOD = 5.0     # days
    _EPOCH  = 2458600.0  # BJD

    def _midpoints(self, offsets_minutes: list[float]) -> tuple[float, ...]:
        """Build midpoints: exact linear ephemeris + given offsets in minutes."""
        days = [self._EPOCH + n * self._PERIOD for n in range(len(offsets_minutes))]
        return tuple(d + o / 1440.0 for d, o in zip(days, offsets_minutes, strict=True))

    def test_perfect_timing_gives_zero(self) -> None:
        mids = self._midpoints([0.0, 0.0, 0.0])
        s = transit_timing_variation_score(mids, self._PERIOD, self._EPOCH)
        assert s is not None and s == pytest.approx(0.0)

    def test_large_scatter_clips_to_one(self) -> None:
        mids = self._midpoints([100.0, -100.0, 100.0, -100.0])
        s = transit_timing_variation_score(mids, self._PERIOD, self._EPOCH)
        assert s is not None and s == pytest.approx(1.0)

    def test_none_when_single_midpoint(self) -> None:
        mids = (self._EPOCH,)
        assert transit_timing_variation_score(mids, self._PERIOD, self._EPOCH) is None

    def test_none_when_empty(self) -> None:
        assert transit_timing_variation_score((), self._PERIOD, self._EPOCH) is None

    def test_moderate_scatter_in_range(self) -> None:
        # rms = 5 min / 10 min threshold → 0.5
        mids = self._midpoints([5.0, -5.0])
        s = transit_timing_variation_score(mids, self._PERIOD, self._EPOCH, rms_threshold_minutes=10.0)
        assert s is not None and pytest.approx(s, abs=0.01) == 0.5

    def test_custom_threshold_scales_score(self) -> None:
        mids = self._midpoints([5.0, -5.0])
        s_default = transit_timing_variation_score(mids, self._PERIOD, self._EPOCH, rms_threshold_minutes=10.0)
        s_tight   = transit_timing_variation_score(mids, self._PERIOD, self._EPOCH, rms_threshold_minutes=5.0)
        assert s_tight is not None and s_default is not None
        assert s_tight > s_default

    def test_output_in_zero_one(self) -> None:
        mids = self._midpoints([3.0, -2.0, 1.0])
        s = transit_timing_variation_score(mids, self._PERIOD, self._EPOCH)
        assert s is not None and 0.0 <= s <= 1.0

    def test_extract_features_propagates_ttv(self) -> None:
        sig = CandidateSignal(
            candidate_id="x",
            mission="TESS",
            target_id="TIC 1",
            period_days=self._PERIOD,
            epoch_bjd=self._EPOCH,
            duration_hours=2.0,
            depth_ppm=1000.0,
            transit_count=3,
            snr=10.0,
        )
        mids = self._midpoints([0.0, 0.0, 0.0])
        diag = RawDiagnostics(individual_transit_midpoints=mids)
        f = extract_features(sig, diag)
        assert f.transit_timing_variation_score is not None
        assert f.transit_timing_variation_score == pytest.approx(0.0)

    def test_none_midpoints_gives_none_feature(self) -> None:
        sig = CandidateSignal(
            candidate_id="x",
            mission="TESS",
            target_id="TIC 1",
            period_days=self._PERIOD,
            epoch_bjd=self._EPOCH,
            duration_hours=2.0,
            depth_ppm=1000.0,
            transit_count=3,
            snr=10.0,
        )
        f = extract_features(sig, RawDiagnostics())
        assert f.transit_timing_variation_score is None

    def test_two_midpoints_sufficient(self) -> None:
        mids = self._midpoints([0.0, 0.0])
        s = transit_timing_variation_score(mids, self._PERIOD, self._EPOCH)
        assert s is not None
