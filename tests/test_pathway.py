"""Tests for exo_toolkit.pathway."""
import pytest

from exo_toolkit.pathway import classify_submission_pathway
from exo_toolkit.schemas import (
    CandidateFeatures,
    CandidateScores,
    CandidateSignal,
    HypothesisPosterior,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _posterior(
    *,
    planet: float = 0.70,
    eb: float = 0.08,
    beb: float = 0.07,
    sv: float = 0.06,
    ia: float = 0.05,
    ko: float = 0.04,
) -> HypothesisPosterior:
    """Build a HypothesisPosterior that sums to 1.0."""
    total = planet + eb + beb + sv + ia + ko
    scale = 1.0 / total
    return HypothesisPosterior(
        planet_candidate=round(planet * scale, 9),
        eclipsing_binary=round(eb * scale, 9),
        background_eclipsing_binary=round(beb * scale, 9),
        stellar_variability=round(sv * scale, 9),
        instrumental_artifact=round(ia * scale, 9),
        known_object=round(ko * scale, 9),
    )


def _scores(
    *,
    fpp: float = 0.25,
    dc: float = 0.80,
    ns: float = 0.95,
    hi: float = 0.30,
    fv: float = 0.55,
    sr: float = 0.60,
) -> CandidateScores:
    return CandidateScores(
        false_positive_probability=fpp,
        detection_confidence=dc,
        novelty_score=ns,
        habitability_interest=hi,
        followup_value=fv,
        submission_readiness=sr,
    )


def _signal(
    *,
    mission: str = "TESS",
    snr: float = 10.0,
    transit_count: int = 3,
    period_days: float = 10.0,
) -> CandidateSignal:
    return CandidateSignal(
        candidate_id="TIC_1_s01",
        mission=mission,  # type: ignore[arg-type]
        target_id="TIC 1",
        period_days=period_days,
        epoch_bjd=2459000.0,
        duration_hours=2.5,
        depth_ppm=1000.0,
        transit_count=transit_count,
        snr=snr,
    )


def _tfop_features() -> CandidateFeatures:
    """Features that satisfy all tfop_ready gate conditions."""
    return CandidateFeatures(
        contamination_score=0.10,
        secondary_eclipse_score=0.05,
        odd_even_mismatch_score=0.05,
    )


# ---------------------------------------------------------------------------
# Priority gate ordering
# ---------------------------------------------------------------------------


class TestGatePriority:
    def test_known_object_takes_priority_over_everything(self) -> None:
        # Even with low fpp, high transit count, and TESS — known_object wins
        post = _posterior(planet=0.05, ko=0.82, eb=0.04, beb=0.03, sv=0.03, ia=0.03)
        result = classify_submission_pathway(
            _signal(),
            _tfop_features(),
            post,
            _scores(fpp=0.10),
            provenance_score=0.90,
        )
        assert result == "known_object_annotation"

    def test_high_fpp_takes_priority_over_mission_routes(self) -> None:
        # fpp >= 0.70 must short-circuit before TESS/Kepler branch
        post = _posterior(planet=0.20, ko=0.02, eb=0.30, beb=0.20, sv=0.15, ia=0.13)
        result = classify_submission_pathway(
            _signal(transit_count=5),
            _tfop_features(),
            post,
            _scores(fpp=0.75, dc=0.80),
        )
        assert result == "github_only_reproducibility"

    def test_single_transit_routed_to_planet_hunters_before_mission_check(
        self,
    ) -> None:
        result = classify_submission_pathway(
            _signal(transit_count=1),
            _tfop_features(),
            _posterior(),
            _scores(fpp=0.25),
        )
        assert result == "planet_hunters_discussion"

    def test_two_transits_clears_single_transit_gate(self) -> None:
        # transit_count == 2 should NOT route to planet_hunters via that gate
        result = classify_submission_pathway(
            _signal(transit_count=2),
            _tfop_features(),
            _posterior(),
            _scores(fpp=0.25),
            provenance_score=0.90,
        )
        assert result == "tfop_ready"


# ---------------------------------------------------------------------------
# TESS pathways
# ---------------------------------------------------------------------------


class TestTESS:
    def test_tfop_ready_when_all_conditions_met(self) -> None:
        result = classify_submission_pathway(
            _signal(snr=9.0, transit_count=3),
            _tfop_features(),
            _posterior(planet=0.75, ko=0.02, eb=0.08, beb=0.06, sv=0.05, ia=0.04),
            _scores(fpp=0.25, dc=0.85),
            provenance_score=0.90,
        )
        assert result == "tfop_ready"

    def test_tfop_blocked_by_low_planet_posterior(self) -> None:
        post = _posterior(planet=0.55, ko=0.02, eb=0.18, beb=0.10, sv=0.08, ia=0.07)
        result = classify_submission_pathway(
            _signal(),
            _tfop_features(),
            post,
            _scores(fpp=0.25, dc=0.80),
            provenance_score=0.90,
        )
        assert result != "tfop_ready"

    def test_tfop_blocked_by_high_fpp(self) -> None:
        # fpp = 0.36 just over threshold
        result = classify_submission_pathway(
            _signal(),
            _tfop_features(),
            _posterior(),
            _scores(fpp=0.36, dc=0.80),
            provenance_score=0.90,
        )
        assert result != "tfop_ready"

    def test_tfop_blocked_by_low_snr(self) -> None:
        result = classify_submission_pathway(
            _signal(snr=7.5),
            _tfop_features(),
            _posterior(),
            _scores(fpp=0.25, dc=0.80),
            provenance_score=0.90,
        )
        assert result != "tfop_ready"

    def test_tfop_blocked_when_contamination_is_none(self) -> None:
        features = CandidateFeatures(
            contamination_score=None,  # not computed
            secondary_eclipse_score=0.05,
            odd_even_mismatch_score=0.05,
        )
        result = classify_submission_pathway(
            _signal(),
            features,
            _posterior(),
            _scores(fpp=0.25),
            provenance_score=0.90,
        )
        assert result != "tfop_ready"

    def test_tfop_blocked_by_high_contamination(self) -> None:
        features = CandidateFeatures(
            contamination_score=0.55,
            secondary_eclipse_score=0.05,
            odd_even_mismatch_score=0.05,
        )
        result = classify_submission_pathway(
            _signal(),
            features,
            _posterior(),
            _scores(fpp=0.25),
            provenance_score=0.90,
        )
        assert result != "tfop_ready"

    def test_tfop_blocked_when_secondary_eclipse_is_none(self) -> None:
        features = CandidateFeatures(
            contamination_score=0.10,
            secondary_eclipse_score=None,
            odd_even_mismatch_score=0.05,
        )
        result = classify_submission_pathway(
            _signal(),
            features,
            _posterior(),
            _scores(fpp=0.25),
            provenance_score=0.90,
        )
        assert result != "tfop_ready"

    def test_tfop_blocked_by_strong_secondary_eclipse(self) -> None:
        features = CandidateFeatures(
            contamination_score=0.10,
            secondary_eclipse_score=0.45,
            odd_even_mismatch_score=0.05,
        )
        result = classify_submission_pathway(
            _signal(),
            features,
            _posterior(),
            _scores(fpp=0.25),
            provenance_score=0.90,
        )
        assert result != "tfop_ready"

    def test_tfop_blocked_when_odd_even_is_none(self) -> None:
        features = CandidateFeatures(
            contamination_score=0.10,
            secondary_eclipse_score=0.05,
            odd_even_mismatch_score=None,
        )
        result = classify_submission_pathway(
            _signal(),
            features,
            _posterior(),
            _scores(fpp=0.25),
            provenance_score=0.90,
        )
        assert result != "tfop_ready"

    def test_tfop_blocked_by_strong_odd_even_mismatch(self) -> None:
        features = CandidateFeatures(
            contamination_score=0.10,
            secondary_eclipse_score=0.05,
            odd_even_mismatch_score=0.45,
        )
        result = classify_submission_pathway(
            _signal(),
            features,
            _posterior(),
            _scores(fpp=0.25),
            provenance_score=0.90,
        )
        assert result != "tfop_ready"

    def test_tfop_blocked_by_low_provenance_score(self) -> None:
        result = classify_submission_pathway(
            _signal(),
            _tfop_features(),
            _posterior(),
            _scores(fpp=0.25),
            provenance_score=0.70,  # below _TFOP_PROVENANCE = 0.80
        )
        assert result != "tfop_ready"

    def test_tfop_blocked_by_missing_provenance_score_default(self) -> None:
        # Default provenance_score=0.0 must prevent tfop_ready
        result = classify_submission_pathway(
            _signal(),
            _tfop_features(),
            _posterior(),
            _scores(fpp=0.25),
        )
        assert result != "tfop_ready"

    def test_planet_hunters_when_dc_high_but_not_tfop(self) -> None:
        # fpp too high for tfop but dc >= 0.45
        post = _posterior(planet=0.50, ko=0.02, eb=0.18, beb=0.10, sv=0.10, ia=0.10)
        result = classify_submission_pathway(
            _signal(),
            CandidateFeatures(),
            post,
            _scores(fpp=0.50, dc=0.70),
        )
        assert result == "planet_hunters_discussion"

    def test_github_only_when_dc_low_and_not_tfop(self) -> None:
        post = _posterior(planet=0.50, ko=0.02, eb=0.18, beb=0.10, sv=0.10, ia=0.10)
        result = classify_submission_pathway(
            _signal(),
            CandidateFeatures(),
            post,
            _scores(fpp=0.50, dc=0.30),
        )
        assert result == "github_only_reproducibility"

    def test_planet_hunters_when_transit_count_is_two(self) -> None:
        # 2 transits, TESS, but not tfop quality → planet_hunters via dc gate
        post = _posterior(planet=0.50, ko=0.02, eb=0.18, beb=0.10, sv=0.10, ia=0.10)
        result = classify_submission_pathway(
            _signal(transit_count=2),
            CandidateFeatures(),
            post,
            _scores(fpp=0.50, dc=0.60),
        )
        assert result == "planet_hunters_discussion"


# ---------------------------------------------------------------------------
# Kepler / K2 pathways
# ---------------------------------------------------------------------------


class TestKeplerK2:
    def test_kepler_archive_when_conditions_met(self) -> None:
        post = _posterior(planet=0.75, ko=0.02, eb=0.08, beb=0.06, sv=0.05, ia=0.04)
        result = classify_submission_pathway(
            _signal(mission="Kepler"),
            CandidateFeatures(),
            post,
            _scores(fpp=0.25, ns=0.90),
        )
        assert result == "kepler_archive_candidate"

    def test_k2_archive_when_conditions_met(self) -> None:
        post = _posterior(planet=0.70, ko=0.02, eb=0.10, beb=0.08, sv=0.05, ia=0.05)
        result = classify_submission_pathway(
            _signal(mission="K2"),
            CandidateFeatures(),
            post,
            _scores(fpp=0.30, ns=0.75),
        )
        assert result == "kepler_archive_candidate"

    def test_kepler_github_only_when_low_planet_posterior(self) -> None:
        post = _posterior(planet=0.55, ko=0.02, eb=0.18, beb=0.10, sv=0.08, ia=0.07)
        result = classify_submission_pathway(
            _signal(mission="Kepler"),
            CandidateFeatures(),
            post,
            _scores(fpp=0.25, ns=0.90),
        )
        assert result == "github_only_reproducibility"

    def test_kepler_github_only_when_low_novelty(self) -> None:
        post = _posterior(planet=0.70, ko=0.02, eb=0.10, beb=0.08, sv=0.05, ia=0.05)
        result = classify_submission_pathway(
            _signal(mission="Kepler"),
            CandidateFeatures(),
            post,
            _scores(fpp=0.25, ns=0.60),  # ns below 0.70 threshold
        )
        assert result == "github_only_reproducibility"

    def test_kepler_github_only_when_fpp_too_high(self) -> None:
        post = _posterior(planet=0.70, ko=0.02, eb=0.10, beb=0.08, sv=0.05, ia=0.05)
        result = classify_submission_pathway(
            _signal(mission="Kepler"),
            CandidateFeatures(),
            post,
            _scores(fpp=0.40, ns=0.80),  # fpp above 0.35 threshold
        )
        assert result == "github_only_reproducibility"

    def test_k2_github_only_when_conditions_not_met(self) -> None:
        post = _posterior(planet=0.60, ko=0.02, eb=0.15, beb=0.10, sv=0.07, ia=0.06)
        result = classify_submission_pathway(
            _signal(mission="K2"),
            CandidateFeatures(),
            post,
            _scores(fpp=0.40, ns=0.65),
        )
        assert result == "github_only_reproducibility"


# ---------------------------------------------------------------------------
# Threshold boundary conditions
# ---------------------------------------------------------------------------


class TestBoundaryConditions:
    def test_known_object_threshold_is_inclusive(self) -> None:
        post = _posterior(planet=0.10, ko=0.80, eb=0.03, beb=0.03, sv=0.02, ia=0.02)
        result = classify_submission_pathway(
            _signal(),
            CandidateFeatures(),
            post,
            _scores(fpp=0.10),
        )
        assert result == "known_object_annotation"

    def test_just_below_known_object_threshold_is_not_annotation(self) -> None:
        # ko = 0.79 — should not trigger known_object_annotation
        post = _posterior(planet=0.10, ko=0.79, eb=0.04, beb=0.03, sv=0.02, ia=0.02)
        result = classify_submission_pathway(
            _signal(),
            CandidateFeatures(),
            post,
            _scores(fpp=0.10),
        )
        assert result != "known_object_annotation"

    def test_fpp_exactly_at_high_threshold_is_github_only(self) -> None:
        post = _posterior(planet=0.30, ko=0.02, eb=0.28, beb=0.15, sv=0.13, ia=0.12)
        result = classify_submission_pathway(
            _signal(),
            CandidateFeatures(),
            post,
            _scores(fpp=0.70),
        )
        assert result == "github_only_reproducibility"

    def test_tfop_planet_posterior_at_exact_threshold(self) -> None:
        # planet_candidate = 0.65 exactly → should qualify
        post = _posterior(planet=0.65, ko=0.02, eb=0.13, beb=0.08, sv=0.06, ia=0.06)
        result = classify_submission_pathway(
            _signal(snr=9.0),
            _tfop_features(),
            post,
            _scores(fpp=0.25, dc=0.85),
            provenance_score=0.90,
        )
        assert result == "tfop_ready"

    def test_tfop_fpp_at_exact_threshold(self) -> None:
        # fpp = 0.35 exactly → should qualify (<=)
        result = classify_submission_pathway(
            _signal(snr=9.0),
            _tfop_features(),
            _posterior(),
            _scores(fpp=0.35, dc=0.85),
            provenance_score=0.90,
        )
        assert result == "tfop_ready"

    def test_tfop_snr_at_exact_threshold(self) -> None:
        # snr = 8.0 exactly → should qualify (>=)
        result = classify_submission_pathway(
            _signal(snr=8.0),
            _tfop_features(),
            _posterior(),
            _scores(fpp=0.25, dc=0.85),
            provenance_score=0.90,
        )
        assert result == "tfop_ready"

    def test_dc_at_exact_planet_hunters_threshold(self) -> None:
        # dc = 0.45 exactly → planet_hunters_discussion (>=)
        post = _posterior(planet=0.50, ko=0.02, eb=0.18, beb=0.10, sv=0.10, ia=0.10)
        result = classify_submission_pathway(
            _signal(),
            CandidateFeatures(),
            post,
            _scores(fpp=0.50, dc=0.45),
        )
        assert result == "planet_hunters_discussion"

    def test_kepler_novelty_at_exact_threshold(self) -> None:
        post = _posterior(planet=0.70, ko=0.02, eb=0.10, beb=0.08, sv=0.05, ia=0.05)
        result = classify_submission_pathway(
            _signal(mission="Kepler"),
            CandidateFeatures(),
            post,
            _scores(fpp=0.25, ns=0.70),  # exactly at threshold
        )
        assert result == "kepler_archive_candidate"


# ---------------------------------------------------------------------------
# Empty features (all None)
# ---------------------------------------------------------------------------


class TestEmptyFeatures:
    def test_tess_with_no_features_never_reaches_tfop(self) -> None:
        result = classify_submission_pathway(
            _signal(),
            CandidateFeatures(),
            _posterior(),
            _scores(fpp=0.25, dc=0.80),
            provenance_score=0.90,
        )
        # None features block tfop_ready; dc >= 0.45 → planet_hunters
        assert result == "planet_hunters_discussion"

    def test_kepler_with_no_features_still_routes_via_posterior(self) -> None:
        post = _posterior(planet=0.70, ko=0.02, eb=0.10, beb=0.08, sv=0.05, ia=0.05)
        result = classify_submission_pathway(
            _signal(mission="Kepler"),
            CandidateFeatures(),
            post,
            _scores(fpp=0.25, ns=0.80),
        )
        assert result == "kepler_archive_candidate"


# ---------------------------------------------------------------------------
# Parametric tfop_ready gate tests (Milestone 12k)
# ---------------------------------------------------------------------------


def _all_passing_kwargs() -> dict:
    """All inputs that satisfy every tfop_ready gate simultaneously."""
    return {
        "signal": _signal(snr=9.0, transit_count=3),
        "features": _tfop_features(),
        "posterior": _posterior(planet=0.70, ko=0.02, eb=0.12, beb=0.06, sv=0.05, ia=0.05),
        "scores": _scores(fpp=0.25, dc=0.85),
        "provenance_score": 0.90,
    }


class TestTfopReadyGatesParametric:
    @pytest.mark.parametrize(
        "gate_name,override",
        [
            (
                "planet_posterior",
                {"posterior": _posterior(
                    planet=0.40, ko=0.02, eb=0.30, beb=0.12, sv=0.08, ia=0.08
                )},
            ),
            ("fpp", {"scores": _scores(fpp=0.40, dc=0.85)}),
            ("snr", {"signal": _signal(snr=7.5, transit_count=3)}),
            ("transit_count", {"signal": _signal(snr=9.0, transit_count=1)}),
            (
                "contamination_high",
                {"features": CandidateFeatures(
                    contamination_score=0.60,
                    secondary_eclipse_score=0.05,
                    odd_even_mismatch_score=0.05,
                )},
            ),
            ("provenance_score", {"provenance_score": 0.50}),
            (
                "contamination_none",
                {"features": CandidateFeatures(
                    secondary_eclipse_score=0.05, odd_even_mismatch_score=0.05
                )},
            ),
            (
                "secondary_none",
                {"features": CandidateFeatures(
                    contamination_score=0.10, odd_even_mismatch_score=0.05
                )},
            ),
            (
                "odd_even_none",
                {"features": CandidateFeatures(
                    contamination_score=0.10, secondary_eclipse_score=0.05
                )},
            ),
        ],
    )
    def test_failing_gate_blocks_tfop(self, gate_name: str, override: dict) -> None:
        kwargs = _all_passing_kwargs()
        kwargs.update(override)
        result = classify_submission_pathway(**kwargs)
        assert result != "tfop_ready", f"Gate '{gate_name}' should have blocked tfop_ready"

    def test_all_gates_passing_gives_tfop(self) -> None:
        result = classify_submission_pathway(**_all_passing_kwargs())
        assert result == "tfop_ready"


# ---------------------------------------------------------------------------
# Extended branch coverage (Milestone 13l)
# ---------------------------------------------------------------------------


class TestAllPathwayBranches:
    """Cover every return-value branch of classify_submission_pathway."""

    def test_known_object_annotation_returned(self) -> None:
        post = _posterior(planet=0.05, ko=0.82, eb=0.04, beb=0.03, sv=0.03, ia=0.03)
        result = classify_submission_pathway(
            _signal(), CandidateFeatures(), post, _scores(fpp=0.05),
        )
        assert result == "known_object_annotation"

    def test_github_only_returned_for_high_fpp(self) -> None:
        post = _posterior(planet=0.20, ko=0.02, eb=0.28, beb=0.20, sv=0.15, ia=0.15)
        result = classify_submission_pathway(
            _signal(transit_count=5), CandidateFeatures(), post, _scores(fpp=0.75),
        )
        assert result == "github_only_reproducibility"

    def test_planet_hunters_returned_for_single_transit(self) -> None:
        result = classify_submission_pathway(
            _signal(transit_count=1), _tfop_features(), _posterior(), _scores(fpp=0.25),
        )
        assert result == "planet_hunters_discussion"

    def test_tfop_ready_returned_for_tess_all_conditions(self) -> None:
        result = classify_submission_pathway(
            _signal(snr=9.0, transit_count=3),
            _tfop_features(),
            _posterior(planet=0.70, ko=0.02, eb=0.12, beb=0.06, sv=0.05, ia=0.05),
            _scores(fpp=0.25, dc=0.85),
            provenance_score=0.90,
        )
        assert result == "tfop_ready"

    def test_planet_hunters_tess_good_dc_missing_features(self) -> None:
        result = classify_submission_pathway(
            _signal(transit_count=3),
            CandidateFeatures(),
            _posterior(planet=0.60, ko=0.02, eb=0.15, beb=0.10, sv=0.08, ia=0.05),
            _scores(fpp=0.25, dc=0.55),
        )
        assert result == "planet_hunters_discussion"

    def test_github_only_tess_low_dc(self) -> None:
        result = classify_submission_pathway(
            _signal(transit_count=3),
            CandidateFeatures(),
            _posterior(planet=0.60, ko=0.02, eb=0.15, beb=0.10, sv=0.08, ia=0.05),
            _scores(fpp=0.25, dc=0.30),
        )
        assert result == "github_only_reproducibility"

    def test_kepler_archive_returned_for_kepler(self) -> None:
        post = _posterior(planet=0.70, ko=0.02, eb=0.10, beb=0.08, sv=0.05, ia=0.05)
        result = classify_submission_pathway(
            _signal(mission="Kepler"), CandidateFeatures(), post, _scores(fpp=0.25, ns=0.80),
        )
        assert result == "kepler_archive_candidate"

    def test_kepler_archive_returned_for_k2(self) -> None:
        post = _posterior(planet=0.70, ko=0.02, eb=0.10, beb=0.08, sv=0.05, ia=0.05)
        result = classify_submission_pathway(
            _signal(mission="K2"), CandidateFeatures(), post, _scores(fpp=0.25, ns=0.80),
        )
        assert result == "kepler_archive_candidate"

    def test_github_only_kepler_low_planet_posterior(self) -> None:
        post = _posterior(planet=0.55, ko=0.02, eb=0.18, beb=0.10, sv=0.08, ia=0.07)
        result = classify_submission_pathway(
            _signal(mission="Kepler"), CandidateFeatures(), post, _scores(fpp=0.25, ns=0.80),
        )
        assert result == "github_only_reproducibility"

    def test_github_only_kepler_low_novelty(self) -> None:
        post = _posterior(planet=0.70, ko=0.02, eb=0.10, beb=0.08, sv=0.05, ia=0.05)
        result = classify_submission_pathway(
            _signal(mission="Kepler"), CandidateFeatures(), post, _scores(fpp=0.25, ns=0.50),
        )
        assert result == "github_only_reproducibility"

    def test_github_only_kepler_high_fpp(self) -> None:
        post = _posterior(planet=0.70, ko=0.02, eb=0.10, beb=0.08, sv=0.05, ia=0.05)
        result = classify_submission_pathway(
            _signal(mission="Kepler"), CandidateFeatures(), post, _scores(fpp=0.40, ns=0.80),
        )
        assert result == "github_only_reproducibility"

    def test_low_provenance_blocks_tfop(self) -> None:
        result = classify_submission_pathway(
            _signal(snr=9.0, transit_count=3),
            _tfop_features(),
            _posterior(planet=0.70, ko=0.02, eb=0.12, beb=0.06, sv=0.05, ia=0.05),
            _scores(fpp=0.25, dc=0.85),
            provenance_score=0.50,
        )
        assert result != "tfop_ready"

    def test_known_object_beats_tfop_conditions(self) -> None:
        post = _posterior(planet=0.10, ko=0.82, eb=0.03, beb=0.02, sv=0.02, ia=0.01)
        result = classify_submission_pathway(
            _signal(snr=9.0, transit_count=3),
            _tfop_features(),
            post,
            _scores(fpp=0.10, dc=0.85),
            provenance_score=0.90,
        )
        assert result == "known_object_annotation"

    def test_fpp_gate_fires_before_mission_routing(self) -> None:
        post = _posterior(planet=0.50, ko=0.02, eb=0.20, beb=0.12, sv=0.08, ia=0.08)
        result = classify_submission_pathway(
            _signal(transit_count=5),
            _tfop_features(),
            post,
            _scores(fpp=0.71, dc=0.90),
            provenance_score=0.95,
        )
        assert result == "github_only_reproducibility"

    def test_transit_count_gate_fires_before_mission_routing(self) -> None:
        post = _posterior(planet=0.70, ko=0.02, eb=0.10, beb=0.08, sv=0.05, ia=0.05)
        result = classify_submission_pathway(
            _signal(transit_count=1),
            _tfop_features(),
            post,
            _scores(fpp=0.20, dc=0.90),
            provenance_score=0.95,
        )
        assert result == "planet_hunters_discussion"
