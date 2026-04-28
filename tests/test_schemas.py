"""Tests for exo_toolkit.schemas."""
import pytest
from pydantic import ValidationError

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
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def valid_signal() -> CandidateSignal:
    return CandidateSignal(
        candidate_id="TIC_123456789_signal_01",
        mission="TESS",
        target_id="TIC 123456789",
        period_days=12.345,
        epoch_bjd=2459000.123,
        duration_hours=2.4,
        depth_ppm=850.0,
        transit_count=4,
        snr=11.2,
    )


@pytest.fixture
def uniform_posterior() -> HypothesisPosterior:
    """Each hypothesis gets equal weight (1/6 ≈ 0.1667)."""
    p = round(1.0 / 6.0, 10)
    # Distribute any floating-point remainder on planet_candidate
    remainder = round(1.0 - p * 5, 10)
    return HypothesisPosterior(
        planet_candidate=remainder,
        eclipsing_binary=p,
        background_eclipsing_binary=p,
        stellar_variability=p,
        instrumental_artifact=p,
        known_object=p,
    )


@pytest.fixture
def minimal_metadata() -> ScoringMetadata:
    return ScoringMetadata(
        scoring_model_name="bayesian_logscore_v0",
        scoring_model_version="0.1",
        scoring_model_commit="abc123",
        threshold_config_hash="def456",
    )


@pytest.fixture
def minimal_scored_candidate(
    valid_signal: CandidateSignal,
    uniform_posterior: HypothesisPosterior,
    minimal_metadata: ScoringMetadata,
) -> ScoredCandidate:
    return ScoredCandidate(
        signal=valid_signal,
        features=CandidateFeatures(),
        posterior=uniform_posterior,
        scores=CandidateScores(),
        recommended_pathway="github_only_reproducibility",
        explanation=CandidateExplanation(),
        metadata=minimal_metadata,
    )


# ---------------------------------------------------------------------------
# CandidateSignal
# ---------------------------------------------------------------------------


class TestCandidateSignal:
    def test_valid_construction(self, valid_signal: CandidateSignal) -> None:
        assert valid_signal.candidate_id == "TIC_123456789_signal_01"
        assert valid_signal.mission == "TESS"
        assert valid_signal.snr == 11.2
        assert valid_signal.transit_count == 4

    def test_kepler_and_k2_missions_accepted(self) -> None:
        for mission in ("Kepler", "K2"):
            s = CandidateSignal(
                candidate_id="x",
                mission=mission,  # type: ignore[arg-type]
                target_id="y",
                period_days=1.0,
                epoch_bjd=0.0,
                duration_hours=1.0,
                depth_ppm=100.0,
                transit_count=1,
                snr=5.0,
            )
            assert s.mission == mission

    def test_rejects_invalid_mission(self) -> None:
        with pytest.raises(ValidationError):
            CandidateSignal(
                candidate_id="x",
                mission="HST",  # type: ignore[arg-type]
                target_id="y",
                period_days=1.0,
                epoch_bjd=0.0,
                duration_hours=1.0,
                depth_ppm=100.0,
                transit_count=1,
                snr=5.0,
            )

    def test_rejects_zero_period(self) -> None:
        with pytest.raises(ValidationError):
            CandidateSignal(
                candidate_id="x",
                mission="TESS",
                target_id="y",
                period_days=0.0,
                epoch_bjd=0.0,
                duration_hours=1.0,
                depth_ppm=100.0,
                transit_count=1,
                snr=5.0,
            )

    def test_rejects_negative_period(self) -> None:
        with pytest.raises(ValidationError):
            CandidateSignal(
                candidate_id="x",
                mission="TESS",
                target_id="y",
                period_days=-1.0,
                epoch_bjd=0.0,
                duration_hours=1.0,
                depth_ppm=100.0,
                transit_count=1,
                snr=5.0,
            )

    def test_rejects_zero_transit_count(self) -> None:
        with pytest.raises(ValidationError):
            CandidateSignal(
                candidate_id="x",
                mission="TESS",
                target_id="y",
                period_days=1.0,
                epoch_bjd=0.0,
                duration_hours=1.0,
                depth_ppm=100.0,
                transit_count=0,
                snr=5.0,
            )

    def test_rejects_negative_snr(self) -> None:
        with pytest.raises(ValidationError):
            CandidateSignal(
                candidate_id="x",
                mission="TESS",
                target_id="y",
                period_days=1.0,
                epoch_bjd=0.0,
                duration_hours=1.0,
                depth_ppm=100.0,
                transit_count=1,
                snr=-1.0,
            )

    def test_frozen(self, valid_signal: CandidateSignal) -> None:
        with pytest.raises(ValidationError):
            valid_signal.snr = 99.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# CandidateFeatures
# ---------------------------------------------------------------------------


class TestCandidateFeatures:
    def test_all_none_by_default(self) -> None:
        f = CandidateFeatures()
        for field_name in CandidateFeatures.model_fields:
            assert getattr(f, field_name) is None, f"{field_name} should default to None"

    def test_accepts_valid_scores(self) -> None:
        f = CandidateFeatures(snr_score=0.75, odd_even_mismatch_score=0.1)
        assert f.snr_score == 0.75
        assert f.odd_even_mismatch_score == 0.1

    def test_accepts_boundary_values(self) -> None:
        f = CandidateFeatures(snr_score=0.0, contamination_score=1.0)
        assert f.snr_score == 0.0
        assert f.contamination_score == 1.0

    def test_rejects_score_above_one(self) -> None:
        with pytest.raises(ValidationError):
            CandidateFeatures(snr_score=1.01)

    def test_rejects_negative_score(self) -> None:
        with pytest.raises(ValidationError):
            CandidateFeatures(contamination_score=-0.01)

    def test_frozen(self) -> None:
        f = CandidateFeatures(snr_score=0.5)
        with pytest.raises(ValidationError):
            f.snr_score = 0.9  # type: ignore[misc]


# ---------------------------------------------------------------------------
# HypothesisPosterior
# ---------------------------------------------------------------------------


class TestHypothesisPosterior:
    def test_uniform_posterior_sums_to_one(
        self, uniform_posterior: HypothesisPosterior
    ) -> None:
        total = (
            uniform_posterior.planet_candidate
            + uniform_posterior.eclipsing_binary
            + uniform_posterior.background_eclipsing_binary
            + uniform_posterior.stellar_variability
            + uniform_posterior.instrumental_artifact
            + uniform_posterior.known_object
        )
        assert abs(total - 1.0) < 1e-9

    def test_valid_uneven_posterior(self) -> None:
        p = HypothesisPosterior(
            planet_candidate=0.70,
            eclipsing_binary=0.10,
            background_eclipsing_binary=0.08,
            stellar_variability=0.07,
            instrumental_artifact=0.03,
            known_object=0.02,
        )
        assert p.planet_candidate == 0.70

    def test_rejects_probabilities_not_summing_to_one(self) -> None:
        with pytest.raises(ValidationError, match="sum"):
            HypothesisPosterior(
                planet_candidate=0.50,
                eclipsing_binary=0.50,
                background_eclipsing_binary=0.50,
                stellar_variability=0.00,
                instrumental_artifact=0.00,
                known_object=0.00,
            )

    def test_rejects_probability_above_one(self) -> None:
        with pytest.raises(ValidationError):
            HypothesisPosterior(planet_candidate=1.1)

    def test_rejects_negative_probability(self) -> None:
        with pytest.raises(ValidationError):
            HypothesisPosterior(planet_candidate=-0.1)

    def test_frozen(self, uniform_posterior: HypothesisPosterior) -> None:
        with pytest.raises(ValidationError):
            uniform_posterior.planet_candidate = 0.5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# CandidateScores
# ---------------------------------------------------------------------------


class TestCandidateScores:
    def test_defaults_to_zero(self) -> None:
        s = CandidateScores()
        assert s.false_positive_probability == 0.0
        assert s.detection_confidence == 0.0
        assert s.submission_readiness == 0.0

    def test_accepts_valid_scores(self) -> None:
        s = CandidateScores(
            false_positive_probability=0.35,
            detection_confidence=0.80,
            novelty_score=0.90,
        )
        assert s.false_positive_probability == 0.35

    def test_rejects_score_above_one(self) -> None:
        with pytest.raises(ValidationError):
            CandidateScores(detection_confidence=1.01)

    def test_rejects_negative_score(self) -> None:
        with pytest.raises(ValidationError):
            CandidateScores(novelty_score=-0.01)


# ---------------------------------------------------------------------------
# CandidateExplanation
# ---------------------------------------------------------------------------


class TestCandidateExplanation:
    def test_empty_by_default(self) -> None:
        e = CandidateExplanation()
        assert e.positive_evidence == ()
        assert e.negative_evidence == ()
        assert e.blocking_issues == ()

    def test_construction_with_evidence(self) -> None:
        e = CandidateExplanation(
            positive_evidence=("SNR = 11.2 exceeds strong-signal threshold",),
            negative_evidence=("Nearby Gaia source within aperture",),
            blocking_issues=("No centroid analysis available",),
        )
        assert len(e.positive_evidence) == 1
        assert len(e.negative_evidence) == 1
        assert len(e.blocking_issues) == 1

    def test_frozen(self) -> None:
        e = CandidateExplanation(positive_evidence=("SNR is strong",))
        with pytest.raises(ValidationError):
            e.positive_evidence = ("something else",)  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ScoringMetadata
# ---------------------------------------------------------------------------


class TestScoringMetadata:
    def test_construction(self, minimal_metadata: ScoringMetadata) -> None:
        assert minimal_metadata.scoring_model_name == "bayesian_logscore_v0"
        assert minimal_metadata.scoring_model_version == "0.1"

    def test_frozen(self, minimal_metadata: ScoringMetadata) -> None:
        with pytest.raises(ValidationError):
            minimal_metadata.scoring_model_version = "9.9"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ScoredCandidate
# ---------------------------------------------------------------------------


class TestScoredCandidate:
    def test_construction(self, minimal_scored_candidate: ScoredCandidate) -> None:
        c = minimal_scored_candidate
        assert c.signal.mission == "TESS"
        assert c.recommended_pathway == "github_only_reproducibility"
        assert c.secondary_pathway is None

    def test_secondary_pathway_accepts_valid_value(
        self,
        valid_signal: CandidateSignal,
        uniform_posterior: HypothesisPosterior,
        minimal_metadata: ScoringMetadata,
    ) -> None:
        c = ScoredCandidate(
            signal=valid_signal,
            features=CandidateFeatures(),
            posterior=uniform_posterior,
            scores=CandidateScores(),
            recommended_pathway="tfop_ready",
            secondary_pathway="planet_hunters_discussion",
            explanation=CandidateExplanation(),
            metadata=minimal_metadata,
        )
        assert c.secondary_pathway == "planet_hunters_discussion"

    def test_rejects_invalid_pathway(
        self,
        valid_signal: CandidateSignal,
        uniform_posterior: HypothesisPosterior,
        minimal_metadata: ScoringMetadata,
    ) -> None:
        with pytest.raises(ValidationError):
            ScoredCandidate(
                signal=valid_signal,
                features=CandidateFeatures(),
                posterior=uniform_posterior,
                scores=CandidateScores(),
                recommended_pathway="not_a_real_pathway",  # type: ignore[arg-type]
                explanation=CandidateExplanation(),
                metadata=minimal_metadata,
            )

    def test_round_trips_to_dict(
        self, minimal_scored_candidate: ScoredCandidate
    ) -> None:
        d = minimal_scored_candidate.model_dump()
        assert d["signal"]["mission"] == "TESS"
        assert d["recommended_pathway"] == "github_only_reproducibility"
        assert d["secondary_pathway"] is None
