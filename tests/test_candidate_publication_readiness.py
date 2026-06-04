"""Tests for Skills/candidate_publication_readiness.py."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from candidate_publication_readiness import (
    assess_publication_readiness,
    format_readiness_result,
)

_GOOD_CANDIDATE = {
    "false_positive_probability": 0.05,
    "detection_confidence": 0.92,
    "n_transits": 5,
    "snr": 12.0,
    "depth_consistent": True,
    "secondary_snr": 1.0,
    "odd_even_sigma": 0.5,
    "centroid_motion_arcsec": 0.2,
    "stellar_teff_k": 5778.0,
    "stellar_logg": 4.44,
    "stellar_radius_rsun": 1.0,
}


class TestAssessPublicationReadiness:
    def test_good_candidate_ready(self) -> None:
        r = assess_publication_readiness(_GOOD_CANDIDATE)
        assert r.flag == "OK"
        assert r.recommendation == "READY"

    def test_n_gates_is_ten(self) -> None:
        r = assess_publication_readiness(_GOOD_CANDIDATE)
        assert r.n_gates == 10

    def test_empty_candidate_not_ready(self) -> None:
        r = assess_publication_readiness({})
        assert r.recommendation == "NOT_READY"

    def test_score_range(self) -> None:
        r = assess_publication_readiness(_GOOD_CANDIDATE)
        assert 0.0 <= r.readiness_score <= 1.0

    def test_passed_gates_subset(self) -> None:
        r = assess_publication_readiness(_GOOD_CANDIDATE)
        assert len(r.passed_gates) + len(r.failed_gates) == r.n_gates

    def test_high_fpp_fails_gate(self) -> None:
        cand = dict(_GOOD_CANDIDATE)
        cand["false_positive_probability"] = 0.50
        r = assess_publication_readiness(cand)
        low_fpp_label = "FPP < 0.10"
        assert low_fpp_label in r.failed_gates

    def test_needs_follow_up_threshold(self) -> None:
        cand = dict(_GOOD_CANDIDATE)
        cand["false_positive_probability"] = 0.50
        cand["detection_confidence"] = 0.50
        r = assess_publication_readiness(cand)
        assert r.recommendation in ("NEEDS_FOLLOW_UP", "NOT_READY")

    def test_secondary_eclipse_fails(self) -> None:
        cand = dict(_GOOD_CANDIDATE)
        cand["secondary_snr"] = 5.0
        r = assess_publication_readiness(cand)
        assert "No significant secondary eclipse" in r.failed_gates

    def test_scores_nested_dict(self) -> None:
        cand = {
            "scores": {
                "false_positive_probability": 0.05,
                "detection_confidence": 0.92,
            },
            "n_transits": 5,
            "snr": 10.0,
            "depth_consistent": True,
            "stellar_teff_k": 5500.0,
            "stellar_logg": 4.5,
            "stellar_radius_rsun": 0.9,
        }
        r = assess_publication_readiness(cand)
        assert r.flag == "OK"

    def test_centroid_fails_gate(self) -> None:
        cand = dict(_GOOD_CANDIDATE)
        cand["centroid_motion_arcsec"] = 2.0
        r = assess_publication_readiness(cand)
        assert "Centroid shift < 1 arcsec" in r.failed_gates

    def test_format_output(self) -> None:
        r = assess_publication_readiness(_GOOD_CANDIDATE)
        s = format_readiness_result(r)
        assert "READY" in s or "NEEDS_FOLLOW_UP" in s

    def test_spectral_type_giant_fails(self) -> None:
        cand = dict(_GOOD_CANDIDATE)
        cand["stellar_logg"] = 2.5
        r = assess_publication_readiness(cand)
        assert any("FGK dwarf" in g for g in r.failed_gates)
