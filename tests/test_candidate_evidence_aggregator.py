"""Tests for Skills.candidate_evidence_aggregator."""
from __future__ import annotations

from Skills.candidate_evidence_aggregator import (
    EvidenceAggregateResult,
    EvidenceItem,
    aggregate_evidence,
    format_evidence_aggregate,
)

_CLEAN_DIAGNOSTICS = {
    "snr": ("PASS", 15.0),
    "odd_even": ("PASS", 0.5),
    "secondary_eclipse": ("PASS", None),
    "centroid": ("PASS", 0.1),
}

_FP_DIAGNOSTICS = {
    "snr": ("FAIL", 2.0),
    "odd_even": ("FAIL", 5.0),
    "secondary_eclipse": ("FAIL", 2000.0),
    "centroid": ("FAIL", 50.0),
}


class TestAggregateEvidence:
    def test_returns_result(self) -> None:
        r = aggregate_evidence(12345, _CLEAN_DIAGNOSTICS)
        assert isinstance(r, EvidenceAggregateResult)

    def test_empty_diagnostics_insufficient(self) -> None:
        r = aggregate_evidence(12345, {})
        assert r.flag == "INSUFFICIENT"

    def test_ok_flag_with_data(self) -> None:
        r = aggregate_evidence(12345, _CLEAN_DIAGNOSTICS)
        assert r.flag == "OK"

    def test_planet_candidate_clean(self) -> None:
        r = aggregate_evidence(12345, _CLEAN_DIAGNOSTICS, planet_threshold=0.0)
        assert r.classification in {"planet_candidate", "ambiguous"}

    def test_n_pass_correct(self) -> None:
        r = aggregate_evidence(12345, _CLEAN_DIAGNOSTICS)
        assert r.n_pass == 4

    def test_n_fail_correct(self) -> None:
        r = aggregate_evidence(12345, _FP_DIAGNOSTICS)
        assert r.n_fail == 4

    def test_n_unknown_for_warn(self) -> None:
        diag = {"snr": ("WARN", 5.0)}
        r = aggregate_evidence(12345, diag)
        assert r.n_unknown == 1

    def test_net_score_in_range(self) -> None:
        r = aggregate_evidence(12345, _CLEAN_DIAGNOSTICS)
        assert -1.0 <= r.net_score <= 1.0

    def test_items_populated(self) -> None:
        r = aggregate_evidence(12345, _CLEAN_DIAGNOSTICS)
        assert len(r.items) == 4

    def test_items_are_evidence_items(self) -> None:
        r = aggregate_evidence(12345, _CLEAN_DIAGNOSTICS)
        for item in r.items:
            assert isinstance(item, EvidenceItem)

    def test_tic_id_stored(self) -> None:
        r = aggregate_evidence(150428135, _CLEAN_DIAGNOSTICS)
        assert r.tic_id == 150428135

    def test_classification_values_valid(self) -> None:
        r = aggregate_evidence(12345, _CLEAN_DIAGNOSTICS)
        assert r.classification in {"planet_candidate", "likely_fp", "ambiguous"}

    def test_fp_diagnostics_low_score(self) -> None:
        r = aggregate_evidence(12345, _FP_DIAGNOSTICS)
        assert r.net_score <= 0.0


class TestFormatEvidenceAggregate:
    def test_returns_string(self) -> None:
        r = aggregate_evidence(12345, _CLEAN_DIAGNOSTICS)
        assert isinstance(format_evidence_aggregate(r), str)

    def test_contains_tic_id(self) -> None:
        r = aggregate_evidence(12345, _CLEAN_DIAGNOSTICS)
        assert "12345" in format_evidence_aggregate(r)

    def test_contains_classification(self) -> None:
        r = aggregate_evidence(12345, _CLEAN_DIAGNOSTICS)
        assert r.classification in format_evidence_aggregate(r)

    def test_insufficient_handled(self) -> None:
        r = aggregate_evidence(12345, {})
        out = format_evidence_aggregate(r)
        assert "INSUFFICIENT" in out
