"""Tests for Skills/warm_jupiter_migration_classifier.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from warm_jupiter_migration_classifier import (
    classify_warm_jupiter,
    format_warm_jupiter_result,
)


class TestWarmJupiterMigrationClassifier:
    def test_basic_ok(self) -> None:
        r = classify_warm_jupiter(30.0)
        assert r.flag == "OK"
        assert r.migration_class in ("DISK_DRIVEN", "HIGH_E", "IN_SITU", "AMBIGUOUS")

    def test_circular_with_companion_disk_driven(self) -> None:
        r = classify_warm_jupiter(30.0, eccentricity=0.01, has_nearby_companion=True)
        assert r.flag == "OK"
        assert r.disk_score >= r.high_e_score

    def test_high_eccentricity_no_companion_high_e(self) -> None:
        r = classify_warm_jupiter(30.0, eccentricity=0.5, has_nearby_companion=False)
        assert r.flag == "OK"
        assert r.high_e_score >= r.disk_score

    def test_scores_sum_to_one(self) -> None:
        r = classify_warm_jupiter(30.0, eccentricity=0.1)
        total = r.disk_score + r.high_e_score + r.in_situ_score
        assert abs(total - 1.0) < 1e-9

    def test_very_short_period_circular_in_situ(self) -> None:
        r = classify_warm_jupiter(3.0, eccentricity=0.01)
        assert r.flag == "OK"
        assert r.in_situ_score > 0

    def test_evidence_tuple(self) -> None:
        r = classify_warm_jupiter(30.0, eccentricity=0.3)
        assert isinstance(r.evidence, tuple)
        assert len(r.evidence) > 0

    def test_outer_companion_adds_evidence(self) -> None:
        r_no = classify_warm_jupiter(30.0, eccentricity=0.3, has_nearby_companion=False)
        r_yes = classify_warm_jupiter(30.0, eccentricity=0.3, has_nearby_companion=False,
                                       companion_period_days=300.0, companion_mass_mjup=2.0)
        # Outer companion adds evidence lines; both still HIGH_E
        assert len(r_yes.evidence) > len(r_no.evidence)

    def test_invalid_period(self) -> None:
        r = classify_warm_jupiter(0.0)
        assert r.flag == "INVALID_PERIOD"

    def test_invalid_eccentricity_negative(self) -> None:
        r = classify_warm_jupiter(30.0, eccentricity=-0.1)
        assert r.flag == "INVALID_ECCENTRICITY"

    def test_invalid_eccentricity_ge_one(self) -> None:
        r = classify_warm_jupiter(30.0, eccentricity=1.0)
        assert r.flag == "INVALID_ECCENTRICITY"

    def test_result_is_frozen(self) -> None:
        r = classify_warm_jupiter(30.0)
        try:
            r.migration_class = "X"  # type: ignore[misc]
            raise AssertionError("frozen dataclass should not allow setattr")
        except Exception:
            pass

    def test_format_ok(self) -> None:
        r = classify_warm_jupiter(30.0)
        s = format_warm_jupiter_result(r)
        assert r.migration_class in s
        assert "score" in s.lower() or "Score" in s

    def test_format_error(self) -> None:
        r = classify_warm_jupiter(0.0)
        s = format_warm_jupiter_result(r)
        assert "INVALID_PERIOD" in s
