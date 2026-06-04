"""Tests for Skills/planet_formation_pathway.py."""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from planet_formation_pathway import assess_formation_pathway, format_formation_result


class TestAssessFormationPathway:
    def test_rocky_short_period_in_situ(self) -> None:
        r = assess_formation_pathway(period_days=5.0, bulk_density_gcc=5.5)
        assert r.flag == "OK"
        assert r.most_likely_pathway in ("IN_SITU", "CORE_ACCRETION")

    def test_gas_giant_hot_migration(self) -> None:
        r = assess_formation_pathway(period_days=5.0, bulk_density_gcc=0.8)
        assert r.flag == "OK"
        assert r.most_likely_pathway == "MIGRATION"

    def test_gas_giant_wide_disk_instability(self) -> None:
        r = assess_formation_pathway(period_days=200.0, bulk_density_gcc=0.5)
        assert r.flag == "OK"
        # Wide orbit low-density gas giant → disk instability or migration
        assert r.most_likely_pathway in ("DISK_INSTABILITY", "MIGRATION")

    def test_core_accretion_rocky(self) -> None:
        r = assess_formation_pathway(period_days=100.0, bulk_density_gcc=5.0)
        assert r.core_accretion_prob > 0

    def test_probs_sum_to_one(self) -> None:
        r = assess_formation_pathway(period_days=10.0, bulk_density_gcc=2.0)
        total = (r.core_accretion_prob + r.disk_instability_prob
                 + r.migration_prob + r.in_situ_prob)
        assert abs(total - 1.0) < 0.01

    def test_all_probs_positive(self) -> None:
        r = assess_formation_pathway(period_days=30.0, bulk_density_gcc=3.5)
        assert r.core_accretion_prob >= 0
        assert r.disk_instability_prob >= 0
        assert r.migration_prob >= 0
        assert r.in_situ_prob >= 0

    def test_invalid_period(self) -> None:
        r = assess_formation_pathway(period_days=0.0, bulk_density_gcc=3.0)
        assert r.flag == "INVALID_PERIOD"

    def test_invalid_density(self) -> None:
        r = assess_formation_pathway(period_days=10.0, bulk_density_gcc=-1.0)
        assert r.flag == "INVALID_DENSITY"

    def test_migration_hot_gas_giant(self) -> None:
        r = assess_formation_pathway(period_days=3.0, bulk_density_gcc=0.7)
        assert r.migration_prob > r.in_situ_prob

    def test_disk_instability_wide_gas(self) -> None:
        r = assess_formation_pathway(period_days=500.0, bulk_density_gcc=0.5)
        assert r.disk_instability_prob > 0

    def test_format_output(self) -> None:
        r = assess_formation_pathway(period_days=5.0, bulk_density_gcc=1.0)
        s = format_formation_result(r)
        assert "|" in s
        assert "pathway" in s.lower() or "accretion" in s.lower()

    def test_most_likely_is_valid(self) -> None:
        r = assess_formation_pathway(period_days=10.0, bulk_density_gcc=3.0)
        assert r.most_likely_pathway in (
            "CORE_ACCRETION", "DISK_INSTABILITY", "MIGRATION", "IN_SITU"
        )
        assert math.isfinite(r.core_accretion_prob)
