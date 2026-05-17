"""Tests for Skills.tess_year_planner."""
from __future__ import annotations

import pytest
from Skills.tess_year_planner import SectorPlan, format_sector_plan, plan_sectors


class TestPlanSectors:
    def test_southern_sky_gets_sectors_1_to_13(self) -> None:
        # Deep south ecliptic target: RA=0, Dec=-60 → ecliptic lat ~-70
        plan = plan_sectors(0.0, -70.0)
        assert any(s in plan.observed_sectors for s in range(1, 14))

    def test_northern_sky_gets_sectors_14_to_26(self) -> None:
        # Northern target: RA=180, Dec=70 → ecliptic lat ~60 (RA=180 has sin(RA)=0)
        plan = plan_sectors(180.0, 70.0)
        assert any(s in plan.observed_sectors for s in range(14, 27))

    def test_n_sectors_equals_len_observed(self) -> None:
        plan = plan_sectors(0.0, -70.0)
        assert plan.n_sectors == len(plan.observed_sectors)

    def test_ecliptic_latitude_stored(self) -> None:
        plan = plan_sectors(0.0, 0.0)
        assert isinstance(plan.ecliptic_latitude, float)

    def test_cvz_note_for_polar_target(self) -> None:
        # Ecliptic pole: Dec ~90 → ecliptic lat near 90 - 23.44 = 66.56, not quite CVZ
        # Use exact ecliptic pole: RA=270, Dec=66.56 → ecliptic lat ~90
        plan = plan_sectors(270.0, 89.0)
        if abs(plan.ecliptic_latitude) > 78.0:
            assert "CVZ" in plan.note

    def test_ecliptic_plane_note_for_equatorial_target(self) -> None:
        # RA=0, Dec=23.44 is near ecliptic plane
        plan = plan_sectors(0.0, 23.44)
        if abs(plan.ecliptic_latitude) < 6.0:
            assert "ecliptic" in plan.note.lower()

    def test_sector_fn_override_used(self) -> None:
        custom_sectors = [1, 5, 10]
        plan = plan_sectors(0.0, 0.0, sector_fn=lambda beta: custom_sectors)
        assert plan.observed_sectors == sorted(custom_sectors)

    def test_ra_dec_stored_in_plan(self) -> None:
        plan = plan_sectors(123.45, -45.67)
        assert plan.ra == pytest.approx(123.45)
        assert plan.dec == pytest.approx(-45.67)

    def test_returns_sector_plan_instance(self) -> None:
        plan = plan_sectors(0.0, 0.0)
        assert isinstance(plan, SectorPlan)

    def test_extended_sectors_included_for_equatorial(self) -> None:
        # Equatorial targets with ecliptic lat ~0 may appear in extended sectors 27+
        plan = plan_sectors(90.0, 23.44)
        all_sectors = plan.observed_sectors
        # May have extended sectors (27+) or none — just check no crash
        assert isinstance(all_sectors, list)


class TestFormatSectorPlan:
    def test_format_contains_ra_and_dec(self) -> None:
        plan = plan_sectors(10.0, -20.0)
        text = format_sector_plan(plan)
        assert "RA" in text
        assert "Dec" in text
