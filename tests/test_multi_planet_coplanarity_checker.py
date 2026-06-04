"""Tests for Skills/multi_planet_coplanarity_checker.py."""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from multi_planet_coplanarity_checker import check_coplanarity, format_coplanarity_result


class TestCheckCoplanarity:
    def test_coplanar_system(self) -> None:
        # Two planets with duration ratio matching Kepler 3rd law exactly
        r = check_coplanarity(
            periods_days=[5.0, 10.0],
            durations_hours=[2.0, 2.0 * (10.0 / 5.0) ** (1 / 3)],
        )
        assert r.flag == "OK"
        assert r.coplanar

    def test_single_planet_flag(self) -> None:
        r = check_coplanarity(periods_days=[5.0], durations_hours=[2.0])
        assert r.flag == "SINGLE_PLANET"

    def test_length_mismatch(self) -> None:
        r = check_coplanarity(periods_days=[5.0, 10.0], durations_hours=[2.0])
        assert r.flag == "LENGTH_MISMATCH"

    def test_delta_i_nonneg(self) -> None:
        r = check_coplanarity(
            periods_days=[3.0, 10.0, 30.0],
            durations_hours=[2.0, 3.5, 4.0],
        )
        assert r.delta_i_max_deg >= 0

    def test_delta_i_max_gte_min(self) -> None:
        r = check_coplanarity(
            periods_days=[3.0, 10.0, 30.0],
            durations_hours=[2.0, 3.5, 4.0],
        )
        assert r.delta_i_max_deg >= r.delta_i_min_deg

    def test_n_planets_count(self) -> None:
        r = check_coplanarity(
            periods_days=[3.0, 10.0, 30.0],
            durations_hours=[2.0, 3.5, 4.5],
        )
        assert r.n_planets == 3

    def test_custom_threshold(self) -> None:
        r_loose = check_coplanarity(
            periods_days=[5.0, 20.0],
            durations_hours=[2.0, 5.0],
            coplanar_threshold_deg=90.0,
        )
        r_tight = check_coplanarity(
            periods_days=[5.0, 20.0],
            durations_hours=[2.0, 5.0],
            coplanar_threshold_deg=0.001,
        )
        assert r_loose.coplanar
        assert not r_tight.coplanar

    def test_flag_ok_for_valid(self) -> None:
        r = check_coplanarity(
            periods_days=[5.0, 15.0],
            durations_hours=[2.0, 3.0],
        )
        assert r.flag == "OK"

    def test_finite_delta_i(self) -> None:
        r = check_coplanarity(
            periods_days=[5.0, 15.0],
            durations_hours=[2.0, 3.0],
        )
        assert math.isfinite(r.delta_i_max_deg)
        assert math.isfinite(r.delta_i_min_deg)

    def test_stellar_radius_scales_result(self) -> None:
        r1 = check_coplanarity(
            periods_days=[5.0, 15.0],
            durations_hours=[2.0, 4.0],
            stellar_radius_rsun=0.5,
        )
        r2 = check_coplanarity(
            periods_days=[5.0, 15.0],
            durations_hours=[2.0, 4.0],
            stellar_radius_rsun=2.0,
        )
        # Larger star → smaller a/R* → larger inclination bound
        assert r1.delta_i_max_deg != r2.delta_i_max_deg

    def test_format_output(self) -> None:
        r = check_coplanarity(
            periods_days=[5.0, 15.0],
            durations_hours=[2.0, 3.5],
        )
        s = format_coplanarity_result(r)
        assert "|" in s
        assert "coplanar" in s.lower() or "inclination" in s.lower()

    def test_empty_lists(self) -> None:
        r = check_coplanarity(periods_days=[], durations_hours=[])
        assert r.flag == "SINGLE_PLANET"
