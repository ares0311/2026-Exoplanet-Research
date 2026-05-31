"""Tests for Skills/multiplanet_hill_stability.py"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from multiplanet_hill_stability import check_hill_stability, format_stability_result


class TestMultiplanetHillStability:
    def test_well_separated_stable(self) -> None:
        # Earth at 1 AU, Jupiter-analog at 5 AU → stable
        r = check_hill_stability(365.25, 4332.59, 1.0, 318.0)
        assert r.flag == "OK"
        assert r.stable is True

    def test_close_packed_unstable(self) -> None:
        # Very close periods with large masses → unstable
        r = check_hill_stability(10.0, 11.0, 300.0, 300.0, stellar_mass_msun=1.0)
        assert r.stable is False
        assert r.flag == "HILL_UNSTABLE"

    def test_delta_hill_positive(self) -> None:
        r = check_hill_stability(10.0, 20.0, 1.0, 1.0)
        assert r.delta_hill > 0.0

    def test_delta_crit_is_2sqrt3(self) -> None:
        r = check_hill_stability(10.0, 20.0, 1.0, 1.0)
        assert abs(r.delta_crit - 2.0 * math.sqrt(3.0)) < 1e-3

    def test_invalid_inner_period(self) -> None:
        r = check_hill_stability(0.0, 10.0, 1.0, 1.0)
        assert "INVALID" in r.flag

    def test_invalid_outer_period(self) -> None:
        r = check_hill_stability(10.0, 0.0, 1.0, 1.0)
        assert "INVALID" in r.flag

    def test_invalid_inner_mass(self) -> None:
        r = check_hill_stability(10.0, 20.0, 0.0, 1.0)
        assert "INVALID" in r.flag

    def test_invalid_stellar_mass(self) -> None:
        r = check_hill_stability(10.0, 20.0, 1.0, 1.0, stellar_mass_msun=0.0)
        assert "INVALID" in r.flag

    def test_period_order_error(self) -> None:
        r = check_hill_stability(20.0, 10.0, 1.0, 1.0)
        assert r.flag == "PERIOD_ORDER_ERROR"

    def test_heavier_masses_reduce_stability(self) -> None:
        r_light = check_hill_stability(10.0, 15.0, 1.0, 1.0)
        r_heavy = check_hill_stability(10.0, 15.0, 100.0, 100.0)
        assert r_heavy.delta_hill < r_light.delta_hill

    def test_wider_separation_more_stable(self) -> None:
        r_close = check_hill_stability(10.0, 12.0, 10.0, 10.0)
        r_wide = check_hill_stability(10.0, 30.0, 10.0, 10.0)
        assert r_wide.delta_hill > r_close.delta_hill

    def test_format_returns_string(self) -> None:
        r = check_hill_stability(10.0, 20.0, 1.0, 1.0)
        s = format_stability_result(r)
        assert isinstance(s, str)
        assert "Hill" in s
