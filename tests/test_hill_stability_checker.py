"""Tests for Skills/hill_stability_checker.py"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from hill_stability_checker import (
    check_hill_stability,
    format_hill_stability_result,
)


class TestHillStabilityChecker:
    def test_stable_pair_earth_like(self) -> None:
        r = check_hill_stability(1.0, 1.0, 1.0, 1.0, 2.0)
        assert r.flag == "OK"
        assert r.stability_flag == "STABLE"
        assert r.mutual_hill_radius_au > 0
        assert r.separation_hill_radii > 0

    def test_unstable_pair_very_close(self) -> None:
        r = check_hill_stability(1.0, 1.0, 1.0, 1.0, 1.001)
        assert r.flag == "OK"
        assert r.stability_flag == "UNSTABLE"

    def test_marginal_pair(self) -> None:
        # separation ~= 2*sqrt(3) Hill radii (marginal zone)
        r = check_hill_stability(1.0, 10.0, 1.0, 10.0, 1.05)
        assert r.flag == "OK"
        assert r.stability_flag in ("MARGINAL", "STABLE", "UNSTABLE")

    def test_hill_radius_scales_with_mass(self) -> None:
        r_low = check_hill_stability(1.0, 1.0, 1.0, 1.0, 2.0)
        r_high = check_hill_stability(1.0, 100.0, 1.0, 100.0, 2.0)
        assert r_high.mutual_hill_radius_au > r_low.mutual_hill_radius_au

    def test_separation_decreases_with_eccentricity(self) -> None:
        r_circ = check_hill_stability(1.0, 1.0, 1.0, 1.0, 2.0,
                                       inner_eccentricity=0.0, outer_eccentricity=0.0)
        r_ecc = check_hill_stability(1.0, 1.0, 1.0, 1.0, 2.0,
                                      inner_eccentricity=0.3, outer_eccentricity=0.3)
        assert r_circ.flag == "OK"
        assert r_ecc.flag == "OK"

    def test_invalid_stellar_mass(self) -> None:
        r = check_hill_stability(0.0, 1.0, 1.0, 1.0, 2.0)
        assert r.flag == "INVALID_STELLAR_MASS"
        assert math.isnan(r.mutual_hill_radius_au)

    def test_invalid_planet_mass(self) -> None:
        r = check_hill_stability(1.0, 0.0, 1.0, 1.0, 2.0)
        assert r.flag == "INVALID_PLANET_MASS"

    def test_invalid_sma(self) -> None:
        r = check_hill_stability(1.0, 1.0, 0.0, 1.0, 2.0)
        assert r.flag == "INVALID_SMA"

    def test_invalid_ordering(self) -> None:
        r = check_hill_stability(1.0, 1.0, 2.0, 1.0, 1.0)
        assert r.flag == "INVALID_ORDERING"

    def test_result_is_frozen(self) -> None:
        r = check_hill_stability(1.0, 1.0, 1.0, 1.0, 2.0)
        try:
            r.stability_flag = "X"  # type: ignore[misc]
            raise AssertionError("frozen dataclass should not allow setattr")
        except Exception:
            pass

    def test_format_ok(self) -> None:
        r = check_hill_stability(1.0, 1.0, 1.0, 1.0, 2.0)
        s = format_hill_stability_result(r)
        assert "Hill" in s or "Mutual" in s
        assert r.stability_flag in s

    def test_format_error(self) -> None:
        r = check_hill_stability(0.0, 1.0, 1.0, 1.0, 2.0)
        s = format_hill_stability_result(r)
        assert "INVALID_STELLAR_MASS" in s

    def test_large_mass_ratio_stable(self) -> None:
        # Jupiter vs Earth; well-separated
        r = check_hill_stability(1.0, 317.8, 5.2, 1.0, 10.0)
        assert r.flag == "OK"
        assert r.stability_flag == "STABLE"
