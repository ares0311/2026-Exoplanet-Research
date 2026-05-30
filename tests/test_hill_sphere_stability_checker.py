"""Tests for Skills/hill_sphere_stability_checker.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from hill_sphere_stability_checker import check_hill_stability, format_hill_stability


class TestCheckHillStability:
    def test_single_planet(self) -> None:
        r = check_hill_stability([5.0], 1.0, [1.0])
        assert r.flag == "SINGLE_PLANET"
        assert r.is_stable is True

    def test_well_separated_stable(self) -> None:
        r = check_hill_stability([5.0, 50.0], 1.0, [1.0, 1.0])
        assert r.is_stable is True
        assert r.flag == "STABLE"

    def test_tightly_packed_unstable(self) -> None:
        r = check_hill_stability([5.0, 5.01], 1.0, [10.0, 10.0])
        assert r.is_stable is False
        assert r.flag == "POTENTIALLY_UNSTABLE"

    def test_invalid_stellar_mass(self) -> None:
        r = check_hill_stability([5.0, 10.0], 0.0, [1.0, 1.0])
        assert r.flag == "INVALID_STELLAR_MASS"

    def test_mass_period_mismatch(self) -> None:
        r = check_hill_stability([5.0, 10.0], 1.0, [1.0])
        assert r.flag == "MASS_PERIOD_MISMATCH"

    def test_n_planets_stored(self) -> None:
        r = check_hill_stability([5.0, 10.0, 20.0], 1.0, [1.0, 1.0, 1.0])
        assert r.n_planets == 3

    def test_min_hill_separation_positive(self) -> None:
        r = check_hill_stability([5.0, 50.0], 1.0, [1.0, 1.0])
        assert r.min_hill_separation > 0

    def test_is_stable_bool(self) -> None:
        r = check_hill_stability([5.0, 50.0], 1.0, [1.0, 1.0])
        assert isinstance(r.is_stable, bool)

    def test_result_frozen(self) -> None:
        r = check_hill_stability([5.0, 50.0], 1.0, [1.0, 1.0])
        try:
            r.flag = "X"  # type: ignore[misc]
            raise AssertionError("should be frozen")
        except Exception:
            pass

    def test_format_returns_string(self) -> None:
        r = check_hill_stability([5.0, 50.0], 1.0, [1.0, 1.0])
        s = format_hill_stability(r)
        assert isinstance(s, str)
        assert "Hill" in s

    def test_three_planets_stable(self) -> None:
        r = check_hill_stability([3.0, 10.0, 30.0], 1.0, [0.1, 0.1, 0.1])
        assert r.n_planets == 3

    def test_flag_stable(self) -> None:
        r = check_hill_stability([5.0, 100.0], 1.0, [1.0, 1.0])
        assert r.flag == "STABLE"
