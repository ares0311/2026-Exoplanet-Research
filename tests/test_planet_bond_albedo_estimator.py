"""Tests for Skills/planet_bond_albedo_estimator.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from planet_bond_albedo_estimator import compute_bond_albedo, format_bond_albedo_result


class TestComputeBondAlbedo:
    def test_ok_flag(self) -> None:
        r = compute_bond_albedo(255.0, 1.0)
        assert r.flag == "OK"

    def test_albedo_between_zero_and_one(self) -> None:
        r = compute_bond_albedo(255.0, 1.0)
        assert 0.0 <= r.bond_albedo <= 1.0

    def test_geometric_albedo_non_negative(self) -> None:
        r = compute_bond_albedo(255.0, 1.0)
        assert r.geometric_albedo_estimate >= 0.0

    def test_stellar_flux_positive(self) -> None:
        r = compute_bond_albedo(255.0, 1.0)
        assert r.stellar_flux_wm2 > 0.0

    def test_hotter_planet_lower_albedo(self) -> None:
        r_hot = compute_bond_albedo(2000.0, 0.05)
        r_cool = compute_bond_albedo(500.0, 1.0)
        assert r_hot.bond_albedo <= r_cool.bond_albedo

    def test_close_orbit_ultra_hot_class(self) -> None:
        r = compute_bond_albedo(2500.0, 0.03, stellar_teff_k=6000.0)
        assert r.irradiation_class in ("ULTRA_HOT", "HOT")

    def test_earth_like_warm_class(self) -> None:
        r = compute_bond_albedo(255.0, 1.0)
        assert r.irradiation_class == "WARM"

    def test_invalid_teq(self) -> None:
        r = compute_bond_albedo(0.0, 1.0)
        assert r.flag == "INVALID_TEQ"

    def test_invalid_distance(self) -> None:
        r = compute_bond_albedo(255.0, 0.0)
        assert r.flag == "INVALID_DISTANCE"

    def test_invalid_stellar_params(self) -> None:
        r = compute_bond_albedo(255.0, 1.0, stellar_teff_k=0.0)
        assert r.flag == "INVALID_STELLAR_PARAMS"

    def test_result_frozen(self) -> None:
        r = compute_bond_albedo(255.0, 1.0)
        try:
            r.bond_albedo = 0.5  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except (AttributeError, TypeError):
            pass

    def test_format_returns_string(self) -> None:
        r = compute_bond_albedo(255.0, 1.0)
        s = format_bond_albedo_result(r)
        assert isinstance(s, str)
        assert r.flag in s
