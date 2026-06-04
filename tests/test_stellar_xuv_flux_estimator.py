"""Tests for Skills/stellar_xuv_flux_estimator.py"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from stellar_xuv_flux_estimator import compute_stellar_xuv_flux, format_stellar_xuv_flux_result


class TestComputeStellarXuvFlux:
    def test_ok_flag(self) -> None:
        r = compute_stellar_xuv_flux(1.0, 1.0)
        assert r.flag == "OK"

    def test_lx_positive(self) -> None:
        r = compute_stellar_xuv_flux(1.0, 1.0)
        assert r.lx_erg_s > 0

    def test_leuv_positive(self) -> None:
        r = compute_stellar_xuv_flux(1.0, 1.0)
        assert r.leuv_erg_s > 0

    def test_lxuv_greater_than_lx(self) -> None:
        r = compute_stellar_xuv_flux(1.0, 1.0)
        assert r.lxuv_erg_s > r.lx_erg_s

    def test_flux_at_planet_positive(self) -> None:
        r = compute_stellar_xuv_flux(1.0, 1.0)
        assert r.xuv_flux_at_planet_wm2 > 0

    def test_closer_planet_higher_flux(self) -> None:
        r_close = compute_stellar_xuv_flux(1.0, 0.1)
        r_far = compute_stellar_xuv_flux(1.0, 1.0)
        assert r_close.xuv_flux_at_planet_wm2 > r_far.xuv_flux_at_planet_wm2

    def test_saturation_regime_fast_rotator(self) -> None:
        r = compute_stellar_xuv_flux(1.0, 1.0, rotation_period_days=0.5)
        assert r.saturation_regime

    def test_unsaturated_slow_rotator(self) -> None:
        r = compute_stellar_xuv_flux(1.0, 1.0, rotation_period_days=30.0)
        assert not r.saturation_regime

    def test_age_based_rotation_period(self) -> None:
        r = compute_stellar_xuv_flux(1.0, 1.0, stellar_age_gyr=4.6)
        assert r.rotation_period_days is not None
        assert r.rotation_period_days > 0

    def test_invalid_mass(self) -> None:
        r = compute_stellar_xuv_flux(0.0, 1.0)
        assert r.flag == "INVALID_MASS"

    def test_invalid_distance(self) -> None:
        r = compute_stellar_xuv_flux(1.0, 0.0)
        assert r.flag == "INVALID_DISTANCE"

    def test_result_frozen(self) -> None:
        r = compute_stellar_xuv_flux(1.0, 1.0)
        try:
            r.lx_erg_s = 0.0  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except (AttributeError, TypeError):
            pass

    def test_format_returns_string(self) -> None:
        r = compute_stellar_xuv_flux(1.0, 1.0)
        s = format_stellar_xuv_flux_result(r)
        assert isinstance(s, str)
        assert r.flag in s
