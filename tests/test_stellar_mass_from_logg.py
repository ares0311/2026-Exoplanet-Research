"""Tests for Skills/stellar_mass_from_logg.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from stellar_mass_from_logg import compute_stellar_mass_from_logg, format_stellar_mass_result


class TestStellarMassFromLogg:
    def test_solar_logg(self) -> None:
        # Sun: log g = 4.44, R = 1.0 R_sun => M ~1 M_sun
        r = compute_stellar_mass_from_logg(4.44, 1.0)
        assert r.flag == "OK"
        assert 0.8 < r.mass_msun < 1.2

    def test_invalid_logg_negative(self) -> None:
        r = compute_stellar_mass_from_logg(-1.0, 1.0)
        assert r.flag == "INVALID_LOGG"

    def test_invalid_logg_too_high(self) -> None:
        r = compute_stellar_mass_from_logg(7.0, 1.0)
        assert r.flag == "INVALID_LOGG"

    def test_invalid_radius_zero(self) -> None:
        r = compute_stellar_mass_from_logg(4.44, 0.0)
        assert r.flag == "INVALID_RADIUS"

    def test_invalid_radius_negative(self) -> None:
        r = compute_stellar_mass_from_logg(4.44, -1.0)
        assert r.flag == "INVALID_RADIUS"

    def test_mass_scales_with_radius_squared(self) -> None:
        r1 = compute_stellar_mass_from_logg(4.44, 1.0)
        r2 = compute_stellar_mass_from_logg(4.44, 2.0)
        assert abs(r2.mass_msun / r1.mass_msun - 4.0) < 0.01

    def test_mass_scales_with_gravity(self) -> None:
        r1 = compute_stellar_mass_from_logg(4.0, 1.0)
        r2 = compute_stellar_mass_from_logg(5.0, 1.0)
        assert r2.mass_msun > r1.mass_msun

    def test_error_propagation(self) -> None:
        r = compute_stellar_mass_from_logg(4.44, 1.0, logg_err=0.1, radius_err_rsun=0.05)
        assert r.mass_msun_err is not None
        assert r.mass_msun_err > 0.0

    def test_no_error_without_inputs(self) -> None:
        r = compute_stellar_mass_from_logg(4.44, 1.0)
        assert r.mass_msun_err is None

    def test_log_mass_consistent(self) -> None:
        r = compute_stellar_mass_from_logg(4.44, 1.0)
        import math
        assert abs(r.log_mass - math.log10(r.mass_msun)) < 0.001

    def test_off_main_sequence_white_dwarf(self) -> None:
        # Very high logg, tiny radius => very low mass or off-MS
        r = compute_stellar_mass_from_logg(5.5, 0.01)
        assert r.flag == "OFF_MAIN_SEQUENCE"

    def test_format_returns_string(self) -> None:
        r = compute_stellar_mass_from_logg(4.44, 1.0)
        s = format_stellar_mass_result(r)
        assert isinstance(s, str)
        assert "Mass" in s
