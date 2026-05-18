"""Tests for Skills.limb_darkening_calculator."""
from __future__ import annotations

import pytest
from Skills.limb_darkening_calculator import (
    LimbDarkeningResult,
    compute_limb_darkening,
    format_ld_result,
)


class TestComputeLimbDarkening:
    def test_returns_result(self) -> None:
        r = compute_limb_darkening(5778.0, 4.44)
        assert isinstance(r, LimbDarkeningResult)

    def test_ok_flag_for_valid_grid_point(self) -> None:
        r = compute_limb_darkening(5000.0, 4.5)
        assert r.flag == "OK"

    def test_invalid_negative_teff(self) -> None:
        r = compute_limb_darkening(-100.0, 4.5)
        assert r.flag == "INVALID"

    def test_invalid_zero_teff(self) -> None:
        r = compute_limb_darkening(0.0, 4.5)
        assert r.flag == "INVALID"

    def test_invalid_negative_logg(self) -> None:
        r = compute_limb_darkening(5000.0, -1.0)
        assert r.flag == "INVALID"

    def test_extrapolated_high_teff(self) -> None:
        r = compute_limb_darkening(10000.0, 4.5)
        assert r.flag == "EXTRAPOLATED"

    def test_extrapolated_low_teff(self) -> None:
        r = compute_limb_darkening(2000.0, 4.5)
        assert r.flag == "EXTRAPOLATED"

    def test_extrapolated_high_logg(self) -> None:
        r = compute_limb_darkening(5000.0, 6.0)
        assert r.flag == "EXTRAPOLATED"

    def test_u1_in_range(self) -> None:
        r = compute_limb_darkening(5000.0, 4.5)
        assert 0.0 <= r.u1 <= 1.0

    def test_u2_in_range(self) -> None:
        r = compute_limb_darkening(5000.0, 4.5)
        assert 0.0 <= r.u2 <= 1.0

    def test_gamma_formula(self) -> None:
        r = compute_limb_darkening(5000.0, 4.5)
        expected = 1.0 - r.u1 / 3.0 - r.u2 / 6.0
        assert r.gamma == pytest.approx(expected, abs=1e-3)

    def test_passband_stored(self) -> None:
        r = compute_limb_darkening(5000.0, 4.5, passband="TESS")
        assert r.passband == "TESS"

    def test_grid_corner_value(self) -> None:
        # Grid corner at Teff=3500, logg=3.5 → u1=0.587, u2=0.162
        r = compute_limb_darkening(3500.0, 3.5)
        assert r.u1 == pytest.approx(0.587, abs=1e-3)
        assert r.u2 == pytest.approx(0.162, abs=1e-3)

    def test_interpolated_midpoint(self) -> None:
        # Midpoint should be between the two bracketing values
        r_lo = compute_limb_darkening(5000.0, 4.5)
        r_hi = compute_limb_darkening(6000.0, 4.5)
        r_mid = compute_limb_darkening(5500.0, 4.5)
        assert min(r_lo.u1, r_hi.u1) <= r_mid.u1 <= max(r_lo.u1, r_hi.u1)


class TestFormatLDResult:
    def test_returns_string(self) -> None:
        r = compute_limb_darkening(5000.0, 4.5)
        assert isinstance(format_ld_result(r), str)

    def test_contains_u1(self) -> None:
        r = compute_limb_darkening(5000.0, 4.5)
        assert "u1" in format_ld_result(r)

    def test_contains_flag(self) -> None:
        r = compute_limb_darkening(5000.0, 4.5)
        assert r.flag in format_ld_result(r)

    def test_invalid_result_formatted(self) -> None:
        r = compute_limb_darkening(-1.0, 4.5)
        out = format_ld_result(r)
        assert "INVALID" in out
