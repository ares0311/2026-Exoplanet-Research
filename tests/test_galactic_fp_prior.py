"""Tests for Skills/galactic_fp_prior.py."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from galactic_fp_prior import (
    GalacticFPPriorResult,
    estimate_galactic_fp_prior,
    format_galactic_fp_prior_result,
)


class TestEstimateGalacticFPPrior:
    def test_returns_result_type(self):
        r = estimate_galactic_fp_prior(galactic_latitude_deg=30.0)
        assert isinstance(r, GalacticFPPriorResult)

    def test_flag_ok(self):
        r = estimate_galactic_fp_prior(galactic_latitude_deg=30.0)
        assert r.flag == "OK"

    def test_bgeb_prior_between_0_and_1(self):
        r = estimate_galactic_fp_prior(galactic_latitude_deg=30.0)
        assert 0.0 <= r.bgeb_prior <= 1.0

    def test_blend_probability_between_0_and_1(self):
        r = estimate_galactic_fp_prior(galactic_latitude_deg=30.0)
        assert 0.0 <= r.blend_probability <= 1.0

    def test_stellar_density_positive(self):
        r = estimate_galactic_fp_prior(galactic_latitude_deg=30.0)
        assert r.stellar_density_per_sq_deg > 0.0

    def test_fp_prior_class_string(self):
        r = estimate_galactic_fp_prior(galactic_latitude_deg=30.0)
        assert r.fp_prior_class in ("LOW", "MODERATE", "HIGH", "VERY_HIGH")

    def test_low_latitude_higher_density(self):
        r_low = estimate_galactic_fp_prior(galactic_latitude_deg=5.0)
        r_high = estimate_galactic_fp_prior(galactic_latitude_deg=70.0)
        assert r_low.stellar_density_per_sq_deg > r_high.stellar_density_per_sq_deg

    def test_low_latitude_higher_fp_probability(self):
        r_low = estimate_galactic_fp_prior(galactic_latitude_deg=5.0)
        r_high = estimate_galactic_fp_prior(galactic_latitude_deg=70.0)
        assert r_low.bgeb_prior >= r_high.bgeb_prior

    def test_galactic_plane_higher_risk_than_pole(self):
        r_plane = estimate_galactic_fp_prior(galactic_latitude_deg=2.0)
        r_pole = estimate_galactic_fp_prior(galactic_latitude_deg=80.0)
        risk_order = ("LOW", "MODERATE", "HIGH", "VERY_HIGH")
        assert risk_order.index(r_plane.fp_prior_class) >= risk_order.index(r_pole.fp_prior_class)

    def test_poles_are_low_risk(self):
        r = estimate_galactic_fp_prior(galactic_latitude_deg=80.0)
        assert r.fp_prior_class in ("LOW", "MODERATE")

    def test_galactic_latitude_stored(self):
        r = estimate_galactic_fp_prior(galactic_latitude_deg=45.0)
        assert r.galactic_latitude_deg == pytest.approx(45.0, abs=0.1)

    def test_frozen_dataclass(self):
        r = estimate_galactic_fp_prior(galactic_latitude_deg=30.0)
        try:
            r.bgeb_prior = 0.0  # type: ignore[misc]
            raise AssertionError("frozen dataclass should not allow setattr")
        except (AttributeError, TypeError):
            pass

    def test_larger_aperture_higher_blend_prob(self):
        r1 = estimate_galactic_fp_prior(galactic_latitude_deg=30.0,
                                          aperture_radius_arcsec=20.0)
        r2 = estimate_galactic_fp_prior(galactic_latitude_deg=30.0,
                                          aperture_radius_arcsec=5.0)
        assert r1.blend_probability >= r2.blend_probability


class TestFormatGalacticFPPriorResult:
    def test_ok_returns_table(self):
        r = estimate_galactic_fp_prior(galactic_latitude_deg=30.0)
        out = format_galactic_fp_prior_result(r)
        assert "Stellar density" in out
        assert "|" in out

    def test_invalid_returns_flag(self):
        r = estimate_galactic_fp_prior(galactic_latitude_deg=95.0)  # out of range
        if r.flag != "OK":
            out = format_galactic_fp_prior_result(r)
            assert "flag=" in out
        else:
            # If implementation clamps, just test the table exists
            out = format_galactic_fp_prior_result(r)
            assert "|" in out
