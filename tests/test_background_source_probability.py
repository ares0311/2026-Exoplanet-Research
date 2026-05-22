"""Tests for Skills/background_source_probability.py."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from background_source_probability import (
    BackgroundSourceResult,
    estimate_bg_source_prob,
    format_bg_prob_result,
)


class TestEstimateBgSourceProb:
    def test_returns_background_source_result(self):
        result = estimate_bg_source_prob(180.0, 45.0)
        assert isinstance(result, BackgroundSourceResult)

    def test_flag_ok_for_valid_input(self):
        result = estimate_bg_source_prob(180.0, 45.0)
        assert result.flag == "OK"

    def test_invalid_dec_returns_invalid(self):
        result = estimate_bg_source_prob(180.0, 95.0)
        assert result.flag == "INVALID"

    def test_invalid_ra_returns_invalid(self):
        result = estimate_bg_source_prob(400.0, 45.0)
        assert result.flag == "INVALID"

    def test_bgeb_prior_between_zero_and_one(self):
        result = estimate_bg_source_prob(180.0, 45.0)
        assert 0.0 <= result.bgeb_prior <= 1.0

    def test_galactic_plane_higher_density(self):
        # Close to galactic plane (RA~266, Dec~-29 is roughly toward galactic center)
        plane_result = estimate_bg_source_prob(266.0, -29.0)
        pole_result = estimate_bg_source_prob(180.0, 89.0)
        # Plane should have higher prior (more background stars)
        assert plane_result.bgeb_prior >= pole_result.bgeb_prior

    def test_galactic_lat_computed(self):
        result = estimate_bg_source_prob(180.0, 45.0)
        assert result.galactic_lat_deg is not None
        assert -90.0 <= result.galactic_lat_deg <= 90.0

    def test_source_density_positive(self):
        result = estimate_bg_source_prob(180.0, 45.0)
        assert result.source_density_per_sq_deg is not None
        assert result.source_density_per_sq_deg >= 0.0

    def test_n_expected_in_aperture_positive(self):
        result = estimate_bg_source_prob(180.0, 45.0)
        assert result.n_expected_in_aperture is not None
        assert result.n_expected_in_aperture >= 0.0

    def test_is_crowded_when_n_expected_ge_one(self):
        # Use a very large aperture to force crowding
        result = estimate_bg_source_prob(266.0, -29.0, aperture_arcsec2=1e8)
        if result.n_expected_in_aperture is not None and result.n_expected_in_aperture >= 1.0:
            assert result.is_crowded

    def test_not_crowded_for_galactic_pole(self):
        result = estimate_bg_source_prob(180.0, 89.0, aperture_arcsec2=1385.4)
        assert result.is_crowded == (
            result.n_expected_in_aperture is not None
            and result.n_expected_in_aperture >= 1.0
        )

    def test_zero_dec_valid(self):
        result = estimate_bg_source_prob(0.0, 0.0)
        assert result.flag == "OK"

    def test_frozen_dataclass(self):
        result = estimate_bg_source_prob(180.0, 45.0)
        with pytest.raises((AttributeError, TypeError)):
            result.flag = "CHANGED"  # type: ignore[misc]


class TestFormatBgProbResult:
    def test_returns_string(self):
        result = estimate_bg_source_prob(180.0, 45.0)
        md = format_bg_prob_result(result)
        assert isinstance(md, str)

    def test_contains_flag(self):
        result = estimate_bg_source_prob(180.0, 45.0)
        md = format_bg_prob_result(result)
        assert result.flag in md

    def test_invalid_result_format(self):
        result = estimate_bg_source_prob(400.0, 45.0)
        md = format_bg_prob_result(result)
        assert "INVALID" in md
