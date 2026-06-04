"""Tests for Skills/field_star_density_estimator.py."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from field_star_density_estimator import (
    estimate_field_star_density,
    format_field_density_result,
)


class TestEstimateFieldStarDensity:
    def test_galactic_plane_high_density(self) -> None:
        r = estimate_field_star_density(galactic_lat_deg=0.0)
        assert r.flag == "OK"
        assert r.stars_per_sqarcmin > 0

    def test_galactic_pole_low_density(self) -> None:
        r = estimate_field_star_density(galactic_lat_deg=90.0)
        assert r.flag == "OK"
        r_plane = estimate_field_star_density(galactic_lat_deg=0.0)
        assert r.stars_per_sqarcmin < r_plane.stars_per_sqarcmin

    def test_mid_latitude(self) -> None:
        r = estimate_field_star_density(galactic_lat_deg=45.0)
        assert r.flag == "OK"
        assert r.stars_per_sqarcmin > 0

    def test_crowding_risk_at_plane(self) -> None:
        r = estimate_field_star_density(galactic_lat_deg=2.0)
        assert r.crowding_risk in ("MODERATE", "HIGH")

    def test_crowding_risk_at_pole(self) -> None:
        r = estimate_field_star_density(galactic_lat_deg=80.0)
        assert r.crowding_risk == "LOW"

    def test_symmetric_latitude(self) -> None:
        r_pos = estimate_field_star_density(galactic_lat_deg=30.0)
        r_neg = estimate_field_star_density(galactic_lat_deg=-30.0)
        assert abs(r_pos.stars_per_sqarcmin - r_neg.stars_per_sqarcmin) < 1e-6

    def test_invalid_latitude_too_high(self) -> None:
        r = estimate_field_star_density(galactic_lat_deg=100.0)
        assert r.flag == "INVALID_GALACTIC_LAT"

    def test_tmag_limit_affects_density(self) -> None:
        r12 = estimate_field_star_density(galactic_lat_deg=30.0, limiting_tmag=12.0)
        r14 = estimate_field_star_density(galactic_lat_deg=30.0, limiting_tmag=14.0)
        assert r14.stars_per_sqarcmin >= r12.stars_per_sqarcmin

    def test_returns_dataclass(self) -> None:
        r = estimate_field_star_density(galactic_lat_deg=20.0)
        assert hasattr(r, "stars_per_sqarcmin")
        assert hasattr(r, "crowding_risk")

    def test_density_positive(self) -> None:
        r = estimate_field_star_density(galactic_lat_deg=60.0)
        assert r.stars_per_sqarcmin > 0

    def test_tess_pixel_density(self) -> None:
        r = estimate_field_star_density(galactic_lat_deg=10.0)
        assert r.stars_in_tess_pixel > 0

    def test_format_output(self) -> None:
        r = estimate_field_star_density(galactic_lat_deg=30.0)
        s = format_field_density_result(r)
        assert "|" in s
        assert "Crowding" in s or "crowding" in s or "density" in s.lower()
