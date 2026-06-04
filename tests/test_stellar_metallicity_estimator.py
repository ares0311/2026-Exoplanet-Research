"""Tests for Skills/stellar_metallicity_estimator.py."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from stellar_metallicity_estimator import (
    estimate_metallicity,
    format_metallicity_result,
)


class TestEstimateMetallicity:
    def test_solar_bv(self) -> None:
        r = estimate_metallicity("B-V", 0.65)
        assert r.flag == "OK"
        assert abs(r.feh) < 0.5

    def test_solar_bprp(self) -> None:
        r = estimate_metallicity("Bp-Rp", 0.82)
        assert r.flag == "OK"

    def test_blue_star_metal_poor(self) -> None:
        r = estimate_metallicity("B-V", 0.30)
        assert r.flag == "OK"

    def test_red_star(self) -> None:
        r = estimate_metallicity("B-V", 1.2)
        assert r.flag == "OK"

    def test_unknown_index(self) -> None:
        r = estimate_metallicity("V-I", 0.8)
        assert r.flag == "UNKNOWN_COLOR_INDEX"

    def test_giant_out_of_calibration(self) -> None:
        r = estimate_metallicity("B-V", 0.65, luminosity_class="III")
        assert r.flag == "OUT_OF_CALIBRATION"

    def test_out_of_range_bv(self) -> None:
        r = estimate_metallicity("B-V", 2.0)
        assert r.flag == "OUT_OF_RANGE"

    def test_out_of_range_bprp(self) -> None:
        r = estimate_metallicity("Bp-Rp", 4.0)
        assert r.flag == "OUT_OF_RANGE"

    def test_feh_reasonable_range(self) -> None:
        r = estimate_metallicity("B-V", 0.65)
        assert -3.0 <= r.feh <= 1.0

    def test_feh_error_positive(self) -> None:
        r = estimate_metallicity("B-V", 0.65)
        assert r.feh_err > 0

    def test_returns_dataclass_fields(self) -> None:
        r = estimate_metallicity("Bp-Rp", 1.0)
        assert hasattr(r, "feh")
        assert hasattr(r, "feh_err")
        assert hasattr(r, "color_index")

    def test_format_output(self) -> None:
        r = estimate_metallicity("B-V", 0.65)
        s = format_metallicity_result(r)
        assert "|" in s
        assert "Fe" in s or "feh" in s.lower() or "metallicity" in s.lower()
