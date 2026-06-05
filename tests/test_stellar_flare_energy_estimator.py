"""Tests for Skills/stellar_flare_energy_estimator.py"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from stellar_flare_energy_estimator import (
    estimate_flare_energy,
    format_flare_energy_result,
)


class TestStellarFlareEnergyEstimator:
    def test_basic_ok(self) -> None:
        r = estimate_flare_energy(10000.0, 1.0)
        assert r.flag == "OK"
        assert r.flare_energy_j > 0

    def test_energy_scales_with_amplitude(self) -> None:
        r_low = estimate_flare_energy(1000.0, 1.0)
        r_high = estimate_flare_energy(10000.0, 1.0)
        assert r_high.flare_energy_j > r_low.flare_energy_j

    def test_energy_scales_with_duration(self) -> None:
        r_short = estimate_flare_energy(10000.0, 0.5)
        r_long = estimate_flare_energy(10000.0, 2.0)
        assert r_long.flare_energy_j > r_short.flare_energy_j

    def test_energy_scales_with_luminosity(self) -> None:
        r_faint = estimate_flare_energy(10000.0, 1.0, stellar_luminosity_lsun=0.1)
        r_bright = estimate_flare_energy(10000.0, 1.0, stellar_luminosity_lsun=10.0)
        assert r_bright.flare_energy_j > r_faint.flare_energy_j

    def test_log_energy_consistent(self) -> None:
        r = estimate_flare_energy(10000.0, 1.0)
        assert abs(r.flare_energy_log10 - math.log10(r.flare_energy_j)) < 0.01

    def test_flare_class_x_large_flare(self) -> None:
        # Superflare: very large amplitude on bright star
        r = estimate_flare_energy(1e6, 10.0, stellar_luminosity_lsun=1.0, shape="box")
        assert r.flare_class in ("M", "X")

    def test_flare_class_a_small_flare(self) -> None:
        r = estimate_flare_energy(10.0, 0.1, stellar_luminosity_lsun=0.01)
        assert r.flare_class == "A"

    def test_box_shape_double_exponential(self) -> None:
        r_exp = estimate_flare_energy(10000.0, 1.0, shape="exponential")
        r_box = estimate_flare_energy(10000.0, 1.0, shape="box")
        assert abs(r_box.flare_energy_j / r_exp.flare_energy_j - 2.0) < 0.01

    def test_invalid_amplitude(self) -> None:
        r = estimate_flare_energy(0.0, 1.0)
        assert r.flag == "INVALID_AMPLITUDE"
        assert math.isnan(r.flare_energy_j)

    def test_invalid_duration(self) -> None:
        r = estimate_flare_energy(10000.0, 0.0)
        assert r.flag == "INVALID_DURATION"

    def test_invalid_luminosity(self) -> None:
        r = estimate_flare_energy(10000.0, 1.0, stellar_luminosity_lsun=0.0)
        assert r.flag == "INVALID_LUMINOSITY"

    def test_result_is_frozen(self) -> None:
        r = estimate_flare_energy(10000.0, 1.0)
        try:
            r.flare_class = "Z"  # type: ignore[misc]
            raise AssertionError("frozen dataclass should not allow setattr")
        except Exception:
            pass

    def test_format_ok(self) -> None:
        r = estimate_flare_energy(10000.0, 1.0)
        s = format_flare_energy_result(r)
        assert r.flare_class in s
        assert "J" in s or "energy" in s.lower()
