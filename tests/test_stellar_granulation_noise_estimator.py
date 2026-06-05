"""Tests for Skills/stellar_granulation_noise_estimator.py"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from stellar_granulation_noise_estimator import (
    estimate_granulation_noise,
    format_granulation_noise_result,
)


class TestStellarGranulationNoiseEstimator:
    def test_solar_case_ok(self) -> None:
        r = estimate_granulation_noise(5778.0, 4.438)
        assert r.flag == "OK"
        assert r.nu_max_uhz > 0
        assert r.sigma_total_ppm > 0

    def test_nu_max_near_solar(self) -> None:
        r = estimate_granulation_noise(5778.0, 4.438)
        assert 2000.0 < r.nu_max_uhz < 4000.0

    def test_giant_has_lower_nu_max(self) -> None:
        r_dwarf = estimate_granulation_noise(5778.0, 4.438)
        r_giant = estimate_granulation_noise(5000.0, 2.5)
        assert r_giant.nu_max_uhz < r_dwarf.nu_max_uhz

    def test_giant_has_larger_granulation_noise(self) -> None:
        r_dwarf = estimate_granulation_noise(5778.0, 4.438)
        r_giant = estimate_granulation_noise(5000.0, 2.5)
        assert r_giant.sigma_gran_ppm > r_dwarf.sigma_gran_ppm

    def test_total_ge_gran(self) -> None:
        r = estimate_granulation_noise(5778.0, 4.438)
        assert r.sigma_total_ppm >= r.sigma_gran_ppm

    def test_tau_gran_positive(self) -> None:
        r = estimate_granulation_noise(5778.0, 4.438)
        assert r.tau_gran_min > 0

    def test_invalid_teff(self) -> None:
        r = estimate_granulation_noise(0.0, 4.438)
        assert r.flag == "INVALID_TEFF"
        assert math.isnan(r.nu_max_uhz)

    def test_invalid_logg_high(self) -> None:
        r = estimate_granulation_noise(5778.0, 7.0)
        assert r.flag == "INVALID_LOGG"

    def test_invalid_logg_negative(self) -> None:
        r = estimate_granulation_noise(5778.0, -1.0)
        assert r.flag == "INVALID_LOGG"

    def test_invalid_cadence(self) -> None:
        r = estimate_granulation_noise(5778.0, 4.438, cadence_min=0.0)
        assert r.flag == "INVALID_CADENCE"

    def test_result_is_frozen(self) -> None:
        r = estimate_granulation_noise(5778.0, 4.438)
        try:
            r.nu_max_uhz = 0.0  # type: ignore[misc]
            raise AssertionError("frozen dataclass should not allow setattr")
        except Exception:
            pass

    def test_format_ok(self) -> None:
        r = estimate_granulation_noise(5778.0, 4.438)
        s = format_granulation_noise_result(r)
        assert "ppm" in s
        assert "gran" in s.lower() or "µHz" in s

    def test_format_error(self) -> None:
        r = estimate_granulation_noise(0.0, 4.438)
        s = format_granulation_noise_result(r)
        assert "INVALID_TEFF" in s
