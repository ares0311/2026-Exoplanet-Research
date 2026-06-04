"""Tests for Skills/stellar_oscillation_noise.py."""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from stellar_oscillation_noise import (
    estimate_oscillation_noise,
    format_oscillation_noise_result,
)


class TestEstimateOscillationNoise:
    def test_solar_values(self) -> None:
        r = estimate_oscillation_noise(teff_k=5778.0, luminosity_lsun=1.0, mass_msun=1.0)
        assert r.flag == "OK"
        assert math.isfinite(r.nu_max_uhz)
        assert r.nu_max_uhz > 0

    def test_nu_max_near_solar(self) -> None:
        r = estimate_oscillation_noise(teff_k=5778.0, luminosity_lsun=1.0, mass_msun=1.0)
        assert 2000 < r.nu_max_uhz < 4500

    def test_giant_lower_nu_max(self) -> None:
        r_ms = estimate_oscillation_noise(teff_k=5000.0, luminosity_lsun=1.0)
        r_giant = estimate_oscillation_noise(teff_k=4800.0, luminosity_lsun=50.0)
        assert r_giant.nu_max_uhz < r_ms.nu_max_uhz

    def test_oscillation_amplitude_positive(self) -> None:
        r = estimate_oscillation_noise(teff_k=5778.0, luminosity_lsun=1.0)
        assert r.oscillation_amplitude_ppm > 0

    def test_granulation_amplitude_positive(self) -> None:
        r = estimate_oscillation_noise(teff_k=5778.0, luminosity_lsun=1.0)
        assert r.granulation_amplitude_ppm > 0

    def test_total_noise_gte_components(self) -> None:
        r = estimate_oscillation_noise(teff_k=5778.0, luminosity_lsun=1.0)
        assert r.total_noise_ppm >= r.oscillation_amplitude_ppm

    def test_invalid_teff(self) -> None:
        r = estimate_oscillation_noise(teff_k=-100.0)
        assert r.flag == "INVALID_TEFF"

    def test_invalid_mass(self) -> None:
        r = estimate_oscillation_noise(teff_k=5778.0, mass_msun=0.0)
        assert r.flag == "INVALID_MASS"

    def test_logg_fallback(self) -> None:
        r = estimate_oscillation_noise(teff_k=5778.0, logg=4.44)
        assert r.flag == "OK"
        assert r.nu_max_uhz > 0

    def test_more_luminous_higher_amplitude(self) -> None:
        r1 = estimate_oscillation_noise(teff_k=5778.0, luminosity_lsun=1.0)
        r2 = estimate_oscillation_noise(teff_k=5778.0, luminosity_lsun=10.0)
        assert r2.oscillation_amplitude_ppm > r1.oscillation_amplitude_ppm

    def test_format_output(self) -> None:
        r = estimate_oscillation_noise(teff_k=5778.0, luminosity_lsun=1.0)
        s = format_oscillation_noise_result(r)
        assert "|" in s
        assert "max" in s.lower() or "noise" in s.lower()

    def test_delta_nu_positive(self) -> None:
        r = estimate_oscillation_noise(teff_k=5778.0, luminosity_lsun=1.0)
        assert r.delta_nu_uhz > 0
