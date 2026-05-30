"""Tests for Skills/stellar_noise_budget.py"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from stellar_noise_budget import compute_noise_budget, format_noise_budget


class TestComputeNoiseBudget:
    def test_basic_photon_noise(self) -> None:
        r = compute_noise_budget(10000.0, 0.0, 1, 0.0)
        assert r.flag == "OK"
        assert abs(r.photon_noise_ppm - 10000.0) < 1.0

    def test_photon_noise_formula(self) -> None:
        r = compute_noise_budget(1000000.0, 0.0, 1, 0.0)
        assert abs(r.photon_noise_ppm - 1000.0) < 0.1

    def test_invalid_target_counts(self) -> None:
        r = compute_noise_budget(0.0, 0.0, 1, 0.0)
        assert r.flag == "INVALID_TARGET_COUNTS"

    def test_invalid_n_pixels(self) -> None:
        r = compute_noise_budget(10000.0, 0.0, 0, 0.0)
        assert r.flag == "INVALID_N_PIXELS"

    def test_invalid_n_cadences(self) -> None:
        r = compute_noise_budget(10000.0, 0.0, 1, 0.0, n_cadences=0)
        assert r.flag == "INVALID_N_CADENCES"

    def test_total_noise_larger_than_photon(self) -> None:
        r = compute_noise_budget(10000.0, 10.0, 9, 5.0)
        assert r.total_noise_ppm >= r.photon_noise_ppm

    def test_systematic_added_in_quadrature(self) -> None:
        r0 = compute_noise_budget(10000.0, 0.0, 1, 0.0, systematic_ppm=0.0)
        r1 = compute_noise_budget(10000.0, 0.0, 1, 0.0, systematic_ppm=100.0)
        expected = math.sqrt(r0.photon_noise_ppm ** 2 + 100.0 ** 2)
        assert abs(r1.total_noise_ppm - expected) < 0.1

    def test_more_cadences_lowers_noise(self) -> None:
        r1 = compute_noise_budget(10000.0, 0.0, 1, 0.0, n_cadences=1)
        r4 = compute_noise_budget(10000.0, 0.0, 1, 0.0, n_cadences=4)
        assert r4.photon_noise_ppm < r1.photon_noise_ppm

    def test_noise_components_non_negative(self) -> None:
        r = compute_noise_budget(10000.0, 5.0, 9, 3.0)
        assert r.photon_noise_ppm >= 0
        assert r.sky_noise_ppm >= 0
        assert r.read_noise_ppm >= 0

    def test_total_noise_positive(self) -> None:
        r = compute_noise_budget(10000.0, 0.0, 1, 0.0)
        assert r.total_noise_ppm > 0

    def test_format_returns_string(self) -> None:
        r = compute_noise_budget(10000.0, 5.0, 9, 3.0)
        s = format_noise_budget(r)
        assert isinstance(s, str)
        assert "Noise" in s

    def test_flag_ok(self) -> None:
        r = compute_noise_budget(10000.0, 5.0, 9, 3.0)
        assert r.flag == "OK"
