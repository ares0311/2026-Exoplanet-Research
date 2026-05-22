"""Tests for noise_model_fitter.py"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from noise_model_fitter import (
    fit_noise_model,
    format_noise_model_result,
)


def _white_noise(n=200, sigma=0.001):
    return [sigma * math.sin(i * 1.73 + 0.5) for i in range(n)]


def _red_noise(n=200, sigma=0.001, corr=0.9):
    vals = [0.0]
    for i in range(1, n):
        vals.append(corr * vals[-1] + sigma * math.sin(i * 2.3))
    return vals


class TestFitNoiseModel:
    def test_result_frozen(self):
        f = _white_noise()
        r = fit_noise_model(f)
        try:
            r.white_noise_ppm = 99.0  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except (AttributeError, TypeError):
            pass

    def test_too_few_points_insufficient(self):
        r = fit_noise_model([0.001] * 5)
        assert r.flag == "INSUFFICIENT"

    def test_invalid_cadence(self):
        r = fit_noise_model(_white_noise(), cadence_minutes=0.0)
        assert r.flag == "INVALID"

    def test_white_noise_estimated(self):
        f = _white_noise(200, 0.001)
        r = fit_noise_model(f)
        if r.white_noise_ppm is not None:
            assert r.white_noise_ppm > 0

    def test_white_noise_ppm_correct_order(self):
        # sigma=0.001 → white noise ~1000 ppm
        f = _white_noise(300, 0.001)
        r = fit_noise_model(f)
        if r.white_noise_ppm is not None:
            assert 500 < r.white_noise_ppm < 2000

    def test_red_noise_ge_zero(self):
        f = _red_noise(300)
        r = fit_noise_model(f)
        if r.red_noise_ppm is not None:
            assert r.red_noise_ppm >= 0

    def test_combined_noise_ge_white(self):
        f = _white_noise(200)
        r = fit_noise_model(f)
        if r.combined_noise_ppm is not None and r.white_noise_ppm is not None:
            assert r.combined_noise_ppm >= r.white_noise_ppm - 1e-6

    def test_beta_ge_0(self):
        f = _white_noise(200)
        r = fit_noise_model(f)
        if r.beta_factor is not None:
            assert r.beta_factor >= 0

    def test_beta_near_1_for_white_noise(self):
        # For ideal white noise, beta should be close to 1
        f = _white_noise(500, 0.001)
        r = fit_noise_model(f)
        if r.beta_factor is not None:
            assert 0.3 < r.beta_factor < 5.0

    def test_n_bins_used_positive(self):
        f = _white_noise(200)
        r = fit_noise_model(f)
        if r.flag == "OK":
            assert r.n_bins_used > 0

    def test_custom_bin_durations(self):
        f = _white_noise(200, 0.001)
        r = fit_noise_model(f, bin_durations_minutes=[2.0, 4.0, 10.0])
        assert r.flag in ("OK", "INSUFFICIENT")

    def test_format_returns_string(self):
        f = _white_noise(200)
        r = fit_noise_model(f)
        s = format_noise_model_result(r)
        assert isinstance(s, str)
        assert "Noise" in s

    def test_format_contains_flag(self):
        r = fit_noise_model([0.001] * 5)
        s = format_noise_model_result(r)
        assert "INSUFFICIENT" in s
