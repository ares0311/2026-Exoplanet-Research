import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from flux_gradient_analyzer import GradientResult, analyze_flux_gradient, format_gradient

# --- happy path ---

def test_flat_flux_zero_gradient():
    flux = [1.0] * 20
    result = analyze_flux_gradient(flux)
    assert result.max_gradient_ppm_per_step == pytest.approx(0.0, abs=1e-6)
    assert result.mean_abs_gradient_ppm == pytest.approx(0.0, abs=1e-6)
    assert result.n_steep_steps == 0
    assert result.flag == "OK"


def test_ramp_detected_large_gradient():
    # Abrupt jump of 0.1 in flux with mean ~1 → 100000 ppm/step >> 500
    flux = [1.0] * 10 + [1.1] * 10
    result = analyze_flux_gradient(flux)
    assert result.max_gradient_ppm_per_step > 500.0
    assert result.flag == "RAMP_DETECTED"


def test_n_steep_steps_counted_correctly():
    # Two large jumps
    flux = [1.0, 1.1, 1.0, 1.1, 1.0]
    result = analyze_flux_gradient(flux, threshold_ppm=500.0)
    assert result.n_steep_steps >= 2


def test_mean_abs_gradient_computed():
    # Two steps of equal size
    flux = [1.0, 1.001, 1.002]
    result = analyze_flux_gradient(flux)
    assert result.mean_abs_gradient_ppm > 0.0


# --- flag boundary ---

def test_flag_ok_small_gradient():
    # Tiny linear ramp: ~1 ppm/step → OK
    flux = [1.0 + 1e-6 * i for i in range(30)]
    result = analyze_flux_gradient(flux, threshold_ppm=500.0)
    assert result.flag == "OK"


def test_flag_ramp_detected_above_threshold():
    flux = [1.0, 2.0]  # jump = 1.0 relative to mean 1.5 → ~667000 ppm >> 500
    result = analyze_flux_gradient(flux, threshold_ppm=500.0)
    assert result.flag == "RAMP_DETECTED"


def test_custom_threshold_lower():
    flux = [1.0 + 0.0001 * i for i in range(10)]
    result_strict = analyze_flux_gradient(flux, threshold_ppm=10.0)
    result_loose = analyze_flux_gradient(flux, threshold_ppm=5000.0)
    assert result_strict.n_steep_steps >= result_loose.n_steep_steps


# --- edge cases ---

def test_single_point_insufficient_data():
    result = analyze_flux_gradient([1.0])
    assert result.flag == "INSUFFICIENT_DATA"
    assert result.max_gradient_ppm_per_step == 0.0


def test_empty_flux_insufficient_data():
    result = analyze_flux_gradient([])
    assert result.flag == "INSUFFICIENT_DATA"


def test_two_identical_points():
    result = analyze_flux_gradient([1.0, 1.0])
    assert result.max_gradient_ppm_per_step == pytest.approx(0.0, abs=1e-9)
    assert result.flag == "OK"


# --- return type ---

def test_returns_gradient_result():
    result = analyze_flux_gradient([1.0, 0.999, 1.001])
    assert isinstance(result, GradientResult)


def test_result_is_frozen():
    result = analyze_flux_gradient([1.0, 0.999, 1.001])
    with pytest.raises((AttributeError, TypeError)):
        result.flag = "CHANGED"


# --- format output ---

def test_format_contains_header():
    result = analyze_flux_gradient([1.0, 0.999, 1.001])
    text = format_gradient(result)
    assert "## Flux Gradient Analysis" in text


def test_format_contains_flag():
    result = analyze_flux_gradient([1.0, 0.999, 1.001])
    text = format_gradient(result)
    assert result.flag in text
