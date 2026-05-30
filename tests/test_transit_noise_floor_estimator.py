import sys
import math
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from transit_noise_floor_estimator import NoiseFloorResult, estimate_noise_floor, format_noise_floor


# --- happy path ---

def test_flat_flux_near_zero_rms():
    flux = [1.0] * 100
    result = estimate_noise_floor(flux, cadence_minutes=2.0, bin_minutes=30.0)
    assert result.oot_rms_ppm == pytest.approx(0.0, abs=1e-6)
    assert result.flag == "OK"


def test_noise_floor_decreases_with_binning():
    flux = [1.0 + 0.001 * (i % 3 - 1) for i in range(60)]
    result_fine = estimate_noise_floor(flux, cadence_minutes=2.0, bin_minutes=10.0)
    result_coarse = estimate_noise_floor(flux, cadence_minutes=2.0, bin_minutes=60.0)
    # More binning → lower noise floor
    assert result_coarse.noise_floor_ppm <= result_fine.noise_floor_ppm


def test_n_points_matches_input():
    flux = [1.0] * 42
    result = estimate_noise_floor(flux)
    assert result.n_points == 42


def test_cadence_stored_in_result():
    flux = [1.0] * 10
    result = estimate_noise_floor(flux, cadence_minutes=5.0)
    assert result.cadence_minutes == 5.0


def test_default_cadence_and_bin():
    flux = [1.0] * 20
    result = estimate_noise_floor(flux)
    assert result.cadence_minutes == 2.0


# --- flag boundary ---

def test_flag_ok_low_noise():
    # Very flat flux → tiny noise floor → OK
    flux = [1.0 + 1e-7 * i for i in range(50)]
    result = estimate_noise_floor(flux, cadence_minutes=2.0, bin_minutes=30.0)
    assert result.flag == "OK"


def test_flag_high_noise():
    # Large oscillations → rms > 500 ppm after binning
    # 10% amplitude oscillation → rms ~ 70700 ppm, bin_factor=1 (cadence >= bin) → floor > 500
    flux = [1.0 + 0.1 * (i % 2) for i in range(20)]
    result = estimate_noise_floor(flux, cadence_minutes=60.0, bin_minutes=30.0)
    assert result.flag == "HIGH_NOISE"


# --- edge cases ---

def test_empty_flux_returns_no_data():
    result = estimate_noise_floor([])
    assert result.flag == "NO_DATA"
    assert result.n_points == 0
    assert result.noise_floor_ppm == 0.0


def test_single_point():
    result = estimate_noise_floor([1.0])
    assert result.n_points == 1
    assert result.oot_rms_ppm == pytest.approx(0.0, abs=1e-9)


def test_bin_factor_minimum_one():
    # cadence >= bin → bin_factor clamped to 1
    flux = [1.0] * 10
    result_a = estimate_noise_floor(flux, cadence_minutes=60.0, bin_minutes=30.0)
    assert result_a.noise_floor_ppm == result_a.oot_rms_ppm


# --- return type ---

def test_returns_noise_floor_result():
    result = estimate_noise_floor([1.0] * 10)
    assert isinstance(result, NoiseFloorResult)


def test_result_is_frozen():
    result = estimate_noise_floor([1.0] * 10)
    with pytest.raises((AttributeError, TypeError)):
        result.flag = "CHANGED"


# --- format output ---

def test_format_contains_header():
    result = estimate_noise_floor([1.0] * 10)
    text = format_noise_floor(result)
    assert "## Transit Noise Floor Estimate" in text


def test_format_contains_flag():
    result = estimate_noise_floor([1.0] * 10)
    text = format_noise_floor(result)
    assert result.flag in text
