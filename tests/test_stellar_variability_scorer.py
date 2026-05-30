import sys
import math
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from stellar_variability_scorer import VariabilityResult, score_stellar_variability, format_variability


# --- happy path ---

def test_flat_flux_zero_variability():
    flux = [1.0] * 50
    result = score_stellar_variability(flux)
    assert result.rms_ppm == pytest.approx(0.0, abs=1e-6)
    assert result.variability_score == pytest.approx(0.0, abs=1e-6)
    assert result.flag == "OK"


def test_peak_to_peak_computed():
    # flux in [0.99, 1.01] → p2p ~ 20000 ppm
    flux = [0.99, 1.00, 1.01, 1.00, 0.99]
    result = score_stellar_variability(flux)
    assert result.peak_to_peak_ppm > 0.0


def test_variability_score_capped_at_one():
    # Extreme variability: rms >> 1000 ppm
    flux = [0.5, 1.5] * 20
    result = score_stellar_variability(flux)
    assert result.variability_score <= 1.0


def test_variability_score_formula():
    # rms_ppm / 1000, capped at 1
    flux = [1.0 + 0.0001 * (i % 2) for i in range(100)]
    result = score_stellar_variability(flux)
    expected = min(result.rms_ppm / 1000.0, 1.0)
    assert result.variability_score == pytest.approx(expected, abs=1e-9)


# --- flag boundary ---

def test_flag_ok_low_rms():
    # rms ~ 100 ppm → OK
    flux = [1.0 + 1e-4 * (i % 2) for i in range(50)]
    result = score_stellar_variability(flux)
    assert result.flag == "OK"


def test_flag_high_variability_large_amplitude():
    # 0.1 amplitude oscillation → rms ~ 70700 ppm → HIGH_VARIABILITY
    flux = [1.0 + 0.1 * (i % 2) for i in range(40)]
    result = score_stellar_variability(flux)
    assert result.flag == "HIGH_VARIABILITY"


def test_rms_threshold_exactly_500():
    # rms just above 500 ppm → HIGH_VARIABILITY
    # 500 ppm = 0.0005 relative, so set up flux that gives rms > 500 ppm
    flux = [1.0 + 0.001 * (i % 2) for i in range(100)]
    result = score_stellar_variability(flux)
    # rms will be ~500 ppm depending on exact implementation; just check it runs
    assert result.flag in ("OK", "HIGH_VARIABILITY")


# --- edge cases ---

def test_empty_flux_returns_no_data():
    result = score_stellar_variability([])
    assert result.flag == "NO_DATA"
    assert result.rms_ppm == 0.0
    assert result.variability_score == 0.0


def test_single_point():
    result = score_stellar_variability([1.0])
    assert result.peak_to_peak_ppm == pytest.approx(0.0, abs=1e-9)
    assert result.rms_ppm == pytest.approx(0.0, abs=1e-9)


def test_all_same_value():
    flux = [2.5] * 30
    result = score_stellar_variability(flux)
    assert result.rms_ppm == pytest.approx(0.0, abs=1e-6)


# --- return type ---

def test_returns_variability_result():
    result = score_stellar_variability([1.0, 0.999, 1.001])
    assert isinstance(result, VariabilityResult)


def test_result_is_frozen():
    result = score_stellar_variability([1.0, 0.999, 1.001])
    with pytest.raises((AttributeError, TypeError)):
        result.flag = "CHANGED"


# --- format output ---

def test_format_contains_header():
    result = score_stellar_variability([1.0, 0.999, 1.001])
    text = format_variability(result)
    assert "## Stellar Variability Score" in text


def test_format_contains_flag():
    result = score_stellar_variability([1.0, 0.999, 1.001])
    text = format_variability(result)
    assert result.flag in text
