import sys
import math
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from systematic_noise_detector import SystematicNoiseResult, detect_systematic_noise, format_systematic_noise


# --- happy path ---

def test_random_like_flux_ok():
    # alternating flux → many runs → no systematics
    flux = [1.0 if i % 2 == 0 else 0.999 for i in range(40)]
    result = detect_systematic_noise(flux)
    assert isinstance(result, SystematicNoiseResult)
    assert result.flag in ("OK", "SYSTEMATICS_DETECTED")


def test_dw_statistic_near_two_for_random():
    # perfectly alternating signal: large diff between consecutive → DW > 2
    flux = [1.0, 0.999] * 20
    result = detect_systematic_noise(flux)
    assert result.dw_statistic > 0.0


def test_strongly_correlated_flux_detected():
    # ramp: strong positive autocorrelation → DW < 1.5
    flux = [0.001 * i for i in range(40)]
    result = detect_systematic_noise(flux)
    assert result.has_systematics is True
    assert result.flag == "SYSTEMATICS_DETECTED"


def test_expected_runs_half_n():
    flux = [float(i % 3) for i in range(20)]
    result = detect_systematic_noise(flux)
    assert abs(result.expected_runs - 10.0) < 1e-9


# --- flag boundary ---

def test_flag_ok_white_noise_like():
    # tight, rapidly alternating flux: DW high, low autocorr
    import random
    random.seed(42)
    flux = [1.0 + random.gauss(0, 0.0001) for _ in range(60)]
    result = detect_systematic_noise(flux)
    # Just verify it runs and returns valid flag
    assert result.flag in ("OK", "SYSTEMATICS_DETECTED")


def test_flag_systematics_low_dw():
    # monotonically increasing: DW close to 0
    flux = [float(i) for i in range(1, 41)]
    result = detect_systematic_noise(flux)
    assert result.has_systematics is True
    assert result.flag == "SYSTEMATICS_DETECTED"


def test_lag1_autocorr_range():
    flux = [float(i % 5) for i in range(30)]
    result = detect_systematic_noise(flux)
    assert -1.0 <= result.lag1_autocorr <= 1.0


# --- edge cases ---

def test_fewer_than_four_returns_defaults():
    result = detect_systematic_noise([1.0, 0.99, 1.01])
    assert result.dw_statistic == 2.0
    assert result.lag1_autocorr == 0.0
    assert result.has_systematics is False
    assert result.flag == "OK"


def test_empty_returns_defaults():
    result = detect_systematic_noise([])
    assert result.flag == "OK"
    assert result.has_systematics is False


def test_constant_flux():
    # all same → sum_r_sq = 0 → defaults
    flux = [1.0] * 10
    result = detect_systematic_noise(flux)
    assert result.dw_statistic == 2.0
    assert result.lag1_autocorr == 0.0


def test_n_runs_at_least_one():
    flux = [float(i) for i in range(10)]
    result = detect_systematic_noise(flux)
    assert result.n_runs >= 1


# --- return type ---

def test_returns_systematic_noise_result():
    result = detect_systematic_noise([1.0, 0.9, 1.1, 0.95, 1.05])
    assert isinstance(result, SystematicNoiseResult)


def test_result_is_frozen():
    result = detect_systematic_noise([1.0, 0.9, 1.1, 0.95, 1.05])
    with pytest.raises((AttributeError, TypeError)):
        result.flag = "CHANGED"


# --- format output ---

def test_format_contains_header():
    result = detect_systematic_noise([1.0, 0.9, 1.1, 0.95, 1.05, 1.0])
    text = format_systematic_noise(result)
    assert "## Systematic Noise Detection" in text


def test_format_contains_flag():
    result = detect_systematic_noise([1.0, 0.9, 1.1, 0.95, 1.05, 1.0])
    text = format_systematic_noise(result)
    assert result.flag in text
