import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from candidate_recovery_checker import RecoveryCheckResult, check_recovery, format_recovery_check

# --- helpers ---

def _sig(period=10.0, epoch=100.0):
    return {"period_days": period, "epoch_bjd": epoch}


# --- happy path ---

def test_exact_match_recovered():
    known = [_sig(10.0, 100.0)]
    detected = [_sig(10.0, 100.0)]
    result = check_recovery(known, detected)
    assert result.n_recovered == 1
    assert result.n_missed == 0
    assert result.flag == "OK"


def test_all_recovered_multiple():
    known = [_sig(5.0, 50.0), _sig(10.0, 100.0), _sig(15.0, 200.0)]
    detected = [_sig(5.0, 50.0), _sig(10.0, 100.0), _sig(15.0, 200.0)]
    result = check_recovery(known, detected)
    assert result.n_recovered == 3
    assert result.n_missed == 0
    assert result.flag == "OK"


def test_recovered_indices_correct():
    known = [_sig(5.0, 50.0), _sig(10.0, 100.0)]
    detected = [_sig(5.0, 50.0)]
    result = check_recovery(known, detected)
    assert 0 in result.recovered_indices
    assert 1 in result.missed_indices


def test_period_tolerance_allows_small_mismatch():
    known = [_sig(10.0, 100.0)]
    # 1% tolerance: 10.0 * 0.01 = 0.1 → allow period in [9.9, 10.1]
    detected = [_sig(10.05, 100.0)]
    result = check_recovery(known, detected, period_tol_frac=0.01)
    assert result.n_recovered == 1


# --- flag boundary ---

def test_flag_incomplete_recovery_when_missed():
    known = [_sig(5.0, 50.0), _sig(10.0, 100.0)]
    detected = [_sig(5.0, 50.0)]
    result = check_recovery(known, detected)
    assert result.flag == "INCOMPLETE_RECOVERY"
    assert result.n_missed == 1


def test_flag_ok_when_all_recovered():
    known = [_sig(7.0, 70.0)]
    detected = [_sig(7.0, 70.0)]
    result = check_recovery(known, detected)
    assert result.flag == "OK"


def test_none_detected_all_missed():
    known = [_sig(5.0, 50.0), _sig(10.0, 100.0)]
    detected = []
    result = check_recovery(known, detected)
    assert result.n_missed == 2
    assert result.n_recovered == 0
    assert result.flag == "INCOMPLETE_RECOVERY"


def test_period_mismatch_exceeds_tolerance():
    known = [_sig(10.0, 100.0)]
    # Period differs by 5% → outside 1% tolerance
    detected = [_sig(10.5, 100.0)]
    result = check_recovery(known, detected, period_tol_frac=0.01)
    assert result.n_missed == 1
    assert result.flag == "INCOMPLETE_RECOVERY"


# --- edge cases ---

def test_empty_known():
    result = check_recovery([], [_sig(10.0, 100.0)])
    assert result.n_known == 0
    assert result.n_recovered == 0
    assert result.flag == "OK"


def test_empty_both():
    result = check_recovery([], [])
    assert result.n_known == 0
    assert result.flag == "OK"


def test_n_known_correct():
    known = [_sig(), _sig(5.0, 50.0), _sig(15.0, 150.0)]
    result = check_recovery(known, [])
    assert result.n_known == 3


# --- return type ---

def test_returns_recovery_check_result():
    result = check_recovery([_sig()], [_sig()])
    assert isinstance(result, RecoveryCheckResult)


def test_result_is_frozen():
    result = check_recovery([_sig()], [_sig()])
    with pytest.raises((AttributeError, TypeError)):
        result.flag = "CHANGED"


# --- format output ---

def test_format_contains_header():
    result = check_recovery([_sig()], [_sig()])
    text = format_recovery_check(result)
    assert "## Candidate Recovery Check" in text


def test_format_contains_flag():
    result = check_recovery([_sig()], [])
    text = format_recovery_check(result)
    assert result.flag in text
