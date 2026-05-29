"""Tests for Skills/signal_quality_grader.py."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from signal_quality_grader import (
    GradeFactor,
    SignalGrade,
    format_signal_grade,
    grade_signal_quality,
)


def _row(snr=15.0, fpp=0.05, dc=0.90, novelty=0.80, tic_id=12345, period=3.5):
    return {
        "tic_id": tic_id,
        "period_days": period,
        "snr": snr,
        "false_positive_probability": fpp,
        "detection_confidence": dc,
        "novelty_score": novelty,
    }


def test_grade_a():
    result = grade_signal_quality(_row(snr=20.0, fpp=0.01, dc=0.98, novelty=0.95))
    assert result.grade == "A"
    assert result.numeric_score >= 0.85


def test_grade_f_high_fpp():
    result = grade_signal_quality(_row(snr=3.0, fpp=0.95, dc=0.10, novelty=0.05))
    assert result.grade in ("D", "F")


def test_grade_b():
    result = grade_signal_quality(_row(snr=15.0, fpp=0.10, dc=0.80, novelty=0.70))
    assert result.grade in ("A", "B", "C")
    assert result.numeric_score >= 0.55


def test_returns_signal_grade_type():
    result = grade_signal_quality(_row())
    assert isinstance(result, SignalGrade)


def test_factors_count():
    result = grade_signal_quality(_row())
    assert len(result.factors) == 4


def test_factor_names():
    result = grade_signal_quality(_row())
    names = {f.name for f in result.factors}
    assert names == {"snr", "fpp", "detection_confidence", "novelty"}


def test_factor_contribution():
    result = grade_signal_quality(_row())
    for f in result.factors:
        assert isinstance(f, GradeFactor)
        assert abs(f.contribution - f.score * f.weight) < 1e-5


def test_missing_snr_fallback():
    row = _row()
    del row["snr"]
    result = grade_signal_quality(row)
    assert result.grade is not None  # still produces a grade


def test_all_missing_flag():
    result = grade_signal_quality({})
    assert result.flag == "INCOMPLETE"


def test_tic_id_extracted():
    result = grade_signal_quality(_row(tic_id=99999))
    assert result.tic_id == 99999


def test_period_extracted():
    result = grade_signal_quality(_row(period=5.123))
    assert result.period_days == pytest.approx(5.123)


def test_custom_weights():
    result = grade_signal_quality(
        _row(snr=10.0, fpp=0.50, dc=0.50, novelty=0.50),
        weights={"fpp": 1.0, "snr": 0.0, "detection_confidence": 0.0, "novelty": 0.0},
    )
    assert result.numeric_score == pytest.approx(0.50, abs=0.01)


def test_format_contains_grade():
    result = grade_signal_quality(_row())
    md = format_signal_grade(result)
    assert result.grade in md


def test_format_contains_tic_id():
    result = grade_signal_quality(_row(tic_id=55555))
    md = format_signal_grade(result)
    assert "55555" in md
