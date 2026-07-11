from __future__ import annotations

import pytest
from Skills.formal_acceptance import evaluate_case


def _case(kind: str) -> dict[str, object]:
    return {
        "case_id": "control",
        "kind": kind,
        "target_id": "TIC 1",
        "catalog_disposition": "CP" if kind == "confirmed" else "FP",
        "expected_period_days": [10.0],
        "period_relative_tolerance": 0.02,
        "max_fpp": 0.5,
        "min_fpp": 0.5,
        "command": "exo control",
        "result_path": "/tmp/control.json",
    }


def _row(period: float, fpp: float) -> dict[str, object]:
    return {
        "candidate_id": "control-s01",
        "period_days": period,
        "scores": {"false_positive_probability": fpp},
    }


def test_confirmed_requires_ephemeris_and_low_fpp() -> None:
    assert evaluate_case(_case("confirmed"), [_row(10.1, 0.2)])["passed"]
    assert not evaluate_case(_case("confirmed"), [_row(13.0, 0.2)])["passed"]
    assert not evaluate_case(_case("confirmed"), [_row(10.1, 0.8)])["passed"]


def test_false_positive_accepts_clean_rejection() -> None:
    result = evaluate_case(_case("false_positive"), [])
    assert result["passed"]
    assert result["reasons"] == ["no_signal_above_detection_threshold"]


def test_false_positive_detected_ephemeris_requires_high_fpp() -> None:
    assert evaluate_case(_case("false_positive"), [_row(10.1, 0.8)])["passed"]
    assert not evaluate_case(_case("false_positive"), [_row(10.1, 0.2)])["passed"]


def test_named_planet_probability_field_is_converted_to_fpp() -> None:
    case = _case("confirmed")
    case["planet_probability_field"] = "ensemble_planet_probability"
    row = _row(10.1, 0.9)
    row["ensemble_planet_probability"] = 0.8
    result = evaluate_case(case, [row])
    assert result["passed"]
    assert result["observed"][0]["fpp"] == pytest.approx(0.2)
