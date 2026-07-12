from __future__ import annotations

import json
from pathlib import Path

import pytest
from Skills.canonical_regression_eval import (
    _compare_to_baseline,
    _expected_passes,
    _load_formal_case,
    evaluate_suite,
)


def test_load_formal_case_reads_named_sample(tmp_path: Path) -> None:
    report = {
        "suite_id": "accepted",
        "toolkit_version": "1.0",
        "git_commit": "abc",
        "cases": [
            {
                "case_id": "planet",
                "passed": True,
                "observed": [{"period_days": 4.0}],
                "matching_catalog_ephemeris": [{"period_days": 4.0, "fpp": 0.2}],
            }
        ],
    }
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    current = _load_formal_case(
        {"source_path": "report.json", "source_case_id": "planet"}, tmp_path
    )
    assert current["passed"] is True
    assert current["period_days"] == 4.0
    assert current["fpp"] == 0.2


def test_expected_recovery_bounds_fail_closed() -> None:
    case = {"source_type": "injected_transit", "min_recovery_rate": 1.0}
    passed, reasons = _expected_passes(case, {"recovery_rate": 0.5})
    assert passed is False
    assert reasons == ["recovery_rate_below_minimum"]


def test_baseline_recovery_regression_is_sample_level_failure() -> None:
    case = {"source_type": "injected_transit", "max_recovery_rate_drop": 0.0}
    deltas, reasons = _compare_to_baseline(
        case,
        {"recovery_rate": 0.5, "period_days": 3.0},
        {"recovery_rate": 1.0, "period_days": 3.0},
    )
    assert deltas["recovery_rate"] == pytest.approx(-0.5)
    assert reasons == ["recovery_rate_regressed"]


def test_formal_acceptance_outcome_change_is_failure() -> None:
    case = {"source_type": "formal_acceptance"}
    _, reasons = _compare_to_baseline(case, {"passed": False}, {"passed": True})
    assert reasons == ["formal_acceptance_outcome_changed"]


def test_full_committed_suite_passes_without_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("Skills.canonical_regression_eval._git_commit", lambda: "test")
    repo_root = Path(__file__).resolve().parent.parent
    config = json.loads(
        (repo_root / "configs/canonical_regression_eval_v1.json").read_text(encoding="utf-8")
    )
    report = evaluate_suite(config, repo_root=repo_root)
    assert report["status"] == "PASS"
    assert [case["kind"] for case in report["cases"]] == [
        "confirmed_exoplanet",
        "known_false_positive",
        "injected_transit_control",
        "injected_transit_control",
    ]
