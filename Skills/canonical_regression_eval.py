"""Run the offline canonical exoplanet regression suite.

The suite keeps catalog-backed controls separate from deterministic injected
controls, then reports every case beside the committed baseline.  It does not
fetch archive data or make discovery claims.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

# Direct ``Skills/*.py`` execution adds Skills/, not the repository root.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from exo_toolkit import __version__  # noqa: E402
from Skills.injection_recovery import make_mock_lc, run_injection_recovery  # noqa: E402


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _load_formal_case(case: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    report_path = repo_root / str(case["source_path"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    source_case_id = str(case["source_case_id"])
    try:
        source = next(item for item in report["cases"] if item["case_id"] == source_case_id)
    except StopIteration as exc:
        raise ValueError(f"formal acceptance case not found: {source_case_id}") from exc

    matches = source["matching_catalog_ephemeris"]
    best = matches[0] if matches else None
    return {
        "source_suite_id": report["suite_id"],
        "source_toolkit_version": report["toolkit_version"],
        "source_git_commit": report["git_commit"],
        "source_case_id": source_case_id,
        "passed": bool(source["passed"]),
        "observed_count": len(source["observed"]),
        "matching_count": len(matches),
        "period_days": best.get("period_days") if best else None,
        "fpp": best.get("fpp") if best else None,
    }


def _run_injected_case(case: dict[str, Any]) -> dict[str, Any]:
    seed = int(case["seed"])
    lc = make_mock_lc(
        baseline_days=float(case["baseline_days"]),
        cadence_minutes=float(case["cadence_minutes"]),
        noise_ppm=float(case["noise_ppm"]),
        rng=np.random.default_rng(seed),
    )
    grid = run_injection_recovery(
        lc,
        target_id=str(case["target_id"]),
        mission=str(case["mission"]),  # type: ignore[arg-type]
        period_grid=np.array([float(case["period_days"])]),
        depth_grid=np.array([float(case["depth_ppm"])]),
        duration_hours=float(case["duration_hours"]),
        n_trials=int(case["trials"]),
        min_snr=float(case["min_snr"]),
        rng=np.random.default_rng(seed + 1),
    )
    recovered = [item for item in grid.results if item.recovered]
    best = recovered[0] if recovered else None
    return {
        "seed": seed,
        "recovered_count": len(recovered),
        "trial_count": len(grid.results),
        "recovery_rate": float(grid.recovery_rate[0, 0]),
        "period_days": best.recovered_period_days if best else None,
        "depth_ppm": best.recovered_depth_ppm if best else None,
        "snr": best.recovered_snr if best else None,
    }


def _expected_passes(case: dict[str, Any], current: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if case["source_type"] == "formal_acceptance":
        if not current["passed"]:
            reasons.append("recorded_formal_acceptance_case_failed")
    else:
        minimum = float(case.get("min_recovery_rate", 0.0))
        maximum = float(case.get("max_recovery_rate", 1.0))
        rate = float(current["recovery_rate"])
        if rate < minimum:
            reasons.append("recovery_rate_below_minimum")
        if rate > maximum:
            reasons.append("recovery_rate_above_maximum")
    return not reasons, reasons


def _numeric_delta(current: Any, baseline: Any) -> float | None:
    if not isinstance(current, (int, float)) or not isinstance(baseline, (int, float)):
        return None
    return float(current) - float(baseline)


def _compare_to_baseline(
    case: dict[str, Any], current: dict[str, Any], baseline: dict[str, Any] | None
) -> tuple[dict[str, float | bool | None], list[str]]:
    if baseline is None:
        return {}, []
    keys = ("observed_count", "matching_count", "recovery_rate", "period_days", "fpp")
    deltas: dict[str, float | bool | None] = {
        key: _numeric_delta(current.get(key), baseline.get(key)) for key in keys
    }
    reasons: list[str] = []
    if case["source_type"] == "formal_acceptance":
        if bool(current["passed"]) != bool(baseline["passed"]):
            reasons.append("formal_acceptance_outcome_changed")
    else:
        baseline_rate = float(baseline["recovery_rate"])
        allowed_drop = float(case.get("max_recovery_rate_drop", 0.0))
        if float(current["recovery_rate"]) < baseline_rate - allowed_drop:
            reasons.append("recovery_rate_regressed")
    return deltas, reasons


def evaluate_suite(
    config: dict[str, Any], *, repo_root: Path, baseline: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Evaluate every configured sample and return a comparison report."""
    baseline_cases = {
        item["case_id"]: item["current"] for item in baseline.get("cases", [])
    } if baseline else {}
    cases = config["cases"]
    started = time.monotonic()
    results: list[dict[str, Any]] = []
    print(f"Canonical regression eval: {len(cases)} cases", flush=True)
    for index, case in enumerate(cases, 1):
        if case["source_type"] == "formal_acceptance":
            current = _load_formal_case(case, repo_root)
        elif case["source_type"] == "injected_transit":
            current = _run_injected_case(case)
        else:
            raise ValueError(f"unsupported source_type: {case['source_type']}")

        expected_ok, reasons = _expected_passes(case, current)
        deltas, regression_reasons = _compare_to_baseline(
            case, current, baseline_cases.get(case["case_id"])
        )
        reasons.extend(regression_reasons)
        result = {
            "case_id": case["case_id"],
            "dataset_id": case["dataset_id"],
            "kind": case["kind"],
            "source_type": case["source_type"],
            "current": current,
            "baseline": baseline_cases.get(case["case_id"]),
            "deltas": deltas,
            "passed": expected_ok and not regression_reasons,
            "reasons": reasons,
        }
        results.append(result)
        elapsed = time.monotonic() - started
        rate = index / elapsed if elapsed else 0.0
        remaining = (len(cases) - index) / rate if rate else 0.0
        print(
            f"  [{index}/{len(cases)}] {case['case_id']} "
            f"{'PASS' if result['passed'] else 'FAIL'} "
            f"elapsed={elapsed:.1f}s ETA={remaining:.1f}s",
            flush=True,
        )

    return {
        "schema_version": 1,
        "suite_id": config["suite_id"],
        "toolkit_version": __version__,
        "git_commit": _git_commit(),
        "evaluated_at": datetime.now(UTC).isoformat(),
        "baseline_suite_id": baseline.get("suite_id") if baseline else None,
        "dataset_ids": config["dataset_ids"],
        "status": "PASS" if all(item["passed"] for item in results) else "FAIL",
        "cases": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    baseline = (
        json.loads(args.baseline.read_text(encoding="utf-8")) if args.baseline else None
    )
    report = evaluate_suite(config, repo_root=REPO_ROOT, baseline=baseline)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Canonical regression eval {report['status']}: {args.output}", flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
