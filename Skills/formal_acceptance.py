"""Evaluate catalog-backed production CLI controls and write a durable report."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from exo_toolkit import __version__


def _matches_period(observed: float, expected: float, tolerance: float) -> bool:
    return abs(observed - expected) / expected <= tolerance


def evaluate_case(case: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    expected = [float(value) for value in case["expected_period_days"]]
    tolerance = float(case["period_relative_tolerance"])
    observed = [
        {
            "candidate_id": row.get("candidate_id"),
            "period_days": row.get("period_days"),
            "fpp": row.get("scores", {}).get("false_positive_probability"),
        }
        for row in rows
    ]
    matching = [
        item
        for item in observed
        if isinstance(item["period_days"], (int, float))
        and any(
            _matches_period(float(item["period_days"]), period, tolerance)
            for period in expected
        )
    ]
    reasons: list[str] = []
    passed = False
    if case["kind"] == "confirmed":
        if not matching:
            reasons.append("no_catalog_ephemeris_recovered")
        else:
            qualified = [
                item
                for item in matching
                if isinstance(item["fpp"], (int, float))
                and float(item["fpp"]) < float(case["max_fpp"])
            ]
            passed = bool(qualified)
            if not passed:
                reasons.append("catalog_ephemeris_fpp_not_below_ceiling")
    elif not rows:
        passed = True
        reasons.append("no_signal_above_detection_threshold")
    elif not matching:
        passed = True
        reasons.append("catalog_false_positive_ephemeris_not_recovered")
    else:
        passed = all(
            isinstance(item["fpp"], (int, float))
            and float(item["fpp"]) > float(case["min_fpp"])
            for item in matching
        )
        if not passed:
            reasons.append("catalog_false_positive_fpp_not_above_floor")

    return {
        "case_id": case["case_id"],
        "kind": case["kind"],
        "target_id": case["target_id"],
        "catalog_disposition": case["catalog_disposition"],
        "expected_period_days": expected,
        "period_relative_tolerance": tolerance,
        "command": case["command"],
        "result_path": case["result_path"],
        "observed": observed,
        "matching_catalog_ephemeris": matching,
        "passed": passed,
        "reasons": reasons,
    }


def evaluate_suite(config: dict[str, Any]) -> dict[str, Any]:
    cases = config["cases"]
    start = time.monotonic()
    results: list[dict[str, Any]] = []
    print(f"Formal acceptance: {len(cases)} cases", flush=True)
    for index, case in enumerate(cases, 1):
        result_path = Path(case["result_path"])
        rows = json.loads(result_path.read_text(encoding="utf-8"))
        results.append(evaluate_case(case, rows))
        elapsed = time.monotonic() - start
        rate = index / elapsed if elapsed else 0.0
        remaining = (len(cases) - index) / rate if rate else 0.0
        print(
            f"  [{index}/{len(cases)}] {case['case_id']} "
            f"{'PASS' if results[-1]['passed'] else 'FAIL'} "
            f"elapsed={elapsed:.1f}s ETA={remaining:.1f}s",
            flush=True,
        )
    return {
        "schema_version": 1,
        "suite_id": config["suite_id"],
        "toolkit_version": __version__,
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip(),
        "evaluated_at": datetime.now(UTC).isoformat(),
        "status": "PASS" if all(result["passed"] for result in results) else "FAIL",
        "cases": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    report = evaluate_suite(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Formal acceptance {report['status']}: {args.output}", flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
