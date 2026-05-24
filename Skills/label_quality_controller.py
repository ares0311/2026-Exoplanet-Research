"""Quality-control checks on a LabelManifest before CNN training.

Applies three checks to each LabelRecord:
  1. Conflict check — records with unresolved conflict fail if
     ``require_agreement_on_conflict=True``.
  2. Ephemeris plausibility — period must be within [min_period_days,
     max_period_days] when provided; epoch must be a finite float.
  3. Confidence floor — records below ``min_confidence`` are rejected.

Public API
----------
LabelQcResult(n_input, n_passed, n_rejected, rejection_reasons,
              passed_manifest, flag)
run_label_qc(manifest, *, min_confidence, min_period_days,
             max_period_days, require_agreement_on_conflict) -> LabelQcResult
format_qc_result(result) -> str
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

# Allow importing sibling Skills as modules
sys.path.insert(0, str(Path(__file__).resolve().parent))

from multi_source_label_assembler import LabelManifest, LabelRecord  # noqa: E402

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LabelQcResult:
    """Result of running QC over a :class:`LabelManifest`."""

    n_input: int
    n_passed: int
    n_rejected: int
    rejection_reasons: dict     # {"conflict": n, "period_out_of_range": n,
                                #  "low_confidence": n, "invalid_ephemeris": n}
    passed_manifest: LabelManifest
    flag: str  # "OK" | "ALL_REJECTED" | "INVALID"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_label_qc(
    manifest: LabelManifest,
    *,
    min_confidence: float = 0.6,
    min_period_days: float = 0.1,
    max_period_days: float = 1000.0,
    require_agreement_on_conflict: bool = True,
) -> LabelQcResult:
    """Run QC checks on *manifest* and return a filtered manifest.

    Args:
        manifest:                      Input :class:`LabelManifest`.
        min_confidence:                Minimum confidence to pass.
        min_period_days:               Minimum allowed period (if set).
        max_period_days:               Maximum allowed period (if set).
        require_agreement_on_conflict: If True, reject conflicted records.

    Returns:
        :class:`LabelQcResult`
    """
    if not hasattr(manifest, "records"):
        empty_manifest = LabelManifest(
            records=(), n_positive=0, n_negative=0, n_conflicts=0,
            sources=(), created_at="", flag="INVALID",
        )
        return LabelQcResult(
            n_input=0, n_passed=0, n_rejected=0,
            rejection_reasons={},
            passed_manifest=empty_manifest,
            flag="INVALID",
        )

    reasons: dict[str, int] = {
        "conflict": 0,
        "period_out_of_range": 0,
        "low_confidence": 0,
        "invalid_ephemeris": 0,
    }
    passed: list[LabelRecord] = []

    for rec in manifest.records:
        rejected = False

        if require_agreement_on_conflict and rec.conflict:
            reasons["conflict"] += 1
            rejected = True

        if not rejected and rec.confidence < min_confidence:
            reasons["low_confidence"] += 1
            rejected = True

        if not rejected and rec.period_days is not None:
            try:
                p = float(rec.period_days)
                if not (min_period_days <= p <= max_period_days):
                    reasons["period_out_of_range"] += 1
                    rejected = True
            except (ValueError, TypeError):
                reasons["invalid_ephemeris"] += 1
                rejected = True

        if not rejected and rec.epoch is not None:
            try:
                e = float(rec.epoch)
                if not (-1e9 < e < 1e12):
                    reasons["invalid_ephemeris"] += 1
                    rejected = True
            except (ValueError, TypeError):
                reasons["invalid_ephemeris"] += 1
                rejected = True

        if not rejected:
            passed.append(rec)

    n_pos = sum(1 for r in passed if r.label == 1)
    n_neg = sum(1 for r in passed if r.label == 0)
    sources = tuple(sorted({r.source for r in passed}))

    passed_manifest = LabelManifest(
        records=tuple(passed),
        n_positive=n_pos,
        n_negative=n_neg,
        n_conflicts=0,
        sources=sources,
        created_at=manifest.created_at,
        flag="OK" if passed else "EMPTY",
    )

    n_input = len(manifest.records)
    n_passed = len(passed)
    n_rejected = n_input - n_passed

    if n_input == 0:
        flag = "INVALID"
    elif n_passed == 0:
        flag = "ALL_REJECTED"
    else:
        flag = "OK"

    return LabelQcResult(
        n_input=n_input,
        n_passed=n_passed,
        n_rejected=n_rejected,
        rejection_reasons=reasons,
        passed_manifest=passed_manifest,
        flag=flag,
    )


def format_qc_result(result: LabelQcResult) -> str:
    """Return a Markdown summary of a :class:`LabelQcResult`."""
    lines = [
        "## Label QC Result",
        "",
        f"**Flag**: {result.flag}",
        f"**Input records**: {result.n_input}",
        f"**Passed**: {result.n_passed}",
        f"**Rejected**: {result.n_rejected}",
        "",
        "### Rejection Reasons",
    ]
    for reason, count in result.rejection_reasons.items():
        lines.append(f"- {reason}: {count}")
    lines += [
        "",
        "### Passed Manifest",
        f"- Positive: {result.passed_manifest.n_positive}",
        f"- Negative: {result.passed_manifest.n_negative}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(
        prog="label_quality_controller",
        description="Run QC checks on a label manifest JSON.",
    )
    parser.add_argument("manifest", help="Path to label manifest JSON.")
    parser.add_argument("--min-confidence", type=float, default=0.6)
    parser.add_argument("--min-period", type=float, default=0.1)
    parser.add_argument("--max-period", type=float, default=1000.0)
    parser.add_argument("--allow-conflicts", action="store_true")
    args = parser.parse_args(argv)

    data = json.loads(Path(args.manifest).read_text())
    records = tuple(
        LabelRecord(
            tic_id=r["tic_id"], label=r["label"], source=r["source"],
            confidence=r.get("confidence", 1.0),
            period_days=r.get("period_days"), epoch=r.get("epoch"),
            duration_hours=r.get("duration_hours"),
            conflict=r.get("conflict", False),
        )
        for r in data.get("records", [])
    )
    n_pos = sum(1 for r in records if r.label == 1)
    n_neg = sum(1 for r in records if r.label == 0)
    manifest = LabelManifest(
        records=records, n_positive=n_pos, n_negative=n_neg,
        n_conflicts=data.get("n_conflicts", 0),
        sources=tuple(data.get("sources", [])),
        created_at=data.get("created_at", ""),
        flag=data.get("flag", "OK"),
    )

    result = run_label_qc(
        manifest,
        min_confidence=args.min_confidence,
        min_period_days=args.min_period,
        max_period_days=args.max_period,
        require_agreement_on_conflict=not args.allow_conflicts,
    )
    print(format_qc_result(result))
    return 0 if result.flag == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
