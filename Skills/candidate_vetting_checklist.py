"""Generate a pass/fail vetting checklist from a scored candidate dict."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class ChecklistItem:
    name: str
    passed: bool
    value: str
    reason: str


@dataclass(frozen=True)
class VettingChecklistResult:
    tic_id: str
    n_passed: int
    n_failed: int
    n_total: int
    overall: str  # PASS | FAIL | PARTIAL
    items: tuple[ChecklistItem, ...]
    flag: str


def _get(d: dict, *keys: str, default=None):
    for k in keys:
        v = d.get(k)
        if v is not None:
            return v
        scores = d.get("scores", {})
        v = scores.get(k)
        if v is not None:
            return v
    return default


def build_vetting_checklist(
    candidate: dict,
    fpp_threshold: float = 0.10,
    dc_threshold: float = 0.80,
    snr_threshold: float = 7.0,
    n_transits_min: int = 2,
) -> VettingChecklistResult:
    """
    Build a pass/fail checklist from candidate pipeline output.

    Checks: FPP, detection confidence, SNR, n_transits, pathway,
    odd/even depth consistency, secondary eclipse.
    """
    tic_id = str(candidate.get("tic_id", "")).strip()
    if not tic_id:
        return VettingChecklistResult(
            tic_id=tic_id, n_passed=0, n_failed=0, n_total=0,
            overall="FAIL", items=(), flag="MISSING_TIC_ID",
        )

    items: list[ChecklistItem] = []

    # FPP check
    fpp = _get(candidate, "false_positive_probability", "fpp")
    if fpp is not None:
        passed = float(fpp) <= fpp_threshold
        items.append(ChecklistItem(
            name="FPP",
            passed=passed,
            value=f"{float(fpp):.4f}",
            reason=f"FPP {'<=' if passed else '>'} {fpp_threshold}",
        ))

    # Detection confidence
    dc = _get(candidate, "detection_confidence")
    if dc is not None:
        passed = float(dc) >= dc_threshold
        items.append(ChecklistItem(
            name="Detection confidence",
            passed=passed,
            value=f"{float(dc):.4f}",
            reason=f"DC {'>=' if passed else '<'} {dc_threshold}",
        ))

    # SNR
    snr = _get(candidate, "snr", "best_snr")
    if snr is not None:
        passed = float(snr) >= snr_threshold
        items.append(ChecklistItem(
            name="SNR",
            passed=passed,
            value=f"{float(snr):.2f}",
            reason=f"SNR {'>=' if passed else '<'} {snr_threshold}",
        ))

    # N transits
    n_tr = candidate.get("n_transits") or candidate.get("transit_count")
    if n_tr is not None:
        passed = int(n_tr) >= n_transits_min
        items.append(ChecklistItem(
            name="N transits",
            passed=passed,
            value=str(int(n_tr)),
            reason=f"N {'>=' if passed else '<'} {n_transits_min}",
        ))

    # Pathway
    pathway = candidate.get("pathway", "")
    positive_pathways = {"tfop_ready", "planet_hunters_discussion", "kepler_archive_candidate"}
    if pathway:
        passed = pathway in positive_pathways
        items.append(ChecklistItem(
            name="Pathway",
            passed=passed,
            value=pathway,
            reason=(
                "Pathway supports follow-up" if passed else "Pathway does not support follow-up"
            ),
        ))

    # Odd/even depth consistency
    odd_even_flag = candidate.get("odd_even_flag") or _get(candidate, "odd_even_flag")
    if odd_even_flag is not None:
        passed = str(odd_even_flag).upper() == "OK"
        items.append(ChecklistItem(
            name="Odd/even depth",
            passed=passed,
            value=str(odd_even_flag),
            reason=(
                "Odd/even depths consistent" if passed else "Odd/even depth mismatch (EB flag)"
            ),
        ))

    # Secondary eclipse
    secondary_snr = _get(candidate, "secondary_snr")
    if secondary_snr is not None:
        no_secondary = float(secondary_snr) < 3.0
        items.append(ChecklistItem(
            name="Secondary eclipse",
            passed=no_secondary,
            value=f"{float(secondary_snr):.2f}",
            reason=(
                "No significant secondary eclipse" if no_secondary else "Secondary eclipse detected"
            ),
        ))

    n_total = len(items)
    n_passed = sum(1 for it in items if it.passed)
    n_failed = n_total - n_passed

    if n_total == 0:
        overall = "PARTIAL"
    elif n_failed == 0:
        overall = "PASS"
    elif n_passed == 0:
        overall = "FAIL"
    else:
        overall = "PARTIAL"

    return VettingChecklistResult(
        tic_id=tic_id,
        n_passed=n_passed,
        n_failed=n_failed,
        n_total=n_total,
        overall=overall,
        items=tuple(items),
        flag="OK",
    )


def format_checklist(r: VettingChecklistResult) -> str:
    lines = [
        f"## Vetting Checklist - TIC {r.tic_id}",
        f"**Overall: {r.overall}** ({r.n_passed}/{r.n_total} passed)\n",
        "| Check | Result | Value | Reason |",
        "|---|---|---|---|",
    ]
    for it in r.items:
        symbol = "PASS" if it.passed else "FAIL"
        lines.append(f"| {it.name} | {symbol} | {it.value} | {it.reason} |")
    return "\n".join(lines)


def _cli() -> int:
    p = argparse.ArgumentParser(description="Build candidate vetting checklist.")
    p.add_argument("candidate_json", help="JSON string or @file")
    args = p.parse_args()
    raw = args.candidate_json
    if raw.startswith("@"):
        with open(raw[1:]) as f:
            candidate = json.load(f)
    else:
        candidate = json.loads(raw)
    r = build_vetting_checklist(candidate)
    print(format_checklist(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
