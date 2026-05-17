"""Aggregate all pipeline diagnostics into a structured pass/fail scorecard.

Produces a machine-readable and human-readable summary of every vetting check,
with an overall recommendation: PASS, WARN, or FAIL.

Public API
----------
VettingCheck(name, status, value, threshold, note)
VettingScorecard(tic_id, candidate_id, checks, overall, n_pass, n_warn, n_fail)
build_scorecard(candidate_row, *, checks_fn) -> VettingScorecard
format_scorecard(scorecard) -> str
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class VettingCheck:
    name: str
    status: str          # "PASS" | "WARN" | "FAIL" | "SKIP"
    value: float | str | None
    threshold: float | str | None
    note: str = ""


@dataclass(frozen=True)
class VettingScorecard:
    tic_id: int
    candidate_id: str
    checks: tuple[VettingCheck, ...]
    overall: str         # "PASS" | "WARN" | "FAIL"
    n_pass: int
    n_warn: int
    n_fail: int


# ---------------------------------------------------------------------------
# Default check suite
# ---------------------------------------------------------------------------

def _check_fpp(row: dict[str, Any]) -> VettingCheck:
    fpp = row.get("best_fpp") or row.get("fpp") or row.get(
        "false_positive_probability"
    )
    if fpp is None:
        return VettingCheck("FPP", "SKIP", None, None, "FPP not available")
    fpp = float(fpp)
    if fpp < 0.10:
        return VettingCheck("FPP", "PASS", round(fpp, 4), 0.10)
    if fpp < 0.50:
        return VettingCheck("FPP", "WARN", round(fpp, 4), 0.10,
                            "FPP above 10%")
    return VettingCheck("FPP", "FAIL", round(fpp, 4), 0.50,
                        "FPP ≥ 50% — likely false positive")


def _check_snr(row: dict[str, Any]) -> VettingCheck:
    snr = row.get("snr") or row.get("detection_snr")
    if snr is None:
        return VettingCheck("SNR", "SKIP", None, None)
    snr = float(snr)
    if snr >= 7.0:
        return VettingCheck("SNR", "PASS", round(snr, 2), 7.0)
    if snr >= 5.0:
        return VettingCheck("SNR", "WARN", round(snr, 2), 7.0, "SNR < 7")
    return VettingCheck("SNR", "FAIL", round(snr, 2), 5.0, "SNR < 5")


def _check_n_transits(row: dict[str, Any]) -> VettingCheck:
    n = row.get("n_transits") or row.get("transit_count")
    if n is None:
        return VettingCheck("N_TRANSITS", "SKIP", None, None)
    n = int(n)
    if n >= 3:
        return VettingCheck("N_TRANSITS", "PASS", n, 3)
    if n == 2:
        return VettingCheck("N_TRANSITS", "WARN", n, 3, "Only 2 transits")
    return VettingCheck("N_TRANSITS", "FAIL", n, 2, "Fewer than 2 transits")


def _check_odd_even(row: dict[str, Any]) -> VettingCheck:
    sig = row.get("odd_even_significance")
    if sig is None:
        return VettingCheck("ODD_EVEN", "SKIP", None, None)
    sig = float(sig)
    if sig < 2.0:
        return VettingCheck("ODD_EVEN", "PASS", round(sig, 2), 2.0)
    if sig < 3.0:
        return VettingCheck("ODD_EVEN", "WARN", round(sig, 2), 2.0,
                            "Mild odd/even asymmetry")
    return VettingCheck("ODD_EVEN", "FAIL", round(sig, 2), 3.0,
                        "Odd/even depth mismatch — possible EB at 2× period")


def _check_secondary(row: dict[str, Any]) -> VettingCheck:
    sec_snr = row.get("secondary_snr")
    if sec_snr is None:
        return VettingCheck("SECONDARY", "SKIP", None, None)
    sec_snr = float(sec_snr)
    if sec_snr < 3.0:
        return VettingCheck("SECONDARY", "PASS", round(sec_snr, 2), 3.0)
    return VettingCheck("SECONDARY", "FAIL", round(sec_snr, 2), 3.0,
                        "Secondary eclipse detected — likely EB")


def _check_centroid(row: dict[str, Any]) -> VettingCheck:
    delta = row.get("centroid_delta_arcsec")
    if delta is None:
        return VettingCheck("CENTROID", "SKIP", None, None)
    delta = float(delta)
    if delta < 1.0:
        return VettingCheck("CENTROID", "PASS", round(delta, 3), 1.0)
    if delta < 2.0:
        return VettingCheck("CENTROID", "WARN", round(delta, 3), 1.0,
                            "Mild centroid shift")
    return VettingCheck("CENTROID", "FAIL", round(delta, 3), 2.0,
                        "Large centroid shift — likely background EB")


def _check_pathway(row: dict[str, Any]) -> VettingCheck:
    pathway = row.get("best_pathway") or row.get("pathway", "")
    if not pathway:
        return VettingCheck("PATHWAY", "SKIP", None, None)
    strong = {"tfop_ready", "kepler_archive_candidate"}
    weak   = {"planet_hunters_discussion"}
    if pathway in strong:
        return VettingCheck("PATHWAY", "PASS", pathway, None)
    if pathway in weak:
        return VettingCheck("PATHWAY", "WARN", pathway, None,
                            "Needs additional data")
    return VettingCheck("PATHWAY", "FAIL", pathway, None,
                        "Not recommended for formal follow-up")


_DEFAULT_CHECKS: list[Callable[[dict], VettingCheck]] = [
    _check_fpp,
    _check_snr,
    _check_n_transits,
    _check_odd_even,
    _check_secondary,
    _check_centroid,
    _check_pathway,
]


def _overall(n_pass: int, n_warn: int, n_fail: int) -> str:
    if n_fail > 0:
        return "FAIL"
    if n_warn > 0:
        return "WARN"
    return "PASS"


def build_scorecard(
    candidate_row: dict[str, Any],
    *,
    checks_fn: list[Callable[[dict], VettingCheck]] | None = None,
) -> VettingScorecard:
    """Build a vetting scorecard from a candidate result dictionary.

    Args:
        candidate_row: Dict with pipeline output keys (fpp, snr, pathway, etc.).
        checks_fn: Override the list of check callables.

    Returns:
        :class:`VettingScorecard`.
    """
    fns = checks_fn if checks_fn is not None else _DEFAULT_CHECKS
    checks = tuple(fn(candidate_row) for fn in fns)

    n_pass = sum(1 for c in checks if c.status == "PASS")
    n_warn = sum(1 for c in checks if c.status == "WARN")
    n_fail = sum(1 for c in checks if c.status == "FAIL")

    tic_id = int(candidate_row.get("tic_id", 0))
    cid = str(candidate_row.get("candidate_id", f"TIC{tic_id}"))

    return VettingScorecard(
        tic_id=tic_id,
        candidate_id=cid,
        checks=checks,
        overall=_overall(n_pass, n_warn, n_fail),
        n_pass=n_pass,
        n_warn=n_warn,
        n_fail=n_fail,
    )


def format_scorecard(scorecard: VettingScorecard) -> str:
    """Format a VettingScorecard as a Markdown report."""
    icon = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌", "SKIP": "—"}
    lines = [
        f"## Vetting Scorecard — {scorecard.candidate_id}",
        "",
        f"**Overall: {scorecard.overall}**  "
        f"({scorecard.n_pass} pass / {scorecard.n_warn} warn / {scorecard.n_fail} fail)",
        "",
        "| Check | Status | Value | Threshold | Note |",
        "|-------|--------|-------|-----------|------|",
    ]
    for c in scorecard.checks:
        val = str(c.value) if c.value is not None else "—"
        thr = str(c.threshold) if c.threshold is not None else "—"
        lines.append(
            f"| {c.name} | {icon.get(c.status, c.status)} {c.status} "
            f"| {val} | {thr} | {c.note} |"
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(
        prog="vetting_scorecard",
        description="Build a pass/fail vetting scorecard for a candidate.",
    )
    parser.add_argument("--candidate", required=True, metavar="JSON",
                        help="JSON file with candidate result keys.")
    args = parser.parse_args(argv)

    row = json.loads(Path(args.candidate).read_text())
    if isinstance(row, list):
        row = row[0]
    sc = build_scorecard(row)
    print(format_scorecard(sc))
    return 0 if sc.overall != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
