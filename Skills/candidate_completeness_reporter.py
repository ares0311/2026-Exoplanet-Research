"""Aggregate per-target completeness metrics into a summary report.

Public API:
    CompletenessReport  -- frozen dataclass
    build_completeness_report(candidates) -> CompletenessReport
    format_completeness_report(result) -> str
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class CompletenessReport:
    n_candidates: int
    n_fully_vetted: int
    n_missing_snr: int
    n_missing_centroid: int
    n_missing_stellar: int
    mean_fpp: float
    flag: str


def build_completeness_report(candidates: list[dict[str, object]]) -> CompletenessReport:
    if not candidates:
        return CompletenessReport(
            n_candidates=0, n_fully_vetted=0, n_missing_snr=0,
            n_missing_centroid=0, n_missing_stellar=0, mean_fpp=0.0,
            flag="NO_CANDIDATES",
        )
    n = len(candidates)
    n_fully_vetted = 0
    n_missing_snr = 0
    n_missing_centroid = 0
    n_missing_stellar = 0
    fpp_sum = 0.0
    fpp_count = 0
    for c in candidates:
        has_snr = "snr" in c and c["snr"] is not None
        has_centroid = "centroid_offset_arcsec" in c and c["centroid_offset_arcsec"] is not None
        has_stellar = "stellar_radius_rsun" in c and c["stellar_radius_rsun"] is not None
        if not has_snr:
            n_missing_snr += 1
        if not has_centroid:
            n_missing_centroid += 1
        if not has_stellar:
            n_missing_stellar += 1
        if has_snr and has_centroid and has_stellar:
            n_fully_vetted += 1
        fpp = c.get("fpp")
        if isinstance(fpp, (int, float)):
            fpp_sum += float(fpp)
            fpp_count += 1
    mean_fpp = fpp_sum / fpp_count if fpp_count > 0 else 0.0
    flag = "COMPLETE" if n_fully_vetted == n else "PARTIAL"
    return CompletenessReport(
        n_candidates=n,
        n_fully_vetted=n_fully_vetted,
        n_missing_snr=n_missing_snr,
        n_missing_centroid=n_missing_centroid,
        n_missing_stellar=n_missing_stellar,
        mean_fpp=mean_fpp,
        flag=flag,
    )


def format_completeness_report(result: CompletenessReport) -> str:
    lines = [
        "## Candidate Completeness Report",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| N Candidates | {result.n_candidates} |",
        f"| Fully Vetted | {result.n_fully_vetted} |",
        f"| Missing SNR | {result.n_missing_snr} |",
        f"| Missing Centroid | {result.n_missing_centroid} |",
        f"| Missing Stellar Params | {result.n_missing_stellar} |",
        f"| Mean FPP | {result.mean_fpp:.4f} |",
        f"| Flag | {result.flag} |",
    ]
    return "\n".join(lines)


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Build candidate completeness report.")
    parser.add_argument("candidates_file", help="JSON file with list of candidate dicts.")
    args = parser.parse_args()
    with open(args.candidates_file) as fh:
        candidates = json.load(fh)
    result = build_completeness_report(candidates)
    print(format_completeness_report(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
