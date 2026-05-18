"""Aggregate multiple vetting diagnostics into a single evidence summary.

Combines outputs from various Skills into a weighted evidence score for
planet vs. false positive, producing a human-readable report with pass/fail
badges for each diagnostic.

Public API
----------
EvidenceItem(name, value, verdict, weight, contribution)
EvidenceAggregateResult(tic_id, planet_evidence, fp_evidence, net_score,
                        n_pass, n_fail, n_unknown, classification, items, flag)
aggregate_evidence(tic_id, diagnostics, *, planet_threshold,
                   fp_threshold) -> EvidenceAggregateResult
format_evidence_aggregate(result) -> str
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvidenceItem:
    name: str
    value: float | str | None
    verdict: str      # "PASS", "FAIL", "WARN", "UNKNOWN"
    weight: float
    contribution: float   # weight × (1 if PASS, -1 if FAIL, 0 otherwise)


@dataclass(frozen=True)
class EvidenceAggregateResult:
    tic_id: int
    planet_evidence: float
    fp_evidence: float
    net_score: float           # planet_evidence - fp_evidence, in [-1, 1]
    n_pass: int
    n_fail: int
    n_unknown: int
    classification: str        # "planet_candidate" | "likely_fp" | "ambiguous"
    items: tuple[EvidenceItem, ...]
    flag: str                  # "OK", "INSUFFICIENT"


# Default diagnostic weights: (name, planet_weight)
# Positive weight → PASS supports planet; negative weight → PASS supports FP
_DEFAULT_WEIGHTS: dict[str, float] = {
    "snr": 0.20,
    "odd_even": -0.15,           # PASS (no asymmetry) → supports planet
    "secondary_eclipse": -0.20,  # PASS (no secondary) → supports planet
    "centroid": -0.20,           # PASS (no shift) → supports planet
    "depth_consistency": 0.10,
    "period_doubling": -0.15,    # PASS (no doubling) → supports planet
    "stellar_density": 0.10,
    "phase_coverage": 0.05,
    "harmonic": -0.10,           # PASS (no harmonic) → supports planet
    "eb_classifier": -0.20,      # PASS (planet_candidate) → supports planet
}


def aggregate_evidence(
    tic_id: int,
    diagnostics: dict[str, tuple[str, float | str | None]],
    *,
    planet_threshold: float = 0.25,
    fp_threshold: float = -0.25,
) -> EvidenceAggregateResult:
    """Aggregate diagnostic verdicts into a planet/FP evidence score.

    Args:
        tic_id: TESS Input Catalog ID.
        diagnostics: Mapping of diagnostic_name → (verdict, value).
            verdict is "PASS", "FAIL", "WARN", or "UNKNOWN".
            value is the raw diagnostic value (for display).
        planet_threshold: Net score above which → "planet_candidate".
        fp_threshold: Net score below which → "likely_fp".

    Returns:
        :class:`EvidenceAggregateResult`.
    """
    if not diagnostics:
        return EvidenceAggregateResult(
            tic_id, 0.0, 0.0, 0.0, 0, 0, 0,
            "ambiguous", (), "INSUFFICIENT",
        )

    items: list[EvidenceItem] = []
    planet_evidence = 0.0
    fp_evidence = 0.0
    n_pass = n_fail = n_unknown = 0

    for name, (verdict, value) in diagnostics.items():
        weight = abs(_DEFAULT_WEIGHTS.get(name, 0.05))
        if verdict == "PASS":
            contribution = weight
            planet_evidence += weight
            n_pass += 1
        elif verdict == "FAIL":
            contribution = -weight
            fp_evidence += weight
            n_fail += 1
        else:
            contribution = 0.0
            n_unknown += 1

        items.append(EvidenceItem(
            name=name,
            value=value,
            verdict=verdict,
            weight=abs(weight),
            contribution=round(contribution, 4),
        ))

    total_weight = sum(abs(_DEFAULT_WEIGHTS.get(n, 0.05)) for n in diagnostics)
    net_score = (planet_evidence - fp_evidence) / total_weight if total_weight > 0 else 0.0
    net_score = max(-1.0, min(1.0, net_score))

    if net_score >= planet_threshold:
        classification = "planet_candidate"
    elif net_score <= fp_threshold:
        classification = "likely_fp"
    else:
        classification = "ambiguous"

    return EvidenceAggregateResult(
        tic_id=tic_id,
        planet_evidence=round(planet_evidence, 4),
        fp_evidence=round(fp_evidence, 4),
        net_score=round(net_score, 4),
        n_pass=n_pass,
        n_fail=n_fail,
        n_unknown=n_unknown,
        classification=classification,
        items=tuple(items),
        flag="OK",
    )


def format_evidence_aggregate(result: EvidenceAggregateResult) -> str:
    """Format evidence aggregate as Markdown."""
    lines = [
        "## Candidate Evidence Summary",
        "",
        f"- TIC ID: {result.tic_id}",
        f"- Planet evidence: {result.planet_evidence:.4f}",
        f"- FP evidence: {result.fp_evidence:.4f}",
        f"- Net score: {result.net_score:+.4f}",
        f"- Pass: {result.n_pass} | Fail: {result.n_fail} | Unknown: {result.n_unknown}",
        f"- Classification: **{result.classification}**",
        f"- **Flag: {result.flag}**",
        "",
        "### Diagnostics",
        "",
        "| Diagnostic | Verdict | Value | Contribution |",
        "|---|---|---|---|",
    ]
    for item in result.items:
        val_str = str(item.value) if item.value is not None else "—"
        lines.append(
            f"| {item.name} | {item.verdict} | {val_str} | {item.contribution:+.4f} |"
        )
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="candidate_evidence_aggregator",
        description="Aggregate vetting diagnostics into an evidence summary.",
    )
    parser.add_argument("tic_id", type=int)
    parser.add_argument("--planet-threshold", type=float, default=0.25)
    parser.add_argument("--fp-threshold", type=float, default=-0.25)
    args = parser.parse_args(argv)

    result = aggregate_evidence(
        args.tic_id, {},
        planet_threshold=args.planet_threshold,
        fp_threshold=args.fp_threshold,
    )
    print(format_evidence_aggregate(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
