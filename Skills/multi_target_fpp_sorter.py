"""Sort multiple candidates by FPP into labelled quality tiers."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

# Tier thresholds
_TIERS: list[tuple[str, float, str]] = [
    ("A", 0.05,  "High-priority follow-up"),
    ("B", 0.15,  "Moderate-priority follow-up"),
    ("C", 0.50,  "Low-priority, significant FP risk"),
    ("D", 1.01,  "Likely false positive"),
]


@dataclass(frozen=True)
class FppTierEntry:
    tic_id: str
    fpp: float
    tier: str
    tier_label: str


@dataclass(frozen=True)
class FppSortResult:
    n_total: int
    n_tier_a: int
    n_tier_b: int
    n_tier_c: int
    n_tier_d: int
    entries: tuple[FppTierEntry, ...]
    flag: str


def _get_fpp(candidate: dict) -> float | None:
    for key in ("false_positive_probability", "fpp", "best_fpp"):
        val = candidate.get(key)
        if val is not None:
            return float(val)
    scores = candidate.get("scores", {})
    val = scores.get("false_positive_probability")
    if val is not None:
        return float(val)
    return None


def _assign_tier(fpp: float) -> tuple[str, str]:
    for tier, threshold, label in _TIERS:
        if fpp < threshold:
            return tier, label
    return "D", "Likely false positive"


def sort_by_fpp(candidates: list[dict]) -> FppSortResult:
    """
    Sort candidates by FPP and assign tier labels A–D.

    A: FPP < 0.05 (high priority)
    B: 0.05 ≤ FPP < 0.15 (moderate priority)
    C: 0.15 ≤ FPP < 0.50 (low priority)
    D: FPP ≥ 0.50 (likely FP)
    """
    if not candidates:
        return FppSortResult(
            n_total=0, n_tier_a=0, n_tier_b=0, n_tier_c=0, n_tier_d=0,
            entries=(), flag="NO_CANDIDATES",
        )

    entries: list[FppTierEntry] = []
    for cand in candidates:
        tic_id = str(cand.get("tic_id", "unknown"))
        fpp = _get_fpp(cand)
        if fpp is None:
            fpp = 1.0
        tier, label = _assign_tier(fpp)
        entries.append(FppTierEntry(tic_id=tic_id, fpp=fpp, tier=tier, tier_label=label))

    entries.sort(key=lambda e: e.fpp)
    counts = {t: sum(1 for e in entries if e.tier == t) for t in "ABCD"}

    return FppSortResult(
        n_total=len(entries),
        n_tier_a=counts["A"],
        n_tier_b=counts["B"],
        n_tier_c=counts["C"],
        n_tier_d=counts["D"],
        entries=tuple(entries),
        flag="OK",
    )


def format_fpp_tier_table(r: FppSortResult) -> str:
    if r.flag != "OK":
        return f"No candidates (flag: {r.flag}).\n"
    lines = [
        f"**FPP Tier Summary** — {r.n_total} candidates: "
        f"A={r.n_tier_a} B={r.n_tier_b} C={r.n_tier_c} D={r.n_tier_d}\n",
        "| TIC ID | FPP | Tier | Label |",
        "|---|---|---|---|",
    ]
    for e in r.entries:
        lines.append(f"| {e.tic_id} | {e.fpp:.4f} | {e.tier} | {e.tier_label} |")
    return "\n".join(lines)


def _cli() -> int:
    p = argparse.ArgumentParser(description="Sort candidates by FPP into tiers.")
    p.add_argument("candidates_json", help="JSON array string or @file")
    args = p.parse_args()
    raw = args.candidates_json
    if raw.startswith("@"):
        with open(raw[1:]) as f:
            candidates = json.load(f)
    else:
        candidates = json.loads(raw)
    r = sort_by_fpp(candidates)
    print(format_fpp_tier_table(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
