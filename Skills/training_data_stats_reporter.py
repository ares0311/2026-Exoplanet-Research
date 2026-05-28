"""Report statistics on the assembled CNN training corpus.

Summarizes class balance, period distribution, depth distribution, and
source composition. Helps identify gaps before committing to a training run.

Public API
----------
TrainingDataStats(n_total, n_positive, n_negative, period_min, period_max,
                  period_median, depth_ppm_min, depth_ppm_max,
                  depth_ppm_median, source_counts, flag)
compute_training_stats(rows, *, pos_label) -> TrainingDataStats
format_training_stats(stats) -> str
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingDataStats:
    n_total: int
    n_positive: int
    n_negative: int
    period_min: float | None
    period_max: float | None
    period_median: float | None
    depth_ppm_min: float | None
    depth_ppm_max: float | None
    depth_ppm_median: float | None
    source_counts: dict[str, int]
    flag: str  # "OK" | "EMPTY" | "INVALID"


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def compute_training_stats(
    rows: list[dict],
    *,
    pos_label: str = "planet_candidate",
) -> TrainingDataStats:
    """Compute descriptive statistics for a training label corpus.

    Args:
        rows: List of label dicts with at minimum ``label`` key; optionally
            ``period_days``, ``depth_ppm``, ``source``.
        pos_label: Label value for positive class.

    Returns:
        TrainingDataStats with distribution summaries.
    """
    if not isinstance(rows, list):
        return TrainingDataStats(
            n_total=0, n_positive=0, n_negative=0,
            period_min=None, period_max=None, period_median=None,
            depth_ppm_min=None, depth_ppm_max=None, depth_ppm_median=None,
            source_counts={}, flag="INVALID",
        )
    if not rows:
        return TrainingDataStats(
            n_total=0, n_positive=0, n_negative=0,
            period_min=None, period_max=None, period_median=None,
            depth_ppm_min=None, depth_ppm_max=None, depth_ppm_median=None,
            source_counts={}, flag="EMPTY",
        )

    n_pos = sum(1 for r in rows if r.get("label") == pos_label)
    n_neg = len(rows) - n_pos

    import contextlib

    periods: list[float] = []
    depths: list[float] = []
    source_counts: dict[str, int] = {}

    for r in rows:
        p = r.get("period_days")
        if p is not None:
            with contextlib.suppress(TypeError, ValueError):
                periods.append(float(p))
        d = r.get("depth_ppm")
        if d is not None:
            with contextlib.suppress(TypeError, ValueError):
                depths.append(float(d))
        src = str(r.get("source") or "unknown")
        source_counts[src] = source_counts.get(src, 0) + 1

    return TrainingDataStats(
        n_total=len(rows),
        n_positive=n_pos,
        n_negative=n_neg,
        period_min=min(periods) if periods else None,
        period_max=max(periods) if periods else None,
        period_median=_median(periods),
        depth_ppm_min=min(depths) if depths else None,
        depth_ppm_max=max(depths) if depths else None,
        depth_ppm_median=_median(depths),
        source_counts=source_counts,
        flag="OK",
    )


def format_training_stats(stats: TrainingDataStats) -> str:
    """Format a Markdown training data statistics report.

    Args:
        stats: TrainingDataStats to format.

    Returns:
        Markdown string.
    """
    lines = [
        "## Training Data Statistics\n",
        f"Flag: `{stats.flag}` | Total: {stats.n_total}\n",
    ]
    if stats.flag in ("EMPTY", "INVALID"):
        lines.append(f"\n_{stats.flag}: no training data._\n")
        return "\n".join(lines)

    lines += [
        "",
        "### Class Balance",
        "",
        "| Class | Count | Fraction |",
        "|---|---|---|",
    ]
    frac_pos = stats.n_positive / stats.n_total if stats.n_total else 0
    frac_neg = stats.n_negative / stats.n_total if stats.n_total else 0
    lines.append(f"| Positive | {stats.n_positive} | {frac_pos:.1%} |")
    lines.append(f"| Negative | {stats.n_negative} | {frac_neg:.1%} |")

    def _fmt(v: float | None) -> str:
        return f"{v:.3f}" if v is not None else "—"

    lines += [
        "",
        "### Period Distribution (days)",
        "",
        f"Min: {_fmt(stats.period_min)} | Median: {_fmt(stats.period_median)}"
        f" | Max: {_fmt(stats.period_max)}",
        "",
        "### Depth Distribution (ppm)",
        "",
        f"Min: {_fmt(stats.depth_ppm_min)} | Median: {_fmt(stats.depth_ppm_median)}"
        f" | Max: {_fmt(stats.depth_ppm_max)}",
        "",
        "### Source Breakdown",
        "",
        "| Source | Count |",
        "|---|---|",
    ]
    for src, cnt in sorted(stats.source_counts.items(), key=lambda t: -t[1]):
        lines.append(f"| {src} | {cnt} |")

    return "\n".join(lines)


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Report training corpus statistics.")
    parser.add_argument("label_json", help="Path to JSON label rows file.")
    parser.add_argument("--pos-label", default="planet_candidate")
    args = parser.parse_args(argv)

    rows = json.loads(Path(args.label_json).read_text())
    stats = compute_training_stats(rows, pos_label=args.pos_label)
    print(format_training_stats(stats))
    return 0 if stats.flag == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
