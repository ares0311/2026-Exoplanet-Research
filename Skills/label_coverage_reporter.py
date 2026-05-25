"""Report label coverage statistics for a training dataset.

Checks how many labelled examples are available, broken down by source,
period bin, and depth bin. Reports whether the CNN gate threshold is met.

Public API
----------
LabelCoverageResult(n_total, n_positive, n_negative, by_source, by_period_bin,
                    by_depth_bin, gate_open, gate_threshold, flag)
report_label_coverage(records, *, gate_threshold) -> LabelCoverageResult
format_label_coverage(result) -> str
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LabelCoverageResult:
    n_total: int
    n_positive: int
    n_negative: int
    by_source: dict         # {source_str: {"positive": n, "negative": n}}
    by_period_bin: dict     # {"<1d": n, "1-10d": n, "10-100d": n, ">100d": n, "unknown": n}
    by_depth_bin: dict      # {"<500ppm": n, "500-2000ppm": n, "2000-10000ppm": n,
    #                           ">10000ppm": n, "unknown": n}
    gate_open: bool         # n_total >= gate_threshold
    gate_threshold: int
    flag: str               # "OK" | "BELOW_GATE" | "EMPTY"


def _period_bin(period_days: float | None) -> str:
    if period_days is None:
        return "unknown"
    if period_days < 1.0:
        return "<1d"
    if period_days < 10.0:
        return "1-10d"
    if period_days < 100.0:
        return "10-100d"
    return ">100d"


def _depth_bin(depth_ppm: float | None) -> str:
    if depth_ppm is None:
        return "unknown"
    if depth_ppm < 500.0:
        return "<500ppm"
    if depth_ppm < 2000.0:
        return "500-2000ppm"
    if depth_ppm < 10000.0:
        return "2000-10000ppm"
    return ">10000ppm"


def report_label_coverage(
    records: list[dict],
    *,
    gate_threshold: int = 5000,
) -> LabelCoverageResult:
    """Summarise label coverage of a training dataset.

    Args:
        records: List of dicts; each must have "label" key (int 0 or 1).
            Optional keys: "source" (str), "period_days" (float),
            "depth_ppm" (float).
        gate_threshold: Minimum total examples to open the CNN training gate.

    Returns:
        :class:`LabelCoverageResult`.
    """
    period_bins = {"<1d": 0, "1-10d": 0, "10-100d": 0, ">100d": 0, "unknown": 0}
    depth_bins = {
        "<500ppm": 0,
        "500-2000ppm": 0,
        "2000-10000ppm": 0,
        ">10000ppm": 0,
        "unknown": 0,
    }
    by_source: dict[str, dict[str, int]] = {}
    n_positive = 0
    n_negative = 0

    for rec in records:
        label = rec.get("label")
        source = str(rec.get("source", "unknown"))
        period = rec.get("period_days")
        depth = rec.get("depth_ppm")

        if source not in by_source:
            by_source[source] = {"positive": 0, "negative": 0}

        if label == 1:
            n_positive += 1
            by_source[source]["positive"] += 1
        elif label == 0:
            n_negative += 1
            by_source[source]["negative"] += 1

        period_bins[_period_bin(period)] += 1
        depth_bins[_depth_bin(depth)] += 1

    n_total = len(records)
    gate_open = n_total >= gate_threshold

    if n_total == 0:
        flag = "EMPTY"
    elif gate_open:
        flag = "OK"
    else:
        flag = "BELOW_GATE"

    return LabelCoverageResult(
        n_total=n_total,
        n_positive=n_positive,
        n_negative=n_negative,
        by_source=by_source,
        by_period_bin=period_bins,
        by_depth_bin=depth_bins,
        gate_open=gate_open,
        gate_threshold=gate_threshold,
        flag=flag,
    )


def format_label_coverage(result: LabelCoverageResult) -> str:
    """Format label coverage result as Markdown."""
    gate_status = "OPEN" if result.gate_open else "CLOSED"
    lines = [
        "## Label Coverage Report",
        "",
        f"- **Total records:** {result.n_total}",
        f"- **Positive (planet):** {result.n_positive}",
        f"- **Negative (FP):** {result.n_negative}",
        f"- **gate threshold:** {result.gate_threshold} — {gate_status}",
        f"- **Flag:** {result.flag}",
        "",
        "### By Source",
    ]
    for src, counts in result.by_source.items():
        lines.append(f"  - {src}: +{counts['positive']} / -{counts['negative']}")

    lines += ["", "### By Period Bin"]
    for bin_name, count in result.by_period_bin.items():
        lines.append(f"  - {bin_name}: {count}")

    lines += ["", "### By Depth Bin"]
    for bin_name, count in result.by_depth_bin.items():
        lines.append(f"  - {bin_name}: {count}")

    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(
        prog="label_coverage_reporter",
        description="Report label coverage for a training dataset JSON file.",
    )
    parser.add_argument("input", help="JSON file with list of label records.")
    parser.add_argument(
        "--gate-threshold",
        type=int,
        default=5000,
        help="Minimum records to open CNN gate (default: 5000).",
    )
    args = parser.parse_args(argv)

    with open(args.input) as fh:  # noqa: PTH123
        records = json.load(fh)

    result = report_label_coverage(records, gate_threshold=args.gate_threshold)
    print(format_label_coverage(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
