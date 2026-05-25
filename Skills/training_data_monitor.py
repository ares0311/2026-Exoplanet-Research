"""Compute statistics and gate checks on a CNN training dataset.

Summarises label balance, source breakdown, and signal quality (SNR/depth
percentiles) for a list of snippet dicts.  The gate opens when there are
enough total labels and the FP-to-planet ratio is within limits.

Public API
----------
DataMonitorResult(n_total, n_positive, n_negative, balance_ratio,
                  source_breakdown, snr_p10, snr_p50, snr_p90,
                  depth_p10, depth_p50, depth_p90,
                  gate_open, gate_reason, flag)
monitor_training_data(snippets, *, label_threshold, max_balance_ratio) -> DataMonitorResult
monitor_from_path(dataset_path, *, label_threshold, max_balance_ratio) -> DataMonitorResult
format_monitor_result(result) -> str
"""
from __future__ import annotations

import json
import math
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DataMonitorResult:
    """Training-data statistics and CNN gate check."""

    n_total: int
    n_positive: int
    n_negative: int
    balance_ratio: float | None     # n_negative / n_positive; None if n_positive == 0
    source_breakdown: dict          # {source_name: count}
    snr_p10: float | None
    snr_p50: float | None
    snr_p90: float | None
    depth_p10: float | None
    depth_p50: float | None
    depth_p90: float | None
    gate_open: bool
    gate_reason: str   # "OK" or human-readable explanation of why gate is closed
    flag: str  # "OK" | "INSUFFICIENT" | "INVALID"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _percentile(values: list[float], p: float) -> float:
    """Compute the *p*-th percentile (0–100) of a sorted or unsorted list."""
    s = sorted(values)
    n = len(s)
    if n == 0:
        return float("nan")
    idx = p / 100.0 * (n - 1)
    lo = int(math.floor(idx))
    hi = min(lo + 1, n - 1)
    frac = idx - lo
    return s[lo] * (1.0 - frac) + s[hi] * frac


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def monitor_training_data(
    snippets: list[dict],
    *,
    label_threshold: int = 5000,
    max_balance_ratio: float = 3.0,
) -> DataMonitorResult:
    """Compute statistics and gate status for a list of snippet dicts.

    Args:
        snippets:         List of dicts with at least ``label`` (int) and
                          ``source`` (str) keys.  ``snr`` and ``depth_ppm``
                          are optional floats.
        label_threshold:  Minimum total snippets for gate to open.
        max_balance_ratio: Maximum n_negative / n_positive ratio.

    Returns:
        :class:`DataMonitorResult`
    """
    if not isinstance(snippets, list):
        return DataMonitorResult(
            n_total=0, n_positive=0, n_negative=0, balance_ratio=None,
            source_breakdown={}, snr_p10=None, snr_p50=None, snr_p90=None,
            depth_p10=None, depth_p50=None, depth_p90=None,
            gate_open=False, gate_reason="invalid input", flag="INVALID",
        )

    n_positive = 0
    n_negative = 0
    source_breakdown: dict[str, int] = {}
    snr_vals: list[float] = []
    depth_vals: list[float] = []

    for s in snippets:
        lbl = s.get("label")
        if lbl == 1:
            n_positive += 1
        elif lbl == 0:
            n_negative += 1

        src = str(s.get("source", "unknown"))
        source_breakdown[src] = source_breakdown.get(src, 0) + 1

        snr = s.get("snr")
        if snr is not None:
            with suppress(ValueError, TypeError):
                snr_vals.append(float(snr))

        depth = s.get("depth_ppm")
        if depth is not None:
            with suppress(ValueError, TypeError):
                depth_vals.append(float(depth))

    n_total = n_positive + n_negative
    balance_ratio = (n_negative / n_positive) if n_positive > 0 else None

    # Percentile stats
    def _p(vals: list[float], p: float) -> float | None:
        return _percentile(vals, p) if vals else None

    snr_p10 = _p(snr_vals, 10)
    snr_p50 = _p(snr_vals, 50)
    snr_p90 = _p(snr_vals, 90)
    depth_p10 = _p(depth_vals, 10)
    depth_p50 = _p(depth_vals, 50)
    depth_p90 = _p(depth_vals, 90)

    # Gate check
    gate_open = True
    gate_reasons: list[str] = []

    if n_total < label_threshold:
        gate_open = False
        gate_reasons.append(f"need {label_threshold} labels, have {n_total}")

    if balance_ratio is not None and balance_ratio > max_balance_ratio:
        gate_open = False
        gate_reasons.append(
            f"FP/planet ratio {balance_ratio:.2f} exceeds max {max_balance_ratio}"
        )
    elif balance_ratio is None:
        gate_open = False
        gate_reasons.append("no positive (planet) labels")

    gate_reason = "OK" if gate_open else "; ".join(gate_reasons)

    flag = "INSUFFICIENT" if n_total == 0 or not gate_open else "OK"

    return DataMonitorResult(
        n_total=n_total,
        n_positive=n_positive,
        n_negative=n_negative,
        balance_ratio=balance_ratio,
        source_breakdown=source_breakdown,
        snr_p10=snr_p10,
        snr_p50=snr_p50,
        snr_p90=snr_p90,
        depth_p10=depth_p10,
        depth_p50=depth_p50,
        depth_p90=depth_p90,
        gate_open=gate_open,
        gate_reason=gate_reason,
        flag=flag,
    )


def monitor_from_path(
    dataset_path: Path,
    *,
    label_threshold: int = 5000,
    max_balance_ratio: float = 3.0,
) -> DataMonitorResult:
    """Load a snippet JSON file and run :func:`monitor_training_data`.

    Args:
        dataset_path:     Path to JSON file (list of dicts or dict with
                          ``"snippets"`` key).
        label_threshold:  Forwarded to :func:`monitor_training_data`.
        max_balance_ratio: Forwarded to :func:`monitor_training_data`.

    Returns:
        :class:`DataMonitorResult`
    """
    try:
        raw = json.loads(Path(dataset_path).read_text())
    except Exception:  # noqa: BLE001
        return DataMonitorResult(
            n_total=0, n_positive=0, n_negative=0, balance_ratio=None,
            source_breakdown={}, snr_p10=None, snr_p50=None, snr_p90=None,
            depth_p10=None, depth_p50=None, depth_p90=None,
            gate_open=False, gate_reason="could not read file", flag="INVALID",
        )

    if isinstance(raw, list):
        snippets = raw
    elif isinstance(raw, dict):
        snippets = raw.get("snippets", [])
    else:
        snippets = []

    return monitor_training_data(
        snippets,
        label_threshold=label_threshold,
        max_balance_ratio=max_balance_ratio,
    )


def format_monitor_result(result: DataMonitorResult) -> str:
    """Return a Markdown summary of a :class:`DataMonitorResult`."""
    def _fmt(v: float | None) -> str:
        return f"{v:.2f}" if v is not None else "N/A"

    lines = [
        "## Training Data Monitor",
        "",
        f"**Flag**: {result.flag}",
        f"**Gate open**: {'YES' if result.gate_open else 'NO'}",
        f"**Gate reason**: {result.gate_reason}",
        "",
        "### Label Counts",
        f"- Total: {result.n_total}",
        f"- Positive (planet): {result.n_positive}",
        f"- Negative (FP): {result.n_negative}",
        f"- Balance ratio (FP/planet): {_fmt(result.balance_ratio)}",
        "",
        "### Source Breakdown",
    ]
    for src, count in sorted(result.source_breakdown.items()):
        lines.append(f"- {src}: {count}")
    lines += [
        "",
        "### SNR Percentiles",
        f"- P10: {_fmt(result.snr_p10)}",
        f"- P50: {_fmt(result.snr_p50)}",
        f"- P90: {_fmt(result.snr_p90)}",
        "",
        "### Depth Percentiles (ppm)",
        f"- P10: {_fmt(result.depth_p10)}",
        f"- P50: {_fmt(result.depth_p50)}",
        f"- P90: {_fmt(result.depth_p90)}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="training_data_monitor",
        description="Monitor CNN training dataset statistics and gate status.",
    )
    parser.add_argument("dataset", help="Path to snippet JSON file.")
    parser.add_argument("--label-threshold", type=int, default=5000)
    parser.add_argument("--max-balance-ratio", type=float, default=3.0)
    args = parser.parse_args(argv)

    result = monitor_from_path(
        Path(args.dataset),
        label_threshold=args.label_threshold,
        max_balance_ratio=args.max_balance_ratio,
    )
    print(format_monitor_result(result))
    return 0 if result.gate_open else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
