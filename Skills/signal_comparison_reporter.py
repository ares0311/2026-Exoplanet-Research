"""Side-by-side Markdown comparison of multiple transit candidate signals.

Accepts a list of candidate signal dicts and renders a comparison table
plus a qualitative summary for each signal pair.

Public API
----------
SignalComparisonResult(n_signals, headers, rows, summary_lines, flag)
compare_signals(signals, *, title) -> SignalComparisonResult
format_signal_comparison(result) -> str
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SignalComparisonResult:
    n_signals: int
    headers: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    summary_lines: tuple[str, ...]
    flag: str  # "OK" | "SINGLE" | "EMPTY" | "INVALID"


_DISPLAY_COLS = [
    ("tic_id", "TIC ID"),
    ("period_days", "Period (d)"),
    ("epoch_bjd", "Epoch (BJD)"),
    ("depth_ppm", "Depth (ppm)"),
    ("duration_hours", "Duration (h)"),
    ("snr", "SNR"),
    ("false_positive_probability", "FPP"),
    ("pathway", "Pathway"),
]


def _fmt(val: object) -> str:
    if val is None:
        return "—"
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)


def compare_signals(
    signals: list[dict],
    *,
    title: str = "Signal Comparison",
) -> SignalComparisonResult:
    """Build a side-by-side comparison table for candidate signals.

    Args:
        signals: List of candidate signal dicts (from ``run_pipeline`` output).
        title: Report title (unused in output but available to caller).

    Returns:
        :class:`SignalComparisonResult`.
    """
    if not signals:
        return SignalComparisonResult(0, (), (), (), "EMPTY")
    if not isinstance(signals, list):
        return SignalComparisonResult(0, (), (), (), "INVALID")

    if len(signals) == 1:
        return SignalComparisonResult(
            1, (), (), ("Only one signal — nothing to compare.",), "SINGLE"
        )

    # Determine which columns have any data
    active_cols = []
    for key, label in _DISPLAY_COLS:
        if any(s.get(key) is not None for s in signals):
            active_cols.append((key, label))

    headers = ("Signal #",) + tuple(label for _, label in active_cols)
    rows = []
    for i, sig in enumerate(signals):
        row = (str(i + 1),) + tuple(_fmt(sig.get(key)) for key, _ in active_cols)
        rows.append(tuple(row))

    # Summary: flag pairs with matching periods
    summary: list[str] = []
    for i in range(len(signals)):
        for j in range(i + 1, len(signals)):
            pa = signals[i].get("period_days")
            pb = signals[j].get("period_days")
            if pa and pb and pa > 0 and pb > 0:
                ratio = max(pa, pb) / min(pa, pb)
                nearest = round(ratio)
                dev = abs(ratio - nearest) / max(nearest, 1)
                if dev < 0.02:
                    if nearest == 1:
                        summary.append(
                            f"Signals {i+1} and {j+1} have nearly identical periods "
                            f"({pa:.4f} vs {pb:.4f} d) — possible duplicate."
                        )
                    else:
                        summary.append(
                            f"Signals {i+1} and {j+1} periods are in {nearest}:1 ratio "
                            f"— possible alias."
                        )

    return SignalComparisonResult(
        n_signals=len(signals),
        headers=headers,
        rows=tuple(rows),
        summary_lines=tuple(summary),
        flag="OK",
    )


def format_signal_comparison(result: SignalComparisonResult) -> str:
    """Format signal comparison as Markdown."""
    if result.flag == "EMPTY":
        return "## Signal Comparison\n\n_No signals provided._\n"
    if result.flag == "SINGLE":
        return f"## Signal Comparison\n\n{result.summary_lines[0]}\n"

    lines = ["## Signal Comparison", ""]
    if result.headers:
        lines.append("| " + " | ".join(result.headers) + " |")
        lines.append("|" + "|".join("---" for _ in result.headers) + "|")
        for row in result.rows:
            lines.append("| " + " | ".join(row) + " |")

    if result.summary_lines:
        lines.append("")
        lines.append("**Notes:**")
        for s in result.summary_lines:
            lines.append(f"- {s}")

    lines.append(f"\n_Signals compared: {result.n_signals}_")
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="signal_comparison_reporter",
        description="Compare multiple transit candidate signals.",
    )
    parser.parse_args(argv)

    result = compare_signals([])
    print(format_signal_comparison(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
