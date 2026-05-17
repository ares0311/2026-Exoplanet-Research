"""Export a combined pipeline run summary: ranked candidates + alert readiness.

Loads batch_scan or star_scanner JSON output, ranks candidates, flags those
ready for alerts, and writes a single Markdown + JSON bundle.

Public API
----------
RunSummary(n_total, n_candidates, n_alert_ready, candidates, generated_at)
build_run_summary(paths, *, fpp_threshold, alert_fpp_threshold, top_n) -> RunSummary
write_run_summary(summary, output_dir) -> dict[str, Path]
format_run_summary(summary) -> str
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class RunSummary:
    n_total: int
    n_candidates: int
    n_alert_ready: int
    candidates: tuple[dict, ...]
    generated_at: str


def _fpp(row: dict) -> float:
    for key in ("best_fpp", "false_positive_probability"):
        val = row.get(key)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                pass
    scores = row.get("scores") or {}
    val = scores.get("false_positive_probability")
    if val is not None:
        try:
            return float(val)
        except (TypeError, ValueError):
            pass
    return 1.0


def _rank_score(row: dict) -> float:
    val = row.get("rank_score")
    if val is not None:
        try:
            return float(val)
        except (TypeError, ValueError):
            pass
    return 1.0 - _fpp(row)


def _load_rows(paths: list[Path | str]) -> list[dict]:
    rows: list[dict] = []
    for p in paths:
        raw = json.loads(Path(p).read_text())
        if isinstance(raw, list):
            rows.extend(r for r in raw if isinstance(r, dict))
        elif isinstance(raw, dict):
            if "results" in raw:
                rows.extend(r for r in raw["results"] if isinstance(r, dict))
            elif "entries" in raw:
                rows.extend(r for r in raw["entries"].values() if isinstance(r, dict))
            else:
                rows.append(raw)
    return rows


def build_run_summary(
    paths: list[Path | str],
    *,
    fpp_threshold: float = 0.5,
    alert_fpp_threshold: float = 0.1,
    top_n: int = 20,
) -> RunSummary:
    """Build a run summary from pipeline output files.

    Args:
        paths: JSON result file paths.
        fpp_threshold: FPP below which a target counts as a candidate.
        alert_fpp_threshold: FPP below which a candidate is alert-ready.
        top_n: Maximum number of candidates to include in the summary.

    Returns:
        :class:`RunSummary`.
    """
    all_rows = _load_rows(paths)
    n_total = len(all_rows)

    candidates = [r for r in all_rows if _fpp(r) < fpp_threshold]
    candidates.sort(key=_rank_score, reverse=True)
    candidates = candidates[:top_n]

    n_alert = sum(1 for r in candidates if _fpp(r) < alert_fpp_threshold)

    return RunSummary(
        n_total=n_total,
        n_candidates=len(candidates),
        n_alert_ready=n_alert,
        candidates=tuple(candidates),
        generated_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


def write_run_summary(
    summary: RunSummary, output_dir: Path | str
) -> dict[str, Path]:
    """Write Markdown + JSON files to output_dir."""
    d = Path(output_dir)
    d.mkdir(parents=True, exist_ok=True)

    md_path = d / "run_summary.md"
    md_path.write_text(format_run_summary(summary))

    json_path = d / "run_summary.json"
    json_path.write_text(json.dumps({
        "n_total": summary.n_total,
        "n_candidates": summary.n_candidates,
        "n_alert_ready": summary.n_alert_ready,
        "generated_at": summary.generated_at,
        "candidates": list(summary.candidates),
    }, indent=2))

    return {"markdown": md_path, "json": json_path}


def format_run_summary(summary: RunSummary) -> str:
    """Format run summary as Markdown."""
    lines = [
        "# Pipeline Run Summary",
        "",
        f"_Generated: {summary.generated_at}_",
        "",
        f"- Total targets scanned: {summary.n_total}",
        f"- Candidates (FPP < 0.5): {summary.n_candidates}",
        f"- Alert-ready (FPP < 0.1): {summary.n_alert_ready}",
    ]
    if summary.candidates:
        lines += ["", "## Top Candidates", "",
                  "| TIC ID | FPP | Rank Score | Pathway |",
                  "|---|---|---|---|"]
        for r in summary.candidates:
            tid = r.get("tic_id", "?")
            fpp = _fpp(r)
            rs = _rank_score(r)
            pathway = r.get("best_pathway") or r.get("pathway") or "—"
            lines.append(f"| {tid} | {fpp:.3f} | {rs:.3f} | {pathway} |")
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="run_summary_exporter",
        description="Export combined pipeline run summary.",
    )
    parser.add_argument("inputs", nargs="+", metavar="JSON")
    parser.add_argument("--output-dir", default="reports/run_summary")
    parser.add_argument("--fpp", type=float, default=0.5)
    parser.add_argument("--alert-fpp", type=float, default=0.1)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args(argv)

    summary = build_run_summary(
        args.inputs,
        fpp_threshold=args.fpp,
        alert_fpp_threshold=args.alert_fpp,
        top_n=args.top,
    )
    paths = write_run_summary(summary, args.output_dir)
    print(f"Written: {paths['markdown']}, {paths['json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
