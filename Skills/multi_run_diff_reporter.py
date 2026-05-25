"""Compare two pipeline JSON result files for the same target.

Flags regressions (FPP worsened, planet_posterior decreased) and improvements
(FPP improved, planet_posterior increased) between successive pipeline runs.

Public API
----------
SignalDiff(candidate_id, field, old_value, new_value, change_type)
RunDiffResult(n_signals_old, n_signals_new, n_improved, n_regressed,
              n_pathway_changes, diffs, flag)
diff_pipeline_runs(old_rows, new_rows, *, fpp_threshold) -> RunDiffResult
load_and_diff(old_path, new_path, **kwargs) -> RunDiffResult
format_run_diff(result) -> str
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SignalDiff:
    candidate_id: str
    field: str
    old_value: object
    new_value: object
    change_type: str   # "improved" | "regressed" | "changed" | "added" | "removed"


@dataclass(frozen=True)
class RunDiffResult:
    n_signals_old: int
    n_signals_new: int
    n_improved: int
    n_regressed: int
    n_pathway_changes: int
    diffs: tuple[SignalDiff, ...]
    flag: str   # "OK" | "NO_CHANGE" | "EMPTY" | "INVALID"


def _get_fpp(row: dict) -> float | None:
    """Extract FPP from a result dict, trying common key paths."""
    if "scores" in row and isinstance(row["scores"], dict):
        v = row["scores"].get("false_positive_probability")
        if v is not None:
            return float(v)
    v = row.get("fpp") or row.get("false_positive_probability")
    if v is not None:
        return float(v)
    return None


def _get_planet_posterior(row: dict) -> float | None:
    """Extract planet_candidate posterior from a result dict."""
    if "posterior" in row and isinstance(row["posterior"], dict):
        v = row["posterior"].get("planet_candidate")
        if v is not None:
            return float(v)
    v = row.get("planet_posterior") or row.get("planet_candidate")
    if v is not None:
        return float(v)
    return None


def _get_pathway(row: dict) -> str | None:
    return row.get("pathway") or row.get("submission_pathway") or None


def diff_pipeline_runs(
    old_rows: list[dict],
    new_rows: list[dict],
    *,
    fpp_threshold: float = 0.05,
) -> RunDiffResult:
    """Compare two pipeline run result lists signal-by-signal.

    Args:
        old_rows: List of result dicts from the older run.
        new_rows: List of result dicts from the newer run.
        fpp_threshold: Minimum absolute FPP change to flag as improved/regressed.

    Returns:
        :class:`RunDiffResult`.
    """
    if not isinstance(old_rows, list) or not isinstance(new_rows, list):
        return RunDiffResult(0, 0, 0, 0, 0, (), "INVALID")

    if not old_rows and not new_rows:
        return RunDiffResult(0, 0, 0, 0, 0, (), "EMPTY")

    # Index by candidate_id
    old_by_id: dict[str, dict] = {}
    new_by_id: dict[str, dict] = {}
    for row in old_rows:
        cid = str(row.get("candidate_id", ""))
        if cid:
            old_by_id[cid] = row
    for row in new_rows:
        cid = str(row.get("candidate_id", ""))
        if cid:
            new_by_id[cid] = row

    diffs: list[SignalDiff] = []
    n_improved = 0
    n_regressed = 0
    n_pathway_changes = 0

    all_ids = sorted(set(old_by_id) | set(new_by_id))
    for cid in all_ids:
        if cid in old_by_id and cid not in new_by_id:
            diffs.append(SignalDiff(
                candidate_id=cid,
                field="candidate_id",
                old_value=cid,
                new_value=None,
                change_type="removed",
            ))
            continue
        if cid not in old_by_id and cid in new_by_id:
            diffs.append(SignalDiff(
                candidate_id=cid,
                field="candidate_id",
                old_value=None,
                new_value=cid,
                change_type="added",
            ))
            continue

        old_row = old_by_id[cid]
        new_row = new_by_id[cid]

        # FPP comparison
        old_fpp = _get_fpp(old_row)
        new_fpp = _get_fpp(new_row)
        if old_fpp is not None and new_fpp is not None:
            delta = new_fpp - old_fpp
            if abs(delta) >= fpp_threshold:
                change_type = "improved" if delta < 0 else "regressed"
                if change_type == "improved":
                    n_improved += 1
                else:
                    n_regressed += 1
                diffs.append(SignalDiff(
                    candidate_id=cid,
                    field="fpp",
                    old_value=old_fpp,
                    new_value=new_fpp,
                    change_type=change_type,
                ))

        # planet_posterior comparison
        old_pp = _get_planet_posterior(old_row)
        new_pp = _get_planet_posterior(new_row)
        if old_pp is not None and new_pp is not None:
            delta_pp = new_pp - old_pp
            if abs(delta_pp) >= fpp_threshold:
                change_type = "improved" if delta_pp > 0 else "regressed"
                # Don't double-count with FPP
                if old_fpp is None or new_fpp is None or abs(new_fpp - old_fpp) < fpp_threshold:
                    if change_type == "improved":
                        n_improved += 1
                    else:
                        n_regressed += 1
                diffs.append(SignalDiff(
                    candidate_id=cid,
                    field="planet_posterior",
                    old_value=old_pp,
                    new_value=new_pp,
                    change_type=change_type,
                ))

        # Pathway comparison
        old_path = _get_pathway(old_row)
        new_path = _get_pathway(new_row)
        if old_path != new_path and (old_path is not None or new_path is not None):
            n_pathway_changes += 1
            diffs.append(SignalDiff(
                candidate_id=cid,
                field="pathway",
                old_value=old_path,
                new_value=new_path,
                change_type="changed",
            ))

        # Period comparison
        old_period = old_row.get("period_days")
        new_period = new_row.get("period_days")
        if old_period is not None and new_period is not None:
            try:
                if abs(float(new_period) - float(old_period)) > 1e-6:
                    diffs.append(SignalDiff(
                        candidate_id=cid,
                        field="period_days",
                        old_value=old_period,
                        new_value=new_period,
                        change_type="changed",
                    ))
            except (TypeError, ValueError):
                pass

    if not diffs:
        return RunDiffResult(
            n_signals_old=len(old_rows),
            n_signals_new=len(new_rows),
            n_improved=0,
            n_regressed=0,
            n_pathway_changes=0,
            diffs=(),
            flag="NO_CHANGE",
        )

    return RunDiffResult(
        n_signals_old=len(old_rows),
        n_signals_new=len(new_rows),
        n_improved=n_improved,
        n_regressed=n_regressed,
        n_pathway_changes=n_pathway_changes,
        diffs=tuple(diffs),
        flag="OK",
    )


def load_and_diff(
    old_path: str | Path,
    new_path: str | Path,
    **kwargs,
) -> RunDiffResult:
    """Load two JSON pipeline result files and diff them."""
    try:
        old_data = json.loads(Path(old_path).read_text())
        new_data = json.loads(Path(new_path).read_text())
    except Exception:
        return RunDiffResult(0, 0, 0, 0, 0, (), "INVALID")
    if isinstance(old_data, dict):
        old_data = [old_data]
    if isinstance(new_data, dict):
        new_data = [new_data]
    return diff_pipeline_runs(old_data, new_data, **kwargs)


def format_run_diff(result: RunDiffResult) -> str:
    """Format run diff result as Markdown."""
    lines = [
        "## Pipeline Run Diff",
        "",
        f"- Signals (old): {result.n_signals_old}",
        f"- Signals (new): {result.n_signals_new}",
        f"- Improved: {result.n_improved}",
        f"- Regressed: {result.n_regressed}",
        f"- Pathway changes: {result.n_pathway_changes}",
        f"- **Flag: {result.flag}**",
        "",
    ]
    if result.diffs:
        lines.append("### Diffs")
        lines.append("")
        lines.append("| Candidate | Field | Old | New | Type |")
        lines.append("|-----------|-------|-----|-----|------|")
        for d in result.diffs:
            lines.append(
                f"| {d.candidate_id} | {d.field} | {d.old_value} "
                f"| {d.new_value} | {d.change_type} |"
            )
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="multi_run_diff_reporter",
        description="Compare two pipeline JSON run outputs.",
    )
    parser.add_argument("old_path", help="Path to old JSON result file")
    parser.add_argument("new_path", help="Path to new JSON result file")
    parser.add_argument("--fpp-threshold", type=float, default=0.05)
    args = parser.parse_args(argv)

    result = load_and_diff(args.old_path, args.new_path, fpp_threshold=args.fpp_threshold)
    print(format_run_diff(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
