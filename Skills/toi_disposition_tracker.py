"""Compare two TOI CSV snapshots to find disposition changes.

Detects newly confirmed planets, newly flagged false positives, added/removed
TOIs, and other disposition changes between two ExoFOP TOI table exports.

Public API
----------
DispositionChange(tic_id, toi, old_disposition, new_disposition, change_type)
ToiDiffResult(n_added, n_removed, n_confirmed, n_new_fp, n_changed, changes, flag)
diff_toi_snapshots(old_csv, new_csv, *, toi_col, tic_col, disp_col) -> ToiDiffResult
format_toi_diff(result) -> str
"""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass


@dataclass(frozen=True)
class DispositionChange:
    tic_id: str
    toi: str
    old_disposition: str | None   # None if new
    new_disposition: str | None   # None if removed
    change_type: str              # "added" | "removed" | "confirmed" | "fp" | "changed"


@dataclass(frozen=True)
class ToiDiffResult:
    n_added: int
    n_removed: int
    n_confirmed: int   # anything → CP
    n_new_fp: int      # anything → FP
    n_changed: int     # other disposition change
    changes: tuple[DispositionChange, ...]
    flag: str          # "OK" | "EMPTY" | "INVALID"


def _parse_csv(text: str, toi_col: str, tic_col: str, disp_col: str) -> dict[str, dict]:
    """Parse CSV text into a dict keyed by TOI identifier."""
    reader = csv.DictReader(io.StringIO(text.strip()))
    if reader.fieldnames is None:
        return {}
    rows: dict[str, dict] = {}
    for row in reader:
        toi = row.get(toi_col, "").strip()
        if not toi:
            continue
        rows[toi] = {
            "tic_id": row.get(tic_col, "").strip(),
            "disposition": row.get(disp_col, "").strip(),
        }
    return rows


def diff_toi_snapshots(
    old_csv: str,
    new_csv: str,
    *,
    toi_col: str = "TOI",
    tic_col: str = "TIC ID",
    disp_col: str = "TFOPWG Disposition",
) -> ToiDiffResult:
    """Compare two TOI CSV snapshots for disposition changes.

    Args:
        old_csv: CSV text of the older TOI table.
        new_csv: CSV text of the newer TOI table.
        toi_col: Column name for the TOI identifier.
        tic_col: Column name for the TIC ID.
        disp_col: Column name for the TFOPWG disposition.

    Returns:
        :class:`ToiDiffResult`.
    """
    try:
        old_rows = _parse_csv(old_csv, toi_col, tic_col, disp_col)
        new_rows = _parse_csv(new_csv, toi_col, tic_col, disp_col)
    except Exception:
        return ToiDiffResult(0, 0, 0, 0, 0, (), "INVALID")

    if not old_rows and not new_rows:
        return ToiDiffResult(0, 0, 0, 0, 0, (), "EMPTY")

    changes: list[DispositionChange] = []
    n_added = n_removed = n_confirmed = n_new_fp = n_changed = 0

    all_tois = set(old_rows) | set(new_rows)
    for toi in sorted(all_tois):
        if toi in old_rows and toi not in new_rows:
            # Removed
            n_removed += 1
            changes.append(DispositionChange(
                tic_id=old_rows[toi]["tic_id"],
                toi=toi,
                old_disposition=old_rows[toi]["disposition"] or None,
                new_disposition=None,
                change_type="removed",
            ))
        elif toi not in old_rows and toi in new_rows:
            # Added
            n_added += 1
            changes.append(DispositionChange(
                tic_id=new_rows[toi]["tic_id"],
                toi=toi,
                old_disposition=None,
                new_disposition=new_rows[toi]["disposition"] or None,
                change_type="added",
            ))
        else:
            old_disp = old_rows[toi]["disposition"]
            new_disp = new_rows[toi]["disposition"]
            if old_disp == new_disp:
                continue
            # Disposition changed
            change_type: str
            if new_disp.upper() in ("CP", "CONFIRMED PLANET"):
                change_type = "confirmed"
                n_confirmed += 1
            elif new_disp.upper() in ("FP", "FALSE POSITIVE"):
                change_type = "fp"
                n_new_fp += 1
            else:
                change_type = "changed"
                n_changed += 1
            changes.append(DispositionChange(
                tic_id=new_rows[toi]["tic_id"],
                toi=toi,
                old_disposition=old_disp or None,
                new_disposition=new_disp or None,
                change_type=change_type,
            ))

    flag = "EMPTY" if not changes and not old_rows and not new_rows else "OK"
    return ToiDiffResult(
        n_added=n_added,
        n_removed=n_removed,
        n_confirmed=n_confirmed,
        n_new_fp=n_new_fp,
        n_changed=n_changed,
        changes=tuple(changes),
        flag=flag,
    )


def format_toi_diff(result: ToiDiffResult) -> str:
    """Format TOI diff result as Markdown."""
    lines = [
        "## TOI Disposition Tracker",
        "",
        f"- Added: {result.n_added}",
        f"- Removed: {result.n_removed}",
        f"- Newly confirmed (→CP): {result.n_confirmed}",
        f"- Newly FP (→FP): {result.n_new_fp}",
        f"- Other changes: {result.n_changed}",
        f"- **Flag: {result.flag}**",
        "",
    ]
    if result.changes:
        lines.append("### Changes")
        lines.append("")
        lines.append("| TOI | TIC ID | Old | New | Type |")
        lines.append("|-----|--------|-----|-----|------|")
        for c in result.changes:
            lines.append(
                f"| {c.toi} | {c.tic_id} | {c.old_disposition or '—'} "
                f"| {c.new_disposition or '—'} | {c.change_type} |"
            )
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="toi_disposition_tracker",
        description="Compare two TOI CSV snapshots for disposition changes.",
    )
    parser.add_argument("old_csv", help="Path to old TOI CSV file")
    parser.add_argument("new_csv", help="Path to new TOI CSV file")
    parser.add_argument("--toi-col", default="TOI")
    parser.add_argument("--tic-col", default="TIC ID")
    parser.add_argument("--disp-col", default="TFOPWG Disposition")
    args = parser.parse_args(argv)

    from pathlib import Path
    old_text = Path(args.old_csv).read_text()
    new_text = Path(args.new_csv).read_text()
    result = diff_toi_snapshots(
        old_text, new_text,
        toi_col=args.toi_col,
        tic_col=args.tic_col,
        disp_col=args.disp_col,
    )
    print(format_toi_diff(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
