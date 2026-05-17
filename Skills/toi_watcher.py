"""Monitor ExoFOP TOI list for status changes relative to a saved snapshot.

On first run, saves a snapshot.  On subsequent runs, compares with the
snapshot and reports new TOIs, disposition changes, and removed entries.

Public API
----------
TOIChange(tic_id, toi, field, old_value, new_value, change_type)
TOIWatchResult(new_tois, changed_tois, removed_tois, snapshot_age_hours,
               generated_at)
watch_toi_list(snapshot_path, *, toi_table_fn) -> TOIWatchResult
format_watch_result(result) -> str
"""
from __future__ import annotations

import contextlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class TOIChange:
    tic_id: int
    toi: str
    field: str
    old_value: str
    new_value: str
    change_type: str   # "new" | "changed" | "removed"


@dataclass(frozen=True)
class TOIWatchResult:
    new_tois: tuple[TOIChange, ...]
    changed_tois: tuple[TOIChange, ...]
    removed_tois: tuple[TOIChange, ...]
    snapshot_age_hours: float | None
    generated_at: str


_WATCH_FIELDS = ("disposition", "period", "epoch", "depth", "duration")


def _default_toi_fn() -> list[dict]:
    """Stub — returns empty list; real impl fetches ExoFOP CSV."""
    return []


def _toi_key(row: dict) -> str:
    """Canonical TOI identifier string."""
    return str(row.get("toi") or row.get("TOI") or "")


def _extract(row: dict, field: str) -> str:
    """Extract a field from a TOI row as a string."""
    for k in (field, field.upper(), field.capitalize()):
        val = row.get(k)
        if val is not None:
            return str(val).strip()
    return ""


def watch_toi_list(
    snapshot_path: Path | str,
    *,
    toi_table_fn=None,
) -> TOIWatchResult:
    """Compare current TOI list against saved snapshot.

    Args:
        snapshot_path: Path to the JSON snapshot file (created on first run).
        toi_table_fn: Injectable; called with no args, returns list of row dicts.

    Returns:
        :class:`TOIWatchResult` with detected changes.
    """
    if toi_table_fn is None:
        toi_table_fn = _default_toi_fn

    snap_path = Path(snapshot_path)
    now_str = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Load current TOI list
    current_rows = toi_table_fn()
    current: dict[str, dict] = {}
    for row in current_rows:
        key = _toi_key(row)
        if key:
            current[key] = row

    snapshot_age: float | None = None
    previous: dict[str, dict] = {}

    if snap_path.exists():
        try:
            snap_data = json.loads(snap_path.read_text())
            previous = snap_data.get("tois", {})
            snap_ts = snap_data.get("generated_at", "")
            if snap_ts:
                try:
                    snap_dt = datetime.fromisoformat(snap_ts.replace("Z", "+00:00"))
                    now_dt = datetime.now(UTC)
                    snapshot_age = (now_dt - snap_dt).total_seconds() / 3600.0
                except ValueError:
                    pass
        except (json.JSONDecodeError, AttributeError):
            pass

    new_tois: list[TOIChange] = []
    changed_tois: list[TOIChange] = []
    removed_tois: list[TOIChange] = []

    # New and changed
    for key, row in current.items():
        tic_id = int(row.get("tic_id") or row.get("TIC") or 0)
        if key not in previous:
            new_tois.append(TOIChange(
                tic_id=tic_id, toi=key, field="all",
                old_value="", new_value="added",
                change_type="new",
            ))
        else:
            prev_row = previous[key]
            for field in _WATCH_FIELDS:
                old = _extract(prev_row, field)
                new = _extract(row, field)
                if old != new and (old or new):
                    changed_tois.append(TOIChange(
                        tic_id=tic_id, toi=key, field=field,
                        old_value=old, new_value=new,
                        change_type="changed",
                    ))

    # Removed
    for key, prev_row in previous.items():
        if key not in current:
            tic_id = int(prev_row.get("tic_id") or prev_row.get("TIC") or 0)
            removed_tois.append(TOIChange(
                tic_id=tic_id, toi=key, field="all",
                old_value="present", new_value="",
                change_type="removed",
            ))

    # Save snapshot
    snap_path.parent.mkdir(parents=True, exist_ok=True)
    snap_data_out = {"generated_at": now_str, "tois": current}
    fd, tmp = tempfile.mkstemp(dir=snap_path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(snap_data_out, fh, indent=2)
        os.replace(tmp, snap_path)
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise

    return TOIWatchResult(
        new_tois=tuple(new_tois),
        changed_tois=tuple(changed_tois),
        removed_tois=tuple(removed_tois),
        snapshot_age_hours=snapshot_age,
        generated_at=now_str,
    )


def format_watch_result(result: TOIWatchResult) -> str:
    """Format watch result as Markdown."""
    age_str = (
        f"{result.snapshot_age_hours:.1f} h" if result.snapshot_age_hours is not None
        else "first run"
    )
    lines = [
        "## TOI Watch Report",
        "",
        f"- Generated: {result.generated_at}",
        f"- Snapshot age: {age_str}",
        f"- New TOIs: {len(result.new_tois)}",
        f"- Changed fields: {len(result.changed_tois)}",
        f"- Removed TOIs: {len(result.removed_tois)}",
    ]
    if result.new_tois:
        lines += ["", "### New TOIs"]
        for c in result.new_tois[:10]:
            lines.append(f"- TOI {c.toi} (TIC {c.tic_id})")
    if result.changed_tois:
        lines += ["", "### Changed Fields"]
        for c in result.changed_tois[:10]:
            lines.append(
                f"- TOI {c.toi} `{c.field}`: {c.old_value!r} → {c.new_value!r}"
            )
    if result.removed_tois:
        lines += ["", "### Removed TOIs"]
        for c in result.removed_tois[:10]:
            lines.append(f"- TOI {c.toi} (TIC {c.tic_id})")
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="toi_watcher",
        description="Monitor ExoFOP TOI list for status changes.",
    )
    parser.add_argument("--snapshot", default="data/toi_snapshot.json")
    args = parser.parse_args(argv)

    result = watch_toi_list(args.snapshot)
    print(format_watch_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
