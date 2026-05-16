"""Check whether a TIC ID is already on the ExoFOP TESS TOI list.

Queries the same TOI table used by ``fetch_tess_toi.py``.  Returns the TOI
disposition, number, period, and epoch if found — useful for quickly checking
before investing pipeline time on a target.

Public API
----------
check_toi(tic_id, *, toi_table_fn) -> dict | None
format_toi_result(result) -> str
"""
from __future__ import annotations

import io
import urllib.request
from typing import Any, Callable

# Same endpoint as fetch_tess_toi.py
_EXOFOP_URL: str = (
    "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv"
)

# ExoFOP disposition labels considered "active" (not dispositioned as junk)
_ACTIVE_DISPOSITIONS: frozenset[str] = frozenset(
    {"PC", "CP", "KP", "FP", "EB", "IS", "O", "V"}
)


# ---------------------------------------------------------------------------
# Default network fetcher
# ---------------------------------------------------------------------------


def _fetch_toi_csv(url: str = _EXOFOP_URL) -> str:
    """Download the ExoFOP TOI CSV and return it as a string."""
    with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310
        return resp.read().decode("utf-8")


def _default_toi_table_fn() -> str:
    return _fetch_toi_csv()


# ---------------------------------------------------------------------------
# CSV parser (no pandas dependency)
# ---------------------------------------------------------------------------


def _parse_toi_csv(csv_text: str) -> list[dict[str, str]]:
    """Parse ExoFOP TOI CSV into a list of row dicts."""
    lines = [ln for ln in csv_text.splitlines() if ln and not ln.startswith("#")]
    if not lines:
        return []
    reader = io.StringIO("\n".join(lines))
    header_line = reader.readline().strip()
    headers = [h.strip() for h in header_line.split(",")]
    rows: list[dict[str, str]] = []
    for line in reader:
        vals = [v.strip() for v in line.strip().split(",")]
        if len(vals) == len(headers):
            rows.append(dict(zip(headers, vals, strict=True)))
    return rows


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def check_toi(
    tic_id: int,
    *,
    toi_table_fn: Callable[[], str] | None = None,
) -> dict[str, Any] | None:
    """Check whether a TIC ID appears in the ExoFOP TOI list.

    Args:
        tic_id: TESS Input Catalog identifier (integer).
        toi_table_fn: Callable that returns the TOI CSV text.  Defaults to a
            live HTTP fetch from ExoFOP.  Override in tests to avoid network.

    Returns:
        Dict with keys ``toi``, ``tic_id``, ``disposition``, ``period_days``,
        ``epoch_bjd``, ``depth_ppm``, ``duration_hours`` — or ``None`` if the
        TIC ID is not in the list.
    """
    fn = toi_table_fn or _default_toi_table_fn
    csv_text = fn()
    rows = _parse_toi_csv(csv_text)

    target_str = str(tic_id)
    for row in rows:
        # ExoFOP uses "TIC ID" or "tid" column
        row_tic = row.get("TIC ID", row.get("tid", "")).strip()
        if row_tic == target_str:
            return {
                "toi": row.get("TOI", ""),
                "tic_id": tic_id,
                "disposition": row.get("TFOPWG Disposition", row.get("tfopwg_disp", "")),
                "period_days": _safe_float(row.get("Period (days)", row.get("Period", ""))),
                "epoch_bjd": _safe_float(row.get("Epoch (BJD)", row.get("Epoch", ""))),
                "depth_ppm": _safe_float(row.get("Depth (ppm)", row.get("Depth", ""))),
                "duration_hours": _safe_float(row.get("Duration (hours)", row.get("Duration", ""))),
            }
    return None


def format_toi_result(result: dict[str, Any] | None, tic_id: int | None = None) -> str:
    """Return a human-readable one-line status for a TOI lookup result."""
    if result is None:
        suffix = f" TIC {tic_id}" if tic_id is not None else ""
        return f"Not found in ExoFOP TOI list.{suffix}"
    disp = result.get("disposition") or "unknown"
    toi = result.get("toi") or "?"
    period = result.get("period_days")
    period_str = f"  P = {period:.4f} d" if isinstance(period, float) else ""
    return f"TOI {toi}  |  TIC {result['tic_id']}  |  {disp}{period_str}"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _safe_float(s: str) -> float | None:
    """Convert string to float; return None on failure."""
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli(argv: list[str] | None = None) -> int:
    import argparse  # noqa: PLC0415
    import json  # noqa: PLC0415
    import sys  # noqa: PLC0415

    parser = argparse.ArgumentParser(
        prog="toi_checker",
        description="Check whether a TIC ID is on the ExoFOP TOI list.",
    )
    parser.add_argument("tic_id", type=int, metavar="TIC_ID")
    parser.add_argument("--json", action="store_true", help="Output JSON.")
    args = parser.parse_args(argv)

    result = check_toi(args.tic_id)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(format_toi_result(result, tic_id=args.tic_id))

    return 0 if result is not None else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
