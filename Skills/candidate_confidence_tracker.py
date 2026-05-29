"""Track how candidate FPP and detection confidence evolve across pipeline runs.

Persists a per-candidate history of scoring metrics and surfaces trend
information (improving, stable, degrading).

Public API
----------
ConfidenceSnapshot(run_at, fpp, detection_confidence, pathway, scorer, note)
ConfidenceTrend(tic_id, period_days, n_runs, first_fpp, latest_fpp,
                delta_fpp, trend, snapshots, flag)
CandidateConfidenceTracker(path)
    .record(tic_id, period_days, row, *, note) -> ConfidenceSnapshot
    .trend(tic_id, period_days) -> ConfidenceTrend | None
    .all_trends() -> list[ConfidenceTrend]
format_confidence_trend(trend) -> str
"""
from __future__ import annotations

import contextlib
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ConfidenceSnapshot:
    run_at: float
    fpp: float | None
    detection_confidence: float | None
    pathway: str
    scorer: str
    note: str


@dataclass(frozen=True)
class ConfidenceTrend:
    tic_id: int
    period_days: float
    n_runs: int
    first_fpp: float | None
    latest_fpp: float | None
    delta_fpp: float | None  # latest - first; negative = improving
    trend: str  # "IMPROVING" | "STABLE" | "DEGRADING" | "SINGLE_RUN"
    snapshots: tuple[ConfidenceSnapshot, ...]
    flag: str  # "OK" | "DEGRADING" | "INSUFFICIENT_DATA"


def _safe_float(v: object) -> float | None:
    with contextlib.suppress(TypeError, ValueError):
        return float(v)  # type: ignore[arg-type]
    return None


def _candidate_key(tic_id: int, period_days: float) -> str:
    return f"{tic_id}_{period_days:.6f}"


class CandidateConfidenceTracker:
    """Persistent per-candidate confidence history backed by JSON."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._data: dict[str, list[dict]] = self._load()

    def _load(self) -> dict[str, list[dict]]:
        if not self._path.exists():
            return {}
        try:
            return json.loads(self._path.read_text())
        except Exception:
            return {}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = None
        try:
            fd, tmp = tempfile.mkstemp(dir=self._path.parent, suffix=".tmp")
            with os.fdopen(fd, "w") as fh:
                json.dump(self._data, fh, indent=2)
            os.replace(tmp, self._path)
        except Exception:
            with contextlib.suppress(OSError):
                if tmp:
                    os.unlink(tmp)
            raise

    def record(
        self,
        tic_id: int,
        period_days: float,
        row: dict,
        *,
        note: str = "",
    ) -> ConfidenceSnapshot:
        """Record a scoring snapshot for a candidate."""
        fpp = _safe_float(
            row.get("false_positive_probability")
            or (row.get("scores") or {}).get("false_positive_probability")
        )
        dc = _safe_float(
            row.get("detection_confidence")
            or (row.get("scores") or {}).get("detection_confidence")
        )
        snapshot = {
            "run_at": time.time(),
            "fpp": fpp,
            "detection_confidence": dc,
            "pathway": str(row.get("pathway") or ""),
            "scorer": str((row.get("meta") or {}).get("scorer") or ""),
            "note": note,
        }
        key = _candidate_key(tic_id, period_days)
        if key not in self._data:
            self._data[key] = []
        self._data[key].append(snapshot)
        self._save()
        return ConfidenceSnapshot(
            run_at=snapshot["run_at"],
            fpp=fpp,
            detection_confidence=dc,
            pathway=snapshot["pathway"],
            scorer=snapshot["scorer"],
            note=note,
        )

    def trend(self, tic_id: int, period_days: float) -> ConfidenceTrend | None:
        """Get confidence trend for a specific candidate."""
        key = _candidate_key(tic_id, period_days)
        snaps_raw = self._data.get(key)
        if not snaps_raw:
            return None

        snaps = tuple(
            ConfidenceSnapshot(
                run_at=s["run_at"],
                fpp=s.get("fpp"),
                detection_confidence=s.get("detection_confidence"),
                pathway=s.get("pathway", ""),
                scorer=s.get("scorer", ""),
                note=s.get("note", ""),
            )
            for s in snaps_raw
        )

        fpps = [s.fpp for s in snaps if s.fpp is not None]
        first_fpp = fpps[0] if fpps else None
        latest_fpp = fpps[-1] if fpps else None
        delta_fpp = round(latest_fpp - first_fpp, 4) if (
            first_fpp is not None and latest_fpp is not None) else None

        if len(snaps) == 1:
            trend_label = "SINGLE_RUN"
        elif delta_fpp is None:
            trend_label = "STABLE"
        elif delta_fpp < -0.05:
            trend_label = "IMPROVING"
        elif delta_fpp > 0.05:
            trend_label = "DEGRADING"
        else:
            trend_label = "STABLE"

        flag = "DEGRADING" if trend_label == "DEGRADING" else "OK"

        return ConfidenceTrend(
            tic_id=tic_id,
            period_days=period_days,
            n_runs=len(snaps),
            first_fpp=first_fpp,
            latest_fpp=latest_fpp,
            delta_fpp=delta_fpp,
            trend=trend_label,
            snapshots=snaps,
            flag=flag,
        )

    def all_trends(self) -> list[ConfidenceTrend]:
        """Return trends for all tracked candidates."""
        results = []
        for key in self._data:
            parts = key.rsplit("_", 1)
            if len(parts) != 2:
                continue
            with contextlib.suppress(ValueError):
                tic_id = int(parts[0])
                period_days = float(parts[1])
                t = self.trend(tic_id, period_days)
                if t is not None:
                    results.append(t)
        return results


def format_confidence_trend(trend: ConfidenceTrend) -> str:
    """Format confidence trend as Markdown.

    Args:
        trend: ConfidenceTrend to format.

    Returns:
        Markdown string.
    """
    delta_str = f"{trend.delta_fpp:+.4f}" if trend.delta_fpp is not None else "—"
    first_str = f"{trend.first_fpp:.3f}" if trend.first_fpp is not None else "—"
    latest_str = f"{trend.latest_fpp:.3f}" if trend.latest_fpp is not None else "—"
    lines = [
        f"## Confidence Trend — TIC {trend.tic_id} (P={trend.period_days:.4f} d)\n",
        f"**Trend**: `{trend.trend}` | Runs: {trend.n_runs} | "
        f"FPP: {first_str} → {latest_str} (Δ={delta_str})\n",
        "",
        "| Run | FPP | DC | Pathway | Scorer |",
        "|---|---|---|---|---|",
    ]
    for i, s in enumerate(trend.snapshots, 1):
        fpp_str = f"{s.fpp:.3f}" if s.fpp is not None else "—"
        dc_str = f"{s.detection_confidence:.3f}" if s.detection_confidence is not None else "—"
        lines.append(f"| {i} | {fpp_str} | {dc_str} | {s.pathway or '—'} | {s.scorer or '—'} |")
    return "\n".join(lines)


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Track candidate confidence.")
    sub = parser.add_subparsers(dest="cmd")

    rec_p = sub.add_parser("record")
    rec_p.add_argument("--db", required=True)
    rec_p.add_argument("--tic-id", type=int, required=True)
    rec_p.add_argument("--period", type=float, required=True)
    rec_p.add_argument("--row", required=True, help="JSON row file.")
    rec_p.add_argument("--note", default="")

    show_p = sub.add_parser("show")
    show_p.add_argument("--db", required=True)
    show_p.add_argument("--tic-id", type=int, required=True)
    show_p.add_argument("--period", type=float, required=True)

    args = parser.parse_args(argv)
    if args.cmd == "record":
        tracker = CandidateConfidenceTracker(args.db)
        row = json.loads(Path(args.row).read_text())
        if isinstance(row, list):
            row = row[0]
        tracker.record(args.tic_id, args.period, row, note=args.note)
        print("Recorded.")
    elif args.cmd == "show":
        tracker = CandidateConfidenceTracker(args.db)
        trend = tracker.trend(args.tic_id, args.period)
        if trend:
            print(format_confidence_trend(trend))
        else:
            print("No data for this candidate.")
    else:
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
