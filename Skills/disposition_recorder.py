"""Record and retrieve formal vetting dispositions for planet candidates.

Tracks official dispositions (PC, FP, CP, EB, IS, UNK) with confidence
scores and full history, so vetting trends across rounds can be audited.
Distinct from ``candidate_notes`` (free text), ``vetting_scorecard``
(numeric scores), and ``candidate_changelog_tracker`` (field diffs).

Public API
----------
VALID_STATUSES: frozenset[str]
Disposition(tic_id, status, confidence, note, recorded_at, author)
DispositionResult(tic_id, current_status, current_confidence,
                  n_records, history, flag)
record_disposition(tic_id, status, *, confidence, note, author,
                   store_path) -> DispositionResult
get_disposition_history(tic_id, store_path) -> DispositionResult
format_disposition_result(result) -> str
"""
from __future__ import annotations

import contextlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

VALID_STATUSES: frozenset[str] = frozenset({
    "PC",   # Planet Candidate
    "FP",   # False Positive
    "CP",   # Confirmed Planet
    "EB",   # Eclipsing Binary
    "IS",   # Instrumental Systematic
    "UNK",  # Unknown / Needs review
})


@dataclass(frozen=True)
class Disposition:
    tic_id: int | str
    status: str         # one of VALID_STATUSES
    confidence: float   # [0, 1]
    note: str
    recorded_at: str    # ISO-8601
    author: str


@dataclass(frozen=True)
class DispositionResult:
    tic_id: int | str
    current_status: str | None
    current_confidence: float | None
    n_records: int
    history: tuple[Disposition, ...]
    flag: str  # "OK" | "EMPTY" | "INVALID"


def _load_store(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_store(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w") as fh:
            json.dump(data, fh, indent=2)
        os.replace(tmp_path, path)
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise


def record_disposition(
    tic_id: int | str,
    status: str,
    *,
    confidence: float = 0.5,
    note: str = "",
    author: str = "unknown",
    store_path: Path | str = Path("data/dispositions.json"),
) -> DispositionResult:
    """Append a new disposition for *tic_id*.

    Args:
        tic_id: TIC identifier.
        status: Disposition code — one of ``VALID_STATUSES``.
        confidence: Certainty of this disposition in [0, 1].
        note: Optional free-text explanation.
        author: Responsible person / process.
        store_path: JSON store file path.

    Returns:
        :class:`DispositionResult` reflecting post-append state.
    """
    status = status.upper()
    if status not in VALID_STATUSES:
        return DispositionResult(tic_id, None, None, 0, (), "INVALID")
    if not (0.0 <= confidence <= 1.0):
        return DispositionResult(tic_id, None, None, 0, (), "INVALID")

    store_path = Path(store_path)
    key = str(tic_id)
    data = _load_store(store_path)
    if key not in data:
        data[key] = []

    entry = {
        "status": status,
        "confidence": round(confidence, 4),
        "note": note,
        "recorded_at": datetime.now(UTC).replace(tzinfo=None).isoformat() + "Z",
        "author": author,
    }
    data[key].append(entry)
    _save_store(store_path, data)

    return _build_result(tic_id, data[key])


def get_disposition_history(
    tic_id: int | str,
    store_path: Path | str = Path("data/dispositions.json"),
) -> DispositionResult:
    """Retrieve full disposition history for *tic_id*."""
    store_path = Path(store_path)
    key = str(tic_id)
    data = _load_store(store_path)
    return _build_result(tic_id, data.get(key, []))


def _build_result(tic_id, raw: list) -> DispositionResult:
    if not raw:
        return DispositionResult(tic_id, None, None, 0, (), "EMPTY")
    history = tuple(
        Disposition(
            tic_id=tic_id,
            status=e["status"],
            confidence=e["confidence"],
            note=e.get("note", ""),
            recorded_at=e.get("recorded_at", ""),
            author=e.get("author", "unknown"),
        )
        for e in raw
    )
    latest = history[-1]
    return DispositionResult(
        tic_id=tic_id,
        current_status=latest.status,
        current_confidence=latest.confidence,
        n_records=len(history),
        history=history,
        flag="OK",
    )


def format_disposition_result(result: DispositionResult) -> str:
    """Format disposition result as Markdown."""
    lines = [
        f"## Disposition Recorder — TIC {result.tic_id}",
        "",
        f"- Current status: **{result.current_status or '—'}**",
        f"- Current confidence: {result.current_confidence}",
        f"- Total records: {result.n_records}",
        f"- **Flag: {result.flag}**",
    ]
    if result.history:
        lines += ["", "### History", "| Status | Conf | Author | Recorded at |",
                  "|---|---|---|---|"]
        for d in result.history[-10:]:
            ts = d.recorded_at[:19] if d.recorded_at else "—"
            lines.append(f"| {d.status} | {d.confidence} | {d.author} | {ts} |")
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="disposition_recorder",
        description="Record formal vetting dispositions for planet candidates.",
    )
    parser.add_argument("--tic-id", type=int, required=True)
    parser.add_argument("--status", type=str, default=None)
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--note", type=str, default="")
    parser.add_argument("--author", type=str, default="cli")
    parser.add_argument("--store", type=str, default="data/dispositions.json")
    args = parser.parse_args(argv)

    if args.status:
        result = record_disposition(
            args.tic_id, args.status, confidence=args.confidence,
            note=args.note, author=args.author, store_path=Path(args.store)
        )
    else:
        result = get_disposition_history(args.tic_id, store_path=Path(args.store))
    print(format_disposition_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
