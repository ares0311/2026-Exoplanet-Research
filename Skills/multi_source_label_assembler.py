"""Merge label rows from multiple sources into a deduplicated label manifest.

Combines TOI, CTOI, KOI and manual label dicts into a single, conflict-resolved
manifest for CNN training.  All label resolution is deterministic; conflicts are
flagged rather than silently dropped.

Public API
----------
LabelRecord(tic_id, label, source, confidence, period_days, epoch,
            duration_hours, conflict)
LabelManifest(records, n_positive, n_negative, n_conflicts, sources,
              created_at, flag)
assemble_labels(source_rows, *, conflict_policy, max_fp_ratio,
                output_path) -> LabelManifest
format_label_manifest(manifest) -> str
"""
from __future__ import annotations

import contextlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LabelRecord:
    """A single labelled training example."""

    tic_id: str
    label: int          # 1 = planet candidate, 0 = false positive
    source: str         # "toi" | "ctoi" | "koi" | "manual"
    confidence: float   # 0.0–1.0
    period_days: float | None
    epoch: float | None
    duration_hours: float | None
    conflict: bool      # True if other sources disagreed on label


@dataclass(frozen=True)
class LabelManifest:
    """Deduplicated label manifest ready for CNN training."""

    records: tuple[LabelRecord, ...]
    n_positive: int
    n_negative: int
    n_conflicts: int
    sources: tuple[str, ...]
    created_at: str
    flag: str  # "OK" | "EMPTY" | "INVALID"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_row(row: dict) -> str | None:
    """Return an error string if *row* is malformed, else None."""
    if "tic_id" not in row:
        return "missing tic_id"
    if "label" not in row:
        return "missing label"
    if row["label"] not in (0, 1):
        return f"label must be 0 or 1, got {row['label']!r}"
    if "source" not in row:
        return "missing source"
    if "confidence" not in row:
        return "missing confidence"
    return None


def _resolve_conflict(
    candidates: list[dict],
    policy: str,
) -> dict:
    """Choose one row from conflicting candidates per *policy*."""
    if policy == "higher_confidence":
        return max(candidates, key=lambda r: r["confidence"])
    if policy == "majority":
        pos = sum(1 for r in candidates if r["label"] == 1)
        neg = len(candidates) - pos
        majority_label = 1 if pos > neg else 0
        same = [r for r in candidates if r["label"] == majority_label]
        return max(same, key=lambda r: r["confidence"])
    if policy == "conservative":
        # Prefer FP (label=0) — conservative means we rather reject a planet
        fps = [r for r in candidates if r["label"] == 0]
        if fps:
            return max(fps, key=lambda r: r["confidence"])
        return max(candidates, key=lambda r: r["confidence"])
    # fallback
    return max(candidates, key=lambda r: r["confidence"])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def assemble_labels(
    source_rows: list[dict],
    *,
    conflict_policy: str = "higher_confidence",
    max_fp_ratio: float = 3.0,
    output_path: Path | None = None,
) -> LabelManifest:
    """Merge label rows from multiple sources into a deduplicated manifest.

    Args:
        source_rows:     List of dicts, each with keys: tic_id (str), label
                         (int 0/1), source (str), confidence (float).
                         Optional keys: period_days, epoch, duration_hours.
        conflict_policy: How to resolve same-TIC different-label conflicts.
                         One of "higher_confidence", "majority", "conservative".
        max_fp_ratio:    Maximum ratio of negatives to positives.  Excess
                         negatives are trimmed (lowest-confidence first).
        output_path:     If given, write the manifest as JSON (atomic write).

    Returns:
        :class:`LabelManifest`
    """
    now = datetime.now(UTC).isoformat()

    if not source_rows:
        return LabelManifest(
            records=(), n_positive=0, n_negative=0, n_conflicts=0,
            sources=(), created_at=now, flag="EMPTY",
        )

    # Validate; skip invalid rows
    valid_rows: list[dict] = []
    for row in source_rows:
        err = _validate_row(row)
        if err is None:
            valid_rows.append(row)

    if not valid_rows:
        return LabelManifest(
            records=(), n_positive=0, n_negative=0, n_conflicts=0,
            sources=(), created_at=now, flag="INVALID",
        )

    # Group by tic_id
    by_tic: dict[str, list[dict]] = {}
    for row in valid_rows:
        key = str(row["tic_id"])
        by_tic.setdefault(key, []).append(row)

    records: list[LabelRecord] = []
    n_conflicts = 0
    sources_seen: set[str] = set()

    for tic_id, grp in by_tic.items():
        sources_seen.update(r["source"] for r in grp)
        labels_seen = {r["label"] for r in grp}

        if len(labels_seen) == 1:
            # No conflict — keep highest-confidence row
            best = max(grp, key=lambda r: r["confidence"])
            conflict = False
        else:
            # Conflict — resolve and mark
            best = _resolve_conflict(grp, conflict_policy)
            conflict = True
            n_conflicts += 1

        _pd = best.get("period_days")
        _ep = best.get("epoch")
        _dh = best.get("duration_hours")
        records.append(
            LabelRecord(
                tic_id=tic_id,
                label=best["label"],
                source=best["source"],
                confidence=float(best["confidence"]),
                period_days=float(_pd) if _pd is not None else None,
                epoch=float(_ep) if _ep is not None else None,
                duration_hours=float(_dh) if _dh is not None else None,
                conflict=conflict,
            )
        )

    # Apply max_fp_ratio cap
    positives = [r for r in records if r.label == 1]
    negatives = [r for r in records if r.label == 0]
    n_pos = len(positives)

    if n_pos > 0 and len(negatives) > max_fp_ratio * n_pos:
        max_neg = int(max_fp_ratio * n_pos)
        negatives_sorted = sorted(negatives, key=lambda r: r.confidence, reverse=True)
        negatives = negatives_sorted[:max_neg]

    records = positives + negatives

    n_positive = sum(1 for r in records if r.label == 1)
    n_negative = sum(1 for r in records if r.label == 0)

    manifest = LabelManifest(
        records=tuple(records),
        n_positive=n_positive,
        n_negative=n_negative,
        n_conflicts=n_conflicts,
        sources=tuple(sorted(sources_seen)),
        created_at=now,
        flag="OK",
    )

    if output_path is not None:
        _save_manifest(manifest, Path(output_path))

    return manifest


def _save_manifest(manifest: LabelManifest, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "flag": manifest.flag,
        "created_at": manifest.created_at,
        "n_positive": manifest.n_positive,
        "n_negative": manifest.n_negative,
        "n_conflicts": manifest.n_conflicts,
        "sources": list(manifest.sources),
        "records": [
            {
                "tic_id": r.tic_id,
                "label": r.label,
                "source": r.source,
                "confidence": r.confidence,
                "period_days": r.period_days,
                "epoch": r.epoch,
                "duration_hours": r.duration_hours,
                "conflict": r.conflict,
            }
            for r in manifest.records
        ],
    }
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(data, fh, indent=2)
        os.replace(tmp, str(path))
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def format_label_manifest(manifest: LabelManifest) -> str:
    """Return a Markdown summary of a :class:`LabelManifest`."""
    lines = [
        "## Label Manifest",
        "",
        f"**Flag**: {manifest.flag}",
        f"**Created at**: {manifest.created_at}",
        f"**Total records**: {len(manifest.records)}",
        f"- Positive (planet): {manifest.n_positive}",
        f"- Negative (FP): {manifest.n_negative}",
        f"- Conflicts resolved: {manifest.n_conflicts}",
        f"- Sources: {', '.join(manifest.sources) if manifest.sources else '(none)'}",
    ]
    if manifest.n_positive > 0:
        ratio = manifest.n_negative / manifest.n_positive
        lines.append(f"- FP/planet ratio: {ratio:.2f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="multi_source_label_assembler",
        description="Merge multi-source label JSON files into a manifest.",
    )
    parser.add_argument("inputs", nargs="+", help="Input JSON files with label rows.")
    parser.add_argument("--output", default=None, help="Output manifest JSON path.")
    parser.add_argument("--conflict-policy", default="higher_confidence",
                        choices=["higher_confidence", "majority", "conservative"])
    parser.add_argument("--max-fp-ratio", type=float, default=3.0)
    args = parser.parse_args(argv)

    rows: list[dict] = []
    for inp in args.inputs:
        data = json.loads(Path(inp).read_text())
        if isinstance(data, list):
            rows.extend(data)
        elif isinstance(data, dict) and "rows" in data:
            rows.extend(data["rows"])

    out_path = Path(args.output) if args.output else None
    manifest = assemble_labels(
        rows,
        conflict_policy=args.conflict_policy,
        max_fp_ratio=args.max_fp_ratio,
        output_path=out_path,
    )
    print(format_label_manifest(manifest))
    return 0 if manifest.flag == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
