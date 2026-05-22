"""Export candidate annotations with provenance to JSON and CSV formats.

Annotations record human or automated notes attached to a TIC candidate
(e.g. "OOT scatter high", "matched CTOI", "needs RV follow-up").  Each
annotation carries a timestamp, author tag, and category.  This module
manages a simple annotation store and exports it to JSON / CSV.

Public API
----------
Annotation(tic_id, category, note, author, created_at, source)
AnnotationStore(path)           # persistent JSON-backed store
  .add(annotation)
  .get(tic_id) -> list[Annotation]
  .all() -> list[Annotation]
  .remove(tic_id, index)
  .summary() -> dict
export_annotations_csv(annotations, path) -> Path
export_annotations_json(annotations, path) -> Path
"""
from __future__ import annotations

import contextlib
import csv
import datetime as _dt
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Annotation:
    tic_id: int
    category: str         # e.g. "vetting", "follow-up", "note", "flag"
    note: str
    author: str = "auto"
    created_at: str = ""  # ISO-8601 UTC; auto-filled if empty
    source: str = ""      # pipeline run, file, or manual


def _now_iso() -> str:
    return _dt.datetime.now(tz=_dt.UTC).isoformat(timespec="seconds")


class AnnotationStore:
    """Persistent JSON-backed annotation store."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._entries: list[dict] = []
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text())
                self._entries = data.get("annotations", [])
            except (json.JSONDecodeError, KeyError):
                self._entries = []

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {"last_updated": _now_iso(), "annotations": self._entries},
            indent=2,
        )
        fd, tmp = tempfile.mkstemp(dir=self._path.parent, suffix=".tmp")
        try:
            os.write(fd, payload.encode())
            os.close(fd)
            os.replace(tmp, self._path)
        except Exception:
            os.close(fd)
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise

    def add(self, annotation: Annotation) -> None:
        """Add an annotation, filling created_at if blank."""
        d = asdict(annotation)
        if not d["created_at"]:
            d["created_at"] = _now_iso()
        self._entries.append(d)
        self._save()

    def get(self, tic_id: int) -> list[Annotation]:
        """Return all annotations for a given TIC ID."""
        return [
            Annotation(**e) for e in self._entries
            if e.get("tic_id") == tic_id
        ]

    def all(self) -> list[Annotation]:
        """Return all annotations."""
        return [Annotation(**e) for e in self._entries]

    def remove(self, tic_id: int, index: int) -> bool:
        """Remove the index-th annotation for tic_id (0-based). Returns True if removed."""
        matches = [(i, e) for i, e in enumerate(self._entries) if e.get("tic_id") == tic_id]
        if index < 0 or index >= len(matches):
            return False
        real_idx = matches[index][0]
        self._entries.pop(real_idx)
        self._save()
        return True

    def summary(self) -> dict:
        """Return summary stats."""
        from collections import Counter
        cats = Counter(e.get("category", "") for e in self._entries)
        tic_ids = {e.get("tic_id") for e in self._entries}
        return {
            "n_annotations": len(self._entries),
            "n_targets": len(tic_ids),
            "categories": dict(cats),
        }


def export_annotations_csv(annotations: list[Annotation], path: Path) -> Path:
    """Write annotations to a CSV file.

    Args:
        annotations: List of :class:`Annotation` objects.
        path: Output CSV path.

    Returns:
        Resolved output path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["tic_id", "category", "note", "author", "created_at", "source"],
        )
        writer.writeheader()
        for ann in annotations:
            writer.writerow(asdict(ann))
    return path


def export_annotations_json(annotations: list[Annotation], path: Path) -> Path:
    """Write annotations to a JSON file.

    Args:
        annotations: List of :class:`Annotation` objects.
        path: Output JSON path.

    Returns:
        Resolved output path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "exported_at": _now_iso(),
        "n_annotations": len(annotations),
        "annotations": [asdict(a) for a in annotations],
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="candidate_annotation_exporter",
        description="Export candidate annotations to JSON/CSV.",
    )
    parser.add_argument("--store", type=Path, default=Path("data/annotations.json"))
    parser.add_argument("--export-csv", type=Path, default=None)
    parser.add_argument("--export-json", type=Path, default=None)
    args = parser.parse_args(argv)

    store = AnnotationStore(args.store)
    anns = store.all()
    print(f"Loaded {len(anns)} annotations from {args.store}")

    if args.export_csv:
        p = export_annotations_csv(anns, args.export_csv)
        print(f"CSV written to {p}")
    if args.export_json:
        p = export_annotations_json(anns, args.export_json)
        print(f"JSON written to {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
