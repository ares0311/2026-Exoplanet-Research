"""Persistent JSON registry of trained models.

Stores model metadata (type, path, metrics, registration time) in a
JSON array file. Supports registering new models, listing all, and
querying for the best model by metric.

Public API
----------
RegistryEntry(model_id, model_type, model_path, auc, brier, n_train,
              registered_at, notes)
RegistryResult(entries, n_models, best_by_auc, flag)
register(registry_path, entry) -> RegistryResult
get_best(registry_path, *, metric) -> RegistryEntry | None
list_models(registry_path) -> RegistryResult
format_registry(result) -> str
"""
from __future__ import annotations

import contextlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class RegistryEntry:
    model_id: str
    model_type: str   # "xgboost" | "cnn" | "ensemble" | "bayesian"
    model_path: str
    auc: float | None
    brier: float | None
    n_train: int | None
    registered_at: str  # ISO-8601
    notes: str


@dataclass(frozen=True)
class RegistryResult:
    entries: tuple[RegistryEntry, ...]
    n_models: int
    best_by_auc: str | None  # model_id
    flag: str  # "OK" | "EMPTY" | "INVALID"


def _entry_from_dict(d: dict) -> RegistryEntry:
    """Deserialise a dict into a RegistryEntry."""
    return RegistryEntry(
        model_id=str(d["model_id"]),
        model_type=str(d["model_type"]),
        model_path=str(d["model_path"]),
        auc=float(d["auc"]) if d.get("auc") is not None else None,
        brier=float(d["brier"]) if d.get("brier") is not None else None,
        n_train=int(d["n_train"]) if d.get("n_train") is not None else None,
        registered_at=str(d.get("registered_at", "")),
        notes=str(d.get("notes", "")),
    )


def _build_result(entries: list[RegistryEntry]) -> RegistryResult:
    """Build a RegistryResult from a list of entries."""
    if not entries:
        return RegistryResult(entries=(), n_models=0, best_by_auc=None, flag="EMPTY")

    auc_candidates = [(e.model_id, e.auc) for e in entries if e.auc is not None]
    best_by_auc = (
        max(auc_candidates, key=lambda t: t[1])[0] if auc_candidates else None
    )

    return RegistryResult(
        entries=tuple(entries),
        n_models=len(entries),
        best_by_auc=best_by_auc,
        flag="OK",
    )


def _load_entries(registry_path: Path) -> list[RegistryEntry]:
    """Load all entries from the registry file. Returns [] if file missing."""
    if not registry_path.exists():
        return []
    with open(registry_path) as fh:
        raw = json.load(fh)
    return [_entry_from_dict(d) for d in raw]


def _save_entries(registry_path: Path, entries: list[RegistryEntry]) -> None:
    """Atomically write entries to the registry file."""
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    data = [asdict(e) for e in entries]
    fd, tmp_path = tempfile.mkstemp(
        dir=registry_path.parent, suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(data, fh, indent=2)
        os.replace(tmp_path, registry_path)
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise


def register(registry_path: Path, entry: RegistryEntry) -> RegistryResult:
    """Register a new model entry.

    Args:
        registry_path: Path to the registry JSON file (created if absent).
        entry: RegistryEntry to add.

    Returns:
        RegistryResult reflecting the updated registry.

    Raises:
        ValueError: If an entry with the same model_id already exists.
    """
    entries = _load_entries(registry_path)
    existing_ids = {e.model_id for e in entries}
    if entry.model_id in existing_ids:
        raise ValueError(f"model_id '{entry.model_id}' already exists in registry.")

    # Stamp registered_at if blank
    stamped = entry
    if not stamped.registered_at:
        stamped = RegistryEntry(
            model_id=entry.model_id,
            model_type=entry.model_type,
            model_path=entry.model_path,
            auc=entry.auc,
            brier=entry.brier,
            n_train=entry.n_train,
            registered_at=datetime.now(tz=UTC).isoformat(),
            notes=entry.notes,
        )

    entries.append(stamped)
    _save_entries(registry_path, entries)
    return _build_result(entries)


def get_best(registry_path: Path, *, metric: str = "auc") -> RegistryEntry | None:
    """Return the best model entry by metric.

    Args:
        registry_path: Path to the registry JSON file.
        metric: "auc" (highest) or "brier" (lowest).

    Returns:
        Best RegistryEntry or None if registry is empty/missing.
    """
    entries = _load_entries(registry_path)
    if not entries:
        return None

    if metric == "auc":
        candidates = [(e, e.auc) for e in entries if e.auc is not None]
        return max(candidates, key=lambda t: t[1])[0] if candidates else None
    if metric == "brier":
        candidates = [(e, e.brier) for e in entries if e.brier is not None]
        return min(candidates, key=lambda t: t[1])[0] if candidates else None
    # Default: by auc
    candidates = [(e, e.auc) for e in entries if e.auc is not None]
    return max(candidates, key=lambda t: t[1])[0] if candidates else None


def list_models(registry_path: Path) -> RegistryResult:
    """List all models in the registry.

    Args:
        registry_path: Path to the registry JSON file.

    Returns:
        RegistryResult (EMPTY flag if file missing or empty).
    """
    entries = _load_entries(registry_path)
    return _build_result(entries)


def format_registry(result: RegistryResult) -> str:
    """Format registry result as a Markdown table.

    Args:
        result: RegistryResult to format.

    Returns:
        Markdown string.
    """
    lines: list[str] = [
        "## Model Registry\n",
        f"Flag: `{result.flag}` | Models: {result.n_models}\n",
    ]

    if result.flag == "EMPTY":
        lines.append("\n_Registry is empty._\n")
        return "\n".join(lines)

    if result.best_by_auc:
        lines.append(f"**Best by AUC**: {result.best_by_auc}\n")

    lines.append("")
    lines.append("| Model ID | Type | AUC | Brier | N Train | Registered At | Notes |")
    lines.append("|----------|------|-----|-------|---------|---------------|-------|")
    for e in result.entries:
        auc_str = f"{e.auc:.4f}" if e.auc is not None else "—"
        brier_str = f"{e.brier:.4f}" if e.brier is not None else "—"
        n_str = str(e.n_train) if e.n_train is not None else "—"
        lines.append(
            f"| {e.model_id} | {e.model_type} | {auc_str} | {brier_str}"
            f" | {n_str} | {e.registered_at} | {e.notes} |"
        )

    return "\n".join(lines)


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Manage the trained model registry.")
    parser.add_argument("registry_path", help="Path to the registry JSON file.")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("list", help="List all registered models.")

    best_p = sub.add_parser("best", help="Show best model by metric.")
    best_p.add_argument("--metric", default="auc", choices=["auc", "brier"])

    args = parser.parse_args(argv)
    path = Path(args.registry_path)

    if args.command == "best":
        entry = get_best(path, metric=args.metric)
        if entry is None:
            print("No models registered.")
            return 1
        print(f"Best ({args.metric}): {entry.model_id}  path={entry.model_path}")
        return 0

    result = list_models(path)
    print(format_registry(result))
    return 0 if result.flag in ("OK", "EMPTY") else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
