"""Recompute EXO-Hunter validity evidence from a database or compressed snapshot."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from exo_toolkit.hunter_validity import validate_hunter_database  # noqa: E402
from exo_toolkit.search_lifecycle import HUNTER_SCHEMA_VERSION  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recompute Hunter relational, validity, and provenance evidence."
    )
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument(
        "--history-manifest",
        type=Path,
        default=REPO_ROOT / "data_selection" / "hunter_prior_search_history_v1.json",
    )
    parser.add_argument(
        "--history-source-root",
        type=Path,
        default=None,
        help=(
            "Resolve --history-manifest's source_path entries relative to this "
            "directory instead of the repo-root walk-up heuristic. Required for "
            "reliable isolated/scripted operation with a manifest that does not "
            "live under this repo's own tree."
        ),
    )
    return parser


def _validate(
    path: Path, history_manifest: Path, history_source_root: Path | None
) -> dict[str, object]:
    if path.suffix != ".gz":
        return validate_hunter_database(
            path,
            schema_version=HUNTER_SCHEMA_VERSION,
            history_manifest=history_manifest,
            history_source_root=history_source_root,
        )
    compressed_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    with tempfile.NamedTemporaryFile(suffix=".sqlite3") as temporary:
        with gzip.open(path, "rb") as source:
            shutil.copyfileobj(source, temporary)
        temporary.flush()
        result = validate_hunter_database(
            Path(temporary.name),
            schema_version=HUNTER_SCHEMA_VERSION,
            history_manifest=history_manifest,
            history_source_root=history_source_root,
        )
    result["compressed_snapshot_path"] = str(path)
    result["compressed_snapshot_sha256"] = compressed_sha256
    result.pop("database_path", None)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = _validate(args.db, args.history_manifest, args.history_source_root)
    except Exception as exc:  # noqa: BLE001
        print(
            json.dumps(
                {"ok": False, "issues": [f"{type(exc).__name__}: {exc}"]},
                indent=2,
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
