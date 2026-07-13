"""Inventory cached unlabeled TESS light curves for Phase 3 without reading FITS data."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from Skills.run_report import RunReport, report_path_for, run_and_commit_report  # noqa: E402

ReportFn = Callable[..., bool]
_DIR_PATTERN = re.compile(r"tess\d+-s(?P<sector>\d+)-(?P<tic>\d+)-")
_DEFAULT_LABELED_PATHS = (
    Path("data/tess_snippets.jsonl"),
    Path("data/tess_snippets_v2.jsonl"),
    Path("data/tess_snippets_expansion.jsonl"),
    Path("data/tess_combined_snippets.jsonl"),
    Path("data/tess_kepler_overlap_snippets.jsonl"),
    Path("data/tess_k2_overlap_snippets.jsonl"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_labeled_tics(paths: Iterable[Path]) -> set[int]:
    """Load the union of TIC IDs from local labeled JSONL corpora."""
    target_ids: set[int] = set()
    for path in paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                value = row.get("tic_id")
                if value is None:
                    raise ValueError(f"{path}:{line_number}: tic_id is required")
                target_ids.add(int(value))
    return target_ids


def load_live_tics(path: Path) -> set[int]:
    """Load TIC IDs frozen in the production live-search batch manifest."""
    if not path.exists():
        return set()
    raw = json.loads(path.read_text(encoding="utf-8"))
    target_ids: set[int] = set()
    for item in raw.get("product_inventory", []):
        value = str(item["target_id"])
        match = re.fullmatch(r"TIC\s+(\d+)", value)
        if match is None:
            raise ValueError(f"invalid live-search target_id {value!r}")
        target_ids.add(int(match.group(1)))
    return target_ids


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def build_inventory(
    cache_root: Path,
    labeled_paths: Iterable[Path],
    live_batch_path: Path,
    rows_path: Path,
    summary_path: Path,
    *,
    report_fn: ReportFn = run_and_commit_report,
) -> dict[str, Any]:
    """Build a cache-only unlabeled product inventory and summary."""
    if not cache_root.is_dir():
        raise ValueError(f"TESS cache root is missing: {cache_root}")
    started_at = datetime.now(UTC)
    started = time.monotonic()
    labeled = load_labeled_tics(labeled_paths)
    live = load_live_tics(live_batch_path)
    excluded = labeled | live
    directories = sorted(path for path in cache_root.iterdir() if path.is_dir())
    if not directories:
        raise ValueError("TESS cache contains no product directories")
    print(
        f"Representation cache inventory startup: directories={len(directories)} "
        f"labeled_tics={len(labeled)} live_tics={len(live)} mode=metadata-only",
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    cached_tics: set[int] = set()
    eligible_tics: set[int] = set()
    excluded_labeled_tics: set[int] = set()
    excluded_live_tics: set[int] = set()
    unparsed_directories = 0
    product_bytes = 0
    for index, directory in enumerate(directories, 1):
        match = _DIR_PATTERN.match(directory.name)
        if match is None:
            unparsed_directories += 1
            continue
        tic_id = int(match.group("tic"))
        sector = int(match.group("sector"))
        cached_tics.add(tic_id)
        if tic_id in labeled:
            excluded_labeled_tics.add(tic_id)
        if tic_id in live:
            excluded_live_tics.add(tic_id)
        if tic_id not in excluded:
            for product in sorted(directory.glob("*.fits")):
                if not (product.name.endswith("_lc.fits") or product.name.endswith("-lc.fits")):
                    continue
                size_bytes = product.stat().st_size
                product_bytes += size_bytes
                eligible_tics.add(tic_id)
                rows.append(
                    {
                        "dataset_id": "tess_cached_unlabeled_representation_v1",
                        "target_id": f"TIC {tic_id}",
                        "group_key": f"tess:tic:{tic_id}",
                        "sector": sector,
                        "product_filename": product.name,
                        "data_uri": f"mast:TESS/product/{product.name}",
                        "cache_relative_path": str(product.relative_to(cache_root)),
                        "size_bytes": size_bytes,
                        "label_status": "unlabeled_relative_to_local_corpora",
                        "role": "training",
                    }
                )
        if index % 1000 == 0 or index == len(directories):
            elapsed = time.monotonic() - started
            rate = index / elapsed if elapsed else 0.0
            eta = (len(directories) - index) / rate if rate else 0.0
            print(
                f"  [{index}/{len(directories)}] products={len(rows)} "
                f"elapsed={elapsed:.1f}s ETA={eta:.1f}s",
                flush=True,
            )
    if not rows:
        raise ValueError("cache contains no eligible unlabeled light-curve products")
    _write_jsonl(rows_path, rows)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "inventory_id": "tess_cached_unlabeled_representation_v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "acquisition_mode": "metadata_only_existing_cache",
        "cache_root": "~/.lightkurve/cache/mastDownload/TESS",
        "cache_product_directories": len(directories),
        "cache_unique_tics": len(cached_tics),
        "eligible_product_count": len(rows),
        "eligible_unique_tics": len(eligible_tics),
        "eligible_bytes": product_bytes,
        "eligible_gb_decimal": product_bytes / 1_000_000_000,
        "excluded_local_labeled_tics": len(excluded_labeled_tics),
        "excluded_live_search_tics": len(excluded_live_tics),
        "unparsed_directories": unparsed_directories,
        "rows_path": str(rows_path),
        "rows_sha256": _sha256(rows_path),
        "label_status": "unlabeled_relative_to_local_corpora_only",
        "role": "training",
        "allowed_use": "self-supervised pretraining with no label-derived objective",
        "forbidden_uses": [
            "supervised target labels",
            "validation, calibration, or frozen evaluation",
            "candidate discovery or external submission",
        ],
        "limitations": [
            "Inventory reflects mutable local cache state at creation time.",
            "Unlabeled means absent from the listed local labeled corpora, not "
            "astrophysically label-free.",
            "No FITS payloads or headers were opened and raw products were not hashed.",
            "No new data were downloaded and no derived light-curve arrays were generated.",
        ],
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    elapsed = time.monotonic() - started
    report = RunReport(
        script="build_representation_cache_inventory",
        status="success",
        started_at=started_at.isoformat(),
        completed_at=datetime.now(UTC).isoformat(),
        elapsed_seconds=elapsed,
        items_processed=len(directories),
        items_written=len(rows),
        output_paths=(str(rows_path), str(summary_path)),
        notes="metadata-only cached TESS inventory; zero downloads and zero FITS reads",
    )
    report_path = report_path_for("build_representation_cache_inventory")
    if not report_fn(report, report_path):
        print(f"WARNING: Run Report push failed for {report_path}", flush=True)
    print(
        f"Representation cache inventory COMPLETE: eligible_tics={len(eligible_tics)} "
        f"products={len(rows)} bytes={product_bytes} elapsed={elapsed:.1f}s",
        flush=True,
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path.home() / ".lightkurve/cache/mastDownload/TESS",
    )
    parser.add_argument(
        "--labeled-path",
        type=Path,
        action="append",
        dest="labeled_paths",
        help="Repeat to override the default local labeled-corpus list.",
    )
    parser.add_argument(
        "--live-batch",
        type=Path,
        default=Path("data_selection/batch_manifests/tess_live_search_v1.json"),
    )
    parser.add_argument(
        "--rows",
        type=Path,
        default=Path("metadata/representation_tess_unlabeled_cache_v1.jsonl"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("artifacts/manifests/representation_cache_inventory_v1.json"),
    )
    args = parser.parse_args(argv)
    build_inventory(
        args.cache_root,
        args.labeled_paths or _DEFAULT_LABELED_PATHS,
        args.live_batch,
        args.rows,
        args.summary,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
