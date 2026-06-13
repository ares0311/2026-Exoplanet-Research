"""Identify TESS TIC IDs with confirmed labels not yet in the local corpus.

Compares the current local TESS snippet corpus (``data/tess_snippets.jsonl``)
against ExoFOP TOI and CTOI disposition tables to find labeled TIC IDs not
yet downloaded.  Writes a target list that can be fed to the TESS light curve
downloader.

Output is a plain-text file of TIC IDs (one per line) suitable for:

    python Skills/lc_snippet_batch_builder.py --tic-list <output_file>

Public API
----------
load_corpus_tic_ids(jsonl_path) -> set[int]
fetch_toi_labels(toi_table_fn) -> list[dict]
fetch_ctoi_labels(ctoi_table_fn) -> list[dict]
find_new_tic_ids(corpus_ids, labeled_rows, *, positive_only) -> list[dict]
write_target_list(rows, output_path) -> int
format_expansion_summary(corpus_ids, new_rows) -> str
"""
from __future__ import annotations

import csv
import io
import json
import time
from collections.abc import Callable
from pathlib import Path
from urllib.request import urlopen

_TOI_CSV_URL = (
    "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv"
)
_CTOI_CSV_URL = (
    "https://exofop.ipac.caltech.edu/tess/download_ctoi.php?sort=ctoi&output=csv"
)

# ExoFOP disposition values that yield confirmed labels
_POSITIVE_DISPOSITIONS = {"CP", "KP"}   # Confirmed Planet, Known Planet
_NEGATIVE_DISPOSITIONS = {"FP", "FA"}   # False Positive, False Alarm


# ---------------------------------------------------------------------------
# Corpus reader
# ---------------------------------------------------------------------------


def load_corpus_tic_ids(jsonl_path: Path) -> set[int]:
    """Return the set of TIC IDs already present in the snippet corpus.

    Args:
        jsonl_path: Path to the JSONL snippet file.

    Returns:
        Set of integer TIC IDs.
    """
    tic_ids: set[int] = set()
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists():
        return tic_ids
    with jsonl_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                tid = row.get("tic_id")
                if tid is not None:
                    tic_ids.add(int(tid))
            except (json.JSONDecodeError, ValueError):
                pass
    return tic_ids


# ---------------------------------------------------------------------------
# ExoFOP fetchers
# ---------------------------------------------------------------------------


def _fetch_csv(url: str, timeout: int = 60) -> list[dict]:
    """Fetch a CSV from a URL and return as list of dicts."""
    with urlopen(url, timeout=timeout) as resp:  # noqa: S310
        content = resp.read().decode("utf-8")
    reader = csv.DictReader(io.StringIO(content))
    return list(reader)


def _default_toi_table_fn() -> list[dict]:
    return _fetch_csv(_TOI_CSV_URL)


def _default_ctoi_table_fn() -> list[dict]:
    return _fetch_csv(_CTOI_CSV_URL)


def _extract_tic_and_disposition(row: dict) -> tuple[int | None, str, float, float]:
    """Extract TIC ID, disposition, period, epoch from a TOI/CTOI row."""
    tic_id: int | None = None
    disposition = ""
    period = 0.0
    epoch = 0.0

    for key in ("TIC ID", "tic_id", "TIC", "ticid"):
        if key in row and row[key]:
            try:
                tic_id = int(float(row[key]))
                break
            except ValueError:
                pass

    for key in ("TFOPWG Disposition", "tfopwg_disposition", "Disposition", "User Disposition"):
        if key in row and row[key]:
            disposition = str(row[key]).strip().upper()
            break

    for key in ("Period (days)", "Period", "period", "period_days"):
        if key in row and row[key]:
            try:
                period = float(row[key])
                break
            except ValueError:
                pass

    for key in ("Epoch (BJD)", "epoch_bjd", "Epoch", "epoch"):
        if key in row and row[key]:
            try:
                epoch = float(row[key])
                break
            except ValueError:
                pass

    return tic_id, disposition, period, epoch


def fetch_toi_labels(toi_table_fn: Callable | None = None) -> list[dict]:
    """Fetch ExoFOP TOI table and return labeled rows.

    Args:
        toi_table_fn: Injectable function returning list of CSV dicts.

    Returns:
        List of dicts with keys: tic_id, label, disposition, period_days, epoch_bjd, source.
    """
    fn = toi_table_fn or _default_toi_table_fn
    rows = fn()
    labeled: list[dict] = []
    for row in rows:
        tic_id, disposition, period, epoch = _extract_tic_and_disposition(row)
        if tic_id is None or tic_id <= 0:
            continue
        if disposition in _POSITIVE_DISPOSITIONS:
            label = 1
        elif disposition in _NEGATIVE_DISPOSITIONS:
            label = 0
        else:
            continue
        labeled.append({
            "tic_id": tic_id,
            "label": label,
            "disposition": disposition,
            "period_days": period,
            "epoch_bjd": epoch,
            "source": "exofop_toi",
        })
    return labeled


def fetch_ctoi_labels(ctoi_table_fn: Callable | None = None) -> list[dict]:
    """Fetch ExoFOP CTOI table and return labeled rows.

    Args:
        ctoi_table_fn: Injectable function returning list of CSV dicts.

    Returns:
        List of dicts with keys: tic_id, label, disposition, period_days, epoch_bjd, source.
    """
    fn = ctoi_table_fn or _default_ctoi_table_fn
    rows = fn()
    labeled: list[dict] = []
    for row in rows:
        tic_id, disposition, period, epoch = _extract_tic_and_disposition(row)
        if tic_id is None or tic_id <= 0:
            continue
        if disposition in _POSITIVE_DISPOSITIONS:
            label = 1
        elif disposition in _NEGATIVE_DISPOSITIONS:
            label = 0
        else:
            continue
        labeled.append({
            "tic_id": tic_id,
            "label": label,
            "disposition": disposition,
            "period_days": period,
            "epoch_bjd": epoch,
            "source": "exofop_ctoi",
        })
    return labeled


# ---------------------------------------------------------------------------
# Gap finder
# ---------------------------------------------------------------------------


def find_new_tic_ids(
    corpus_ids: set[int],
    labeled_rows: list[dict],
    *,
    positive_only: bool = False,
) -> list[dict]:
    """Return labeled rows whose TIC IDs are not in corpus_ids.

    Deduplicates by TIC ID (keeps first occurrence).

    Args:
        corpus_ids: Set of TIC IDs already in the corpus.
        labeled_rows: Rows from fetch_toi_labels / fetch_ctoi_labels.
        positive_only: If True, return only positive (label=1) rows.

    Returns:
        List of new labeled rows, deduplicated by TIC ID.
    """
    seen: set[int] = set()
    new_rows: list[dict] = []
    for row in labeled_rows:
        tic_id = int(row["tic_id"])
        if tic_id in corpus_ids:
            continue
        if tic_id in seen:
            continue
        if positive_only and row["label"] != 1:
            continue
        seen.add(tic_id)
        new_rows.append(row)
    return new_rows


# ---------------------------------------------------------------------------
# Writers and formatters
# ---------------------------------------------------------------------------


def write_target_list(rows: list[dict], output_path: Path) -> int:
    """Write new TIC IDs to a plain-text file (one per line).

    Also writes a companion JSON file with full metadata at
    ``output_path.with_suffix('.json')``.

    Args:
        rows: New labeled rows from :func:`find_new_tic_ids`.
        output_path: Destination text file path.

    Returns:
        Number of TIC IDs written.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(f"{row['tic_id']}\n")

    meta_path = output_path.with_suffix(".json")
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)
        fh.write("\n")

    return len(rows)


def format_expansion_summary(
    corpus_ids: set[int],
    new_rows: list[dict],
) -> str:
    """Format a Markdown summary of the corpus expansion opportunity.

    Args:
        corpus_ids: Current corpus TIC ID set.
        new_rows: New labeled rows found.

    Returns:
        Markdown string.
    """
    n_pos = sum(1 for r in new_rows if r["label"] == 1)
    n_neg = sum(1 for r in new_rows if r["label"] == 0)
    sources: dict[str, int] = {}
    for r in new_rows:
        sources[r["source"]] = sources.get(r["source"], 0) + 1

    lines = [
        "## TESS Corpus Expansion Opportunity",
        "",
        f"- Existing corpus TIC IDs: {len(corpus_ids)}",
        f"- New labeled TIC IDs found: {len(new_rows)}",
        f"  - Positive (planet/candidate): {n_pos}",
        f"  - Negative (FP/FA): {n_neg}",
        "",
        "### New IDs by source",
    ]
    for src, count in sorted(sources.items()):
        lines.append(f"- {src}: {count}")
    lines += [
        "",
        "### Next step",
        "",
        "Download TESS light curves for the new TIC IDs and extract snippets:",
        "```bash",
        "git pull origin main",
        "caffeinate -dims python Skills/lc_snippet_batch_builder.py \\",
        "    --tic-list data/new_tess_targets.txt \\",
        "    --output data/tess_snippets_expansion.jsonl",
        "```",
        "",
        "Then merge with the existing corpus and rebuild splits:",
        "```bash",
        "cat data/tess_snippets.jsonl data/tess_snippets_expansion.jsonl \\",
        "    > data/tess_snippets_v2.jsonl",
        "python Skills/build_cnn_training_data.py data/tess_snippets_v2.jsonl \\",
        "    --output-dir data/cnn_splits_v2",
        "```",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="fetch_additional_tess_labels",
        description="Find labeled TESS TIC IDs not yet in the local snippet corpus.",
    )
    parser.add_argument(
        "--corpus", type=Path, default=Path("data/tess_snippets.jsonl"),
        metavar="JSONL",
        help="Existing snippet corpus (default: data/tess_snippets.jsonl)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/new_tess_targets.txt"),
        metavar="TXT",
        help="Output TIC ID list (default: data/new_tess_targets.txt)",
    )
    parser.add_argument(
        "--positive-only", action="store_true",
        help="Only include confirmed planet TIC IDs (skip FP/FA)",
    )
    args = parser.parse_args(argv)

    print(f"Loading existing corpus from {args.corpus} ...", flush=True)
    corpus_ids = load_corpus_tic_ids(args.corpus)
    print(f"  {len(corpus_ids)} TIC IDs already in corpus.", flush=True)

    print("Fetching ExoFOP TOI table ...", flush=True)
    t0 = time.monotonic()
    try:
        toi_rows = fetch_toi_labels()
        print(f"  TOI: {len(toi_rows)} labeled rows ({time.monotonic()-t0:.1f}s)", flush=True)
    except Exception as exc:
        print(f"  Warning: TOI fetch failed: {exc}")
        toi_rows = []

    print("Fetching ExoFOP CTOI table ...", flush=True)
    t1 = time.monotonic()
    try:
        ctoi_rows = fetch_ctoi_labels()
        print(f"  CTOI: {len(ctoi_rows)} labeled rows ({time.monotonic()-t1:.1f}s)", flush=True)
    except Exception as exc:
        print(f"  Warning: CTOI fetch failed: {exc}")
        ctoi_rows = []

    all_rows = toi_rows + ctoi_rows
    new_rows = find_new_tic_ids(corpus_ids, all_rows, positive_only=args.positive_only)

    n = write_target_list(new_rows, args.output)
    print(format_expansion_summary(corpus_ids, new_rows))
    print(f"Flag: OK  new_tic_ids={n}  output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
