"""Merge CNN predictions into the T1-2 K2 calibration partial predictions.

Reads ``metadata/t1_2_k2_calibration_partial_predictions.jsonl`` (built by
``score_t1_2_k2_calibration.py`` -- ``bayes_prob``/``xgb_prob`` filled in,
``cnn_prob`` left ``null``) and the fetched native K2 snippets JSONL (built
by ``fetch_t1_2_k2_calibration_snippets.py``), scores every row that has a
matching snippet with the promoted CNN checkpoint, and writes a completed
predictions file ready for ``calibrate_stacking_weights.py``.

Deliberate cross-mission note
------------------------------
``models/cnn/benchmark_cnn_v1`` was trained exclusively on Kepler
prime-mission targets (T1-1) and predates the ``training_mission`` config
field (added in version 0.2.27, the same day as the cross-mission scoring
guard in ``run_pipeline()``/``exo scan``). Applying it here to **K2** targets
is a real cross-mission application of that guard's kind -- but this script
does not go through ``run_pipeline()`` and is not subject to that guard,
because it serves a different purpose: T1-2's whole point is to *measure*
real held-out CNN performance on a disjoint calibration set so
``calibrate_stacking_weights.py`` can appropriately down-weight the CNN tier
if it does not transfer well to K2's noisier systematics. This is the
documented "deliberate out-of-domain testing" case the guard's own
``allow_cross_mission_cnn`` escape hatch exists for -- see
``docs/PRODUCTION_READINESS.md`` T1-2 and
``data_selection/data_selection_decision_log.md`` for the recorded decision.
K2 shares Kepler's raw BKJD time convention and instrument (unlike TESS), so
this is a materially smaller domain shift than the Kepler->TESS transfer
attempts (C11-C19) that motivated the guard in the first place -- but it is
still not same-domain, and every output row/summary line says so explicitly
rather than silently treating K2 as Kepler.

Public API
----------
load_snippets_by_epic(snippets_path) -> dict[int, list[float]]
merge_cnn_predictions(partial_predictions_path, snippets_path, *,
                       checkpoint_path, calibration_path, training_mission,
                       target_mission, cnn_scorer) -> list[dict]
write_completed_predictions(rows, output_path) -> None
format_merge_summary(rows, *, training_mission, target_mission) -> str
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

_DEFAULT_CHECKPOINT_PATH = Path("models/cnn/benchmark_cnn_v1/best.pt")
_DEFAULT_CALIBRATION_PATH = Path("models/cnn/benchmark_cnn_v1/calibration.json")
_DEFAULT_TRAINING_MISSION = "Kepler"
_DEFAULT_TARGET_MISSION = "K2"


def load_snippets_by_epic(snippets_path: Path) -> dict[int, list[float]]:
    """Load fetched K2 snippets, keyed by ``epic_id``.

    Args:
        snippets_path: Path to a snippets JSONL file written by
            ``fetch_t1_2_k2_calibration_snippets.py`` (or the concatenation
            of its per-shard output files).

    Returns:
        Mapping of ``epic_id`` to its 201-value normalised flux array. If a
        target appears more than once (should not happen -- one row per
        EPIC in the manifest), the last occurrence wins.
    """
    snippets_path = Path(snippets_path)
    if not snippets_path.exists():
        return {}
    by_epic: dict[int, list[float]] = {}
    for line in snippets_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        by_epic[int(rec["epic_id"])] = list(rec["flux"])
    return by_epic


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def merge_cnn_predictions(
    partial_predictions_path: Path,
    snippets_path: Path,
    *,
    checkpoint_path: Path = _DEFAULT_CHECKPOINT_PATH,
    calibration_path: Path | None = _DEFAULT_CALIBRATION_PATH,
    training_mission: str = _DEFAULT_TRAINING_MISSION,
    target_mission: str = _DEFAULT_TARGET_MISSION,
    cnn_scorer: Any | None = None,
) -> list[dict[str, Any]]:
    """Fill in ``cnn_prob`` for every partial-prediction row with a fetched snippet.

    Args:
        partial_predictions_path: Path to
            ``t1_2_k2_calibration_partial_predictions.jsonl``.
        snippets_path: Path to the fetched K2 snippets JSONL (see
            :func:`load_snippets_by_epic`).
        checkpoint_path: CNN checkpoint to score with. Defaults to the
            promoted ``benchmark_cnn_v1`` artifact.
        calibration_path: Optional temperature-calibration JSON sitting
            alongside the checkpoint. Pass ``None`` to skip calibration.
        training_mission: The checkpoint's real training domain, declared
            explicitly here because ``benchmark_cnn_v1`` predates the
            ``training_mission`` config field (see module docstring).
        target_mission: The mission these snippets were fetched from. Used
            only to annotate the deliberate cross-mission note on each row
            and in the summary; never gates or blocks scoring.
        cnn_scorer: Injectable scorer (for tests). Defaults to
            ``CnnScorer.from_checkpoint(checkpoint_path, ...)``.

    Returns:
        List of dicts: every partial-prediction row's fields, plus
        ``cnn_prob`` (filled in when a snippet exists, else left ``None``),
        ``cnn_training_mission``, and ``cnn_cross_mission`` (``True`` when
        ``training_mission != target_mission``).
    """
    if cnn_scorer is None:
        from exo_toolkit.ml.cnn_scorer import CnnScorer

        cnn_scorer = CnnScorer.from_checkpoint(
            checkpoint_path,
            calibration_path=calibration_path,
            training_mission=training_mission,
        )

    partial_rows = _load_jsonl(Path(partial_predictions_path))
    snippets_by_epic = load_snippets_by_epic(Path(snippets_path))

    cross_mission = training_mission != target_mission
    epic_ids_with_snippets = [
        int(row["epic_id"]) for row in partial_rows if int(row["epic_id"]) in snippets_by_epic
    ]
    flux_batch = [snippets_by_epic[epic_id] for epic_id in epic_ids_with_snippets]
    cnn_probs = (
        cnn_scorer.predict_proba_batch(flux_batch) if flux_batch else []
    )
    cnn_prob_by_epic = dict(zip(epic_ids_with_snippets, cnn_probs, strict=True))

    merged: list[dict[str, Any]] = []
    for row in partial_rows:
        epic_id = int(row["epic_id"])
        out_row = dict(row)
        out_row["cnn_prob"] = cnn_prob_by_epic.get(epic_id)
        out_row["cnn_training_mission"] = training_mission
        out_row["cnn_cross_mission"] = cross_mission
        merged.append(out_row)
    return merged


def filter_complete_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop rows with no ``cnn_prob`` (missing snippet -- fetch failure or not yet run).

    ``calibrate_stacking_weights.py``'s loader now skips ``cnn_prob: null``
    rows defensively rather than crashing, but filtering here too keeps the
    written completed-predictions file itself exact -- no null rows appear
    in the on-disk artifact used for audit/reproducibility, and this
    project's "no silent caps" convention (see ``CLAUDE.md``) requires
    dropped rows to be explicit and counted -- see :func:`format_merge_summary`.
    """
    return [row for row in rows if row.get("cnn_prob") is not None]


def write_completed_predictions(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Write rows as JSONL. Callers should pass :func:`filter_complete_rows`'s
    output when the destination is ``calibrate_stacking_weights.py`` --
    this function itself does not filter, so it can also be used to persist
    the full (including incomplete) merge for auditing."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def format_merge_summary(
    rows: list[dict[str, Any]],
    *,
    training_mission: str = _DEFAULT_TRAINING_MISSION,
    target_mission: str = _DEFAULT_TARGET_MISSION,
) -> str:
    """Render a concise Markdown summary of the CNN merge pass.

    Args:
        rows: The *full* merged row set (before :func:`filter_complete_rows`),
            so the summary can report how many were dropped.
    """
    n = len(rows)
    n_scored = sum(1 for r in rows if r["cnn_prob"] is not None)
    n_missing = n - n_scored
    cross_mission_note = (
        f"**Cross-mission note**: CNN checkpoint trained on {training_mission!r}, "
        f"applied to {target_mission!r} targets -- a deliberate, measured "
        "out-of-domain application for stacking-weight calibration, not a "
        "same-domain claim. See module docstring for rationale.\n"
        if training_mission != target_mission
        else ""
    )
    return (
        "# T1-2 K2 Calibration — CNN Prediction Merge\n\n"
        f"**Rows total**: {n}\n"
        f"**Scored with CNN (snippet found)**: {n_scored}\n"
        f"**Dropped -- missing snippet, cnn_prob would be null**: {n_missing}\n\n"
        f"{cross_mission_note}"
        "## Next Action\n"
        "Run calibrate_stacking_weights.py on this completed predictions "
        "file to produce models/stacking_weights.json. Only the "
        f"{n_scored} fully-scored rows above are written to the output "
        "file, so the on-disk artifact never contains a null cnn_prob.\n"
    )


def _cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Merge CNN predictions into the T1-2 K2 calibration partial "
            "predictions, producing a completed file for stacking-weight "
            "calibration."
        )
    )
    parser.add_argument(
        "--partial-predictions",
        type=Path,
        default=Path("metadata/t1_2_k2_calibration_partial_predictions.jsonl"),
    )
    parser.add_argument(
        "--snippets", type=Path, default=Path("data/t1_2_k2_calibration_snippets.jsonl")
    )
    parser.add_argument("--checkpoint-path", type=Path, default=_DEFAULT_CHECKPOINT_PATH)
    parser.add_argument(
        "--calibration-path", type=Path, default=_DEFAULT_CALIBRATION_PATH
    )
    parser.add_argument("--training-mission", type=str, default=_DEFAULT_TRAINING_MISSION)
    parser.add_argument("--target-mission", type=str, default=_DEFAULT_TARGET_MISSION)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("metadata/t1_2_k2_calibration_completed_predictions.jsonl"),
    )
    args = parser.parse_args(argv)

    rows = merge_cnn_predictions(
        args.partial_predictions,
        args.snippets,
        checkpoint_path=args.checkpoint_path,
        calibration_path=args.calibration_path,
        training_mission=args.training_mission,
        target_mission=args.target_mission,
    )
    complete_rows = filter_complete_rows(rows)
    write_completed_predictions(complete_rows, args.output)
    print(
        format_merge_summary(
            rows, training_mission=args.training_mission, target_mission=args.target_mission
        ),
        flush=True,
    )
    print(f"Saved completed predictions to {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
