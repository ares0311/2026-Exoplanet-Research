"""Build joint Kepler+TESS training splits for C17.

Merges Kepler train examples with TESS combined train examples to create
a larger joint training set that addresses the data-ceiling bottleneck
identified after C13–C16.  Val and test remain TESS-only so the
production gate is evaluated on in-domain data.

Output layout
-------------
data/joint_cnn_splits/
  train.json     -- Kepler train + TESS combined train (merged, shuffled)
  val.json       -- TESS combined val (unchanged copy)
  test.json      -- TESS combined test (unchanged copy)
  manifest.json  -- counts, provenance, and label balance

Public API
----------
build_joint_splits(tess_split_dir, kepler_split_dir, output_dir, *,
                   seed, max_kepler) -> dict
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path


def build_joint_splits(
    tess_split_dir: Path,
    kepler_split_dir: Path,
    output_dir: Path,
    *,
    seed: int = 7,
    max_kepler: int | None = None,
) -> dict:
    """Merge Kepler and TESS training splits; copy TESS val and test unchanged.

    Args:
        tess_split_dir: Directory containing TESS combined split JSON files.
        kepler_split_dir: Directory containing Kepler split JSON files.
        output_dir: Destination directory for joint split files.
        seed: RNG seed used for shuffling the joint training set.
        max_kepler: Optional cap on Kepler training examples (None = use all).

    Returns:
        Manifest dict with counts and provenance.
    """
    start = time.monotonic()

    def _load(path: Path) -> list[dict]:
        payload = json.loads(path.read_text())
        if isinstance(payload, dict) and "examples" in payload:
            return list(payload["examples"])
        if isinstance(payload, list):
            return list(payload)
        raise ValueError(f"Unrecognised split format in {path}")

    def _label_counts(examples: list[dict]) -> tuple[int, int]:
        pos = sum(1 for e in examples if e.get("label") == 1)
        return pos, len(examples) - pos

    print(f"Loading TESS train from {tess_split_dir}/train.json …", flush=True)
    tess_train = _load(tess_split_dir / "train.json")
    tess_val = _load(tess_split_dir / "val.json")
    tess_test = _load(tess_split_dir / "test.json")

    print(f"Loading Kepler train from {kepler_split_dir}/train.json …", flush=True)
    kepler_train = _load(kepler_split_dir / "train.json")

    if max_kepler is not None and len(kepler_train) > max_kepler:
        rng = random.Random(seed)
        kepler_train = rng.sample(kepler_train, max_kepler)
        print(f"  Capped Kepler train to {max_kepler} examples (seed={seed})", flush=True)

    print(
        f"  TESS train: {len(tess_train)}  Kepler train: {len(kepler_train)}", flush=True
    )

    joint_train = tess_train + kepler_train
    random.Random(seed).shuffle(joint_train)

    print(f"Joint train: {len(joint_train)} examples — writing splits …", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    def _write_split(name: str, examples: list[dict]) -> None:
        payload = {"split": name, "examples": examples}
        (output_dir / f"{name}.json").write_text(json.dumps(payload))

    _write_split("train", joint_train)
    _write_split("val", tess_val)
    _write_split("test", tess_test)

    tess_pos, tess_neg = _label_counts(tess_train)
    kepler_pos, kepler_neg = _label_counts(kepler_train)
    joint_pos, joint_neg = _label_counts(joint_train)
    val_pos, val_neg = _label_counts(tess_val)
    test_pos, test_neg = _label_counts(tess_test)

    manifest = {
        "tess_split_dir": str(tess_split_dir),
        "kepler_split_dir": str(kepler_split_dir),
        "output_dir": str(output_dir),
        "seed": seed,
        "tess_train_n": len(tess_train),
        "tess_train_positive": tess_pos,
        "tess_train_negative": tess_neg,
        "kepler_train_n": len(kepler_train),
        "kepler_train_positive": kepler_pos,
        "kepler_train_negative": kepler_neg,
        "joint_train_n": len(joint_train),
        "joint_train_positive": joint_pos,
        "joint_train_negative": joint_neg,
        "val_n": len(tess_val),
        "val_positive": val_pos,
        "val_negative": val_neg,
        "test_n": len(tess_test),
        "test_positive": test_pos,
        "test_negative": test_neg,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    elapsed = time.monotonic() - start
    print(
        f"Done in {elapsed:.1f}s — joint train={len(joint_train)} "
        f"(pos={joint_pos} neg={joint_neg}) "
        f"val={len(tess_val)} test={len(tess_test)}",
        flush=True,
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build joint Kepler+TESS CNN training splits for C17."
    )
    parser.add_argument(
        "--tess-split-dir",
        type=Path,
        default=Path("data/tess_combined_cnn_splits"),
        help="TESS combined split directory (default: data/tess_combined_cnn_splits)",
    )
    parser.add_argument(
        "--kepler-split-dir",
        type=Path,
        default=Path("data/kepler_cnn_splits"),
        help="Kepler split directory (default: data/kepler_cnn_splits)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/joint_cnn_splits"),
        help="Output split directory (default: data/joint_cnn_splits)",
    )
    parser.add_argument("--seed", type=int, default=7, help="Shuffle seed (default: 7)")
    parser.add_argument(
        "--max-kepler",
        type=int,
        default=None,
        metavar="N",
        help="Cap Kepler train examples at N (default: use all)",
    )
    args = parser.parse_args(argv)

    for d in (args.tess_split_dir, args.kepler_split_dir):
        if not d.is_dir():
            print(f"Error: directory not found: {d}", file=sys.stderr)
            return 1

    manifest = build_joint_splits(
        args.tess_split_dir,
        args.kepler_split_dir,
        args.output_dir,
        seed=args.seed,
        max_kepler=args.max_kepler,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
