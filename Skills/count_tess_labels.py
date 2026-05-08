"""Check the TESS TOI confirmed-planet count against the CNN Tier-2 gate.

Queries ExoFOP-TESS for the current number of CP (confirmed planet)
dispositions and prints whether the 5,000-label threshold for building
the 1D CNN (Tier 2) is met.

Usage
-----
    python Skills/count_tess_labels.py
    python Skills/count_tess_labels.py --threshold 5000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

_CNN_THRESHOLD = 5_000
_EXOFOP_URL = (
    "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv"
)


def count_labels(threshold: int = _CNN_THRESHOLD) -> dict[str, int | bool]:
    """Fetch the TESS TOI table and count CP/FP/EB dispositions.

    Args:
        threshold: Minimum CP count to unlock Tier-2 CNN training.

    Returns:
        Dict with keys: ``cp``, ``fp``, ``eb``, ``total``, ``gate_open``.
    """
    import pandas as pd

    df = pd.read_csv(_EXOFOP_URL, comment="#")
    col = next(
        (c for c in df.columns if "disposition" in c.lower() and "tfop" in c.lower()),
        None,
    )
    if col is None:
        raise ValueError("Could not locate TFOPWG Disposition column in TOI table")

    counts = df[col].value_counts().to_dict()
    cp = int(counts.get("CP", 0))
    fp = int(counts.get("FP", 0))
    eb = int(counts.get("EB", 0))
    total = cp + fp + eb

    return {
        "cp": cp,
        "fp": fp,
        "eb": eb,
        "total": total,
        "gate_open": cp >= threshold,
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--threshold",
        type=int,
        default=_CNN_THRESHOLD,
        help=f"CP count needed to unlock CNN (default: {_CNN_THRESHOLD})",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = count_labels(threshold=args.threshold)
    print("TESS TOI label counts:")
    print(f"  Confirmed planets (CP): {result['cp']:,}")
    print(f"  False positives   (FP): {result['fp']:,}")
    print(f"  Eclipsing binaries(EB): {result['eb']:,}")
    print(f"  Total labeled         : {result['total']:,}")
    print()
    if result["gate_open"]:
        print(f"✓ Gate OPEN — CP count ({result['cp']:,}) ≥ threshold ({args.threshold:,})")
        print("  → Ready to train Tier-2 CNN (see docs/CNN_SPEC.md)")
    else:
        remaining = args.threshold - result["cp"]
        print(f"✗ Gate CLOSED — need {remaining:,} more CP labels")
        print("  → Continue collecting TESS data; re-check with this script")
    sys.exit(0 if result["gate_open"] else 1)
