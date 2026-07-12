"""Machine-readable child-process entry point for one CNN prediction."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def main(argv: list[str] | None = None) -> int:
    """Load one checkpoint and emit one JSON probability on stdout."""
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: cnn_inference_worker CHECKPOINT", file=sys.stderr)
        return 2
    payload: dict[str, Any] = json.load(sys.stdin)
    snippet = payload.get("snippet")
    if not isinstance(snippet, list):
        print("snippet must be a JSON list", file=sys.stderr)
        return 2

    from exo_toolkit.ml.cnn_scorer import CnnScorer

    scorer = CnnScorer.from_checkpoint(Path(args[0]))
    if not scorer.is_available:
        print("CNN scorer is unavailable", file=sys.stderr)
        return 3
    probability = scorer.predict_proba([float(value) for value in snippet])
    print(json.dumps({"probability": probability}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
