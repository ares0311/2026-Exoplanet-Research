"""Known TESS examples used for deterministic background automation bootstrap."""

from __future__ import annotations

import json
from pathlib import Path

from exo_toolkit.background.schemas import KnownTessTarget

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "known_tess_examples.json"


def load_known_tess_examples(path: Path = FIXTURE_PATH) -> list[KnownTessTarget]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [KnownTessTarget(**item) for item in payload["targets"]]


def fixture_summary(path: Path = FIXTURE_PATH) -> list[dict[str, object]]:
    targets = load_known_tess_examples(path)
    return [
        {
            "target_id": target.target_id,
            "target_name": target.target_name,
            "mission": target.mission,
            "fixture_version": target.fixture_version,
            "known_object": target.known_object,
            "synthetic": any(label.startswith("synthetic") for label in target.fixture_labels),
            "fixture_labels": target.fixture_labels,
            "risk_flags": _risk_flags(target),
        }
        for target in targets
    ]


def _risk_flags(target: KnownTessTarget) -> list[str]:
    flags: list[str] = []
    if target.snr < 5:
        flags.append("weak_signal")
    if target.false_positive_risk_score > 0.7:
        flags.append("high_false_positive_risk")
    if not target.provenance:
        flags.append("missing_provenance")
    if target.calibration_confidence_score < 0.6:
        flags.append("calibration_uncertain")
    if target.blocking_issue_penalty >= 0.4:
        flags.append("blocking_issue")
    return flags
