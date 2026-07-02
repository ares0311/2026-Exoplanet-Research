"""Regression checks for the active dataset handoff contract."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HANDOFF = ROOT / "docs" / "exoplanet_exomoon_dataset_handoff.md"


def test_tap_schema_examples_use_case_insensitive_table_name_matching() -> None:
    """The brief must not reintroduce the live CUMULATIVE metadata bug."""
    text = HANDOFF.read_text(encoding="utf-8")
    assert "UPPER(table_name) = UPPER(" in text
    assert "where table_name = 'cumulative'" not in text
    assert "where table_name = 'toi'" not in text
