"""Tests for Skills/multi_source_label_assembler.py (13 tests)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Skills"))

from multi_source_label_assembler import (
    assemble_labels,
    format_label_manifest,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _row(tic_id, label, source="toi", confidence=0.9, **kwargs):
    return dict(tic_id=str(tic_id), label=label, source=source,
                confidence=confidence, **kwargs)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_ok_basic():
    rows = [_row("100", 1), _row("200", 0)]
    manifest = assemble_labels(rows)
    assert manifest.flag == "OK"
    assert manifest.n_positive == 1
    assert manifest.n_negative == 1


def test_empty_input_flag():
    manifest = assemble_labels([])
    assert manifest.flag == "EMPTY"
    assert len(manifest.records) == 0


def test_invalid_rows_all_bad():
    rows = [{"tic_id": "100", "source": "toi"}]  # missing label
    manifest = assemble_labels(rows)
    assert manifest.flag == "INVALID"


def test_deduplication_same_label():
    rows = [
        _row("100", 1, source="toi", confidence=0.8),
        _row("100", 1, source="ctoi", confidence=0.95),
    ]
    manifest = assemble_labels(rows)
    assert len(manifest.records) == 1
    # Keeps highest confidence
    assert manifest.records[0].confidence == pytest.approx(0.95)


def test_conflict_detection():
    rows = [
        _row("100", 1, source="toi", confidence=0.8),
        _row("100", 0, source="ctoi", confidence=0.7),
    ]
    manifest = assemble_labels(rows)
    assert manifest.n_conflicts == 1
    assert manifest.records[0].conflict is True


def test_conflict_policy_higher_confidence():
    rows = [
        _row("100", 1, source="toi", confidence=0.9),
        _row("100", 0, source="ctoi", confidence=0.6),
    ]
    manifest = assemble_labels(rows, conflict_policy="higher_confidence")
    assert manifest.records[0].label == 1


def test_conflict_policy_conservative():
    rows = [
        _row("100", 1, source="toi", confidence=0.9),
        _row("100", 0, source="ctoi", confidence=0.6),
    ]
    manifest = assemble_labels(rows, conflict_policy="conservative")
    # conservative prefers FP (label=0)
    assert manifest.records[0].label == 0


def test_max_fp_ratio_trimming():
    rows = [_row("1", 1)]
    for i in range(10):
        rows.append(_row(str(100 + i), 0, confidence=0.5 + i * 0.01))
    manifest = assemble_labels(rows, max_fp_ratio=2.0)
    # 1 positive → max 2 negatives
    assert manifest.n_negative <= 2
    assert manifest.n_positive == 1


def test_sources_recorded():
    rows = [_row("100", 1, source="toi"), _row("200", 0, source="koi")]
    manifest = assemble_labels(rows)
    assert "toi" in manifest.sources
    assert "koi" in manifest.sources


def test_optional_fields_preserved():
    rows = [_row("100", 1, period_days=3.5, epoch=2458000.0, duration_hours=2.1)]
    manifest = assemble_labels(rows)
    rec = manifest.records[0]
    assert rec.period_days == pytest.approx(3.5)
    assert rec.epoch == pytest.approx(2458000.0)
    assert rec.duration_hours == pytest.approx(2.1)


def test_output_path_writes_file(tmp_path):
    rows = [_row("100", 1), _row("200", 0)]
    out = tmp_path / "manifest.json"
    manifest = assemble_labels(rows, output_path=out)
    assert out.exists()
    assert manifest.flag == "OK"


def test_formatter_contains_flag():
    rows = [_row("100", 1), _row("200", 0)]
    manifest = assemble_labels(rows)
    text = format_label_manifest(manifest)
    assert "OK" in text
    assert len(text) > 0


def test_conflict_policy_majority():
    # 2 say planet, 1 says FP → majority = planet
    rows = [
        _row("100", 1, source="toi", confidence=0.8),
        _row("100", 1, source="ctoi", confidence=0.7),
        _row("100", 0, source="koi", confidence=0.9),
    ]
    manifest = assemble_labels(rows, conflict_policy="majority")
    assert manifest.records[0].label == 1
