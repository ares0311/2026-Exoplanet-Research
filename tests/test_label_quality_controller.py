"""Tests for Skills/label_quality_controller.py (13 tests)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Skills"))

from label_quality_controller import format_qc_result, run_label_qc
from multi_source_label_assembler import LabelManifest, LabelRecord

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_manifest(records):
    n_pos = sum(1 for r in records if r.label == 1)
    n_neg = sum(1 for r in records if r.label == 0)
    return LabelManifest(
        records=tuple(records),
        n_positive=n_pos,
        n_negative=n_neg,
        n_conflicts=0,
        sources=("toi",),
        created_at="2026-01-01T00:00:00+00:00",
        flag="OK",
    )


def _rec(tic_id, label=1, confidence=0.9, period=3.0, epoch=2458000.0,
         conflict=False, source="toi"):
    return LabelRecord(
        tic_id=str(tic_id), label=label, source=source,
        confidence=confidence, period_days=period,
        epoch=epoch, duration_hours=2.0, conflict=conflict,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_all_pass():
    manifest = _make_manifest([_rec(1), _rec(2, label=0)])
    result = run_label_qc(manifest)
    assert result.flag == "OK"
    assert result.n_passed == 2
    assert result.n_rejected == 0


def test_low_confidence_rejected():
    manifest = _make_manifest([_rec(1, confidence=0.3)])
    result = run_label_qc(manifest, min_confidence=0.6)
    assert result.n_rejected == 1
    assert result.rejection_reasons["low_confidence"] == 1


def test_all_rejected_flag():
    manifest = _make_manifest([_rec(1, confidence=0.1)])
    result = run_label_qc(manifest, min_confidence=0.9)
    assert result.flag == "ALL_REJECTED"


def test_conflict_rejected_when_required():
    manifest = _make_manifest([_rec(1, conflict=True)])
    result = run_label_qc(manifest, require_agreement_on_conflict=True)
    assert result.rejection_reasons["conflict"] == 1
    assert result.n_rejected == 1


def test_conflict_allowed_when_not_required():
    manifest = _make_manifest([_rec(1, conflict=True)])
    result = run_label_qc(manifest, require_agreement_on_conflict=False)
    assert result.n_passed == 1


def test_period_out_of_range_low():
    manifest = _make_manifest([_rec(1, period=0.01)])  # below min 0.1
    result = run_label_qc(manifest, min_period_days=0.1)
    assert result.rejection_reasons["period_out_of_range"] == 1


def test_period_out_of_range_high():
    manifest = _make_manifest([_rec(1, period=2000.0)])
    result = run_label_qc(manifest, max_period_days=1000.0)
    assert result.rejection_reasons["period_out_of_range"] == 1


def test_none_period_passes():
    rec = LabelRecord(
        tic_id="1", label=1, source="toi", confidence=0.9,
        period_days=None, epoch=None, duration_hours=None, conflict=False,
    )
    manifest = _make_manifest([rec])
    result = run_label_qc(manifest)
    assert result.n_passed == 1


def test_passed_manifest_counts():
    records = [_rec(i, label=i % 2) for i in range(4)]
    manifest = _make_manifest(records)
    result = run_label_qc(manifest)
    assert result.passed_manifest.n_positive + result.passed_manifest.n_negative == result.n_passed


def test_invalid_manifest():
    result = run_label_qc("not a manifest")
    assert result.flag == "INVALID"


def test_empty_manifest():
    manifest = _make_manifest([])
    result = run_label_qc(manifest)
    assert result.flag == "INVALID"


def test_formatter_contains_flag():
    manifest = _make_manifest([_rec(1), _rec(2, label=0)])
    result = run_label_qc(manifest)
    text = format_qc_result(result)
    assert "OK" in text
    assert len(text) > 0


def test_formatter_on_all_rejected():
    manifest = _make_manifest([_rec(1, confidence=0.1)])
    result = run_label_qc(manifest, min_confidence=0.9)
    text = format_qc_result(result)
    assert "ALL_REJECTED" in text
