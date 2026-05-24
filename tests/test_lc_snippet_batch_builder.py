"""Tests for Skills/lc_snippet_batch_builder.py (13 tests)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Skills"))

from lc_snippet_batch_builder import (
    SnippetEntry,
    build_snippet_batch,
    format_batch_result,
    load_batch_output,
    save_batch_output,
)
from multi_source_label_assembler import LabelManifest, LabelRecord

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_BINS = 5  # small for tests

def _make_manifest(tic_ids, labels=None):
    if labels is None:
        labels = [1] * len(tic_ids)
    records = tuple(
        LabelRecord(
            tic_id=str(tid), label=lbl, source="toi",
            confidence=0.9, period_days=3.0, epoch=2458000.0,
            duration_hours=2.0, conflict=False,
        )
        for tid, lbl in zip(tic_ids, labels, strict=False)
    )
    return LabelManifest(
        records=records,
        n_positive=sum(1 for lbl in labels if lbl == 1),
        n_negative=sum(1 for lbl in labels if lbl == 0),
        n_conflicts=0, sources=("toi",), created_at="2026-01-01T00:00:00+00:00",
        flag="OK",
    )


def _good_snippet_fn(tic_id, period, epoch, duration):
    phase = [i / N_BINS - 0.5 for i in range(N_BINS)]
    flux = [1.0] * N_BINS
    return phase, flux


def _fail_snippet_fn(tic_id, period, epoch, duration):
    return None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_ok_basic(tmp_path):
    manifest = _make_manifest([100, 200])
    result = build_snippet_batch(
        manifest, snippet_fn=_good_snippet_fn,
        output_path=tmp_path / "out.json",
        checkpoint_path=tmp_path / "ckpt.json",
        n_bins=N_BINS, resume=False,
    )
    assert result.flag == "OK"
    assert result.n_succeeded == 2
    assert result.n_failed == 0


def test_n_snippets(tmp_path):
    manifest = _make_manifest([100, 200, 300])
    result = build_snippet_batch(
        manifest, snippet_fn=_good_snippet_fn,
        output_path=tmp_path / "out.json",
        checkpoint_path=tmp_path / "ckpt.json",
        n_bins=N_BINS, resume=False,
    )
    assert result.n_snippets == 3


def test_all_fail_empty_flag(tmp_path):
    manifest = _make_manifest([100, 200])
    result = build_snippet_batch(
        manifest, snippet_fn=_fail_snippet_fn,
        output_path=tmp_path / "out.json",
        checkpoint_path=tmp_path / "ckpt.json",
        n_bins=N_BINS, resume=False,
    )
    assert result.flag == "EMPTY"
    assert result.n_failed == 2


def test_partial_flag(tmp_path):
    calls = [0]
    def _mixed_fn(tic_id, period, epoch, duration):
        calls[0] += 1
        if calls[0] % 2 == 0:
            return None
        return [i / N_BINS - 0.5 for i in range(N_BINS)], [1.0] * N_BINS

    manifest = _make_manifest([100, 200, 300, 400])
    result = build_snippet_batch(
        manifest, snippet_fn=_mixed_fn,
        output_path=tmp_path / "out.json",
        checkpoint_path=tmp_path / "ckpt.json",
        n_bins=N_BINS, resume=False,
    )
    assert result.flag == "PARTIAL"


def test_invalid_manifest(tmp_path):
    result = build_snippet_batch(
        "not a manifest", snippet_fn=_good_snippet_fn,
        output_path=tmp_path / "out.json",
        checkpoint_path=tmp_path / "ckpt.json",
        n_bins=N_BINS, resume=False,
    )
    assert result.flag == "INVALID"


def test_checkpoint_written(tmp_path):
    manifest = _make_manifest([100])
    ckpt = tmp_path / "ckpt.json"
    build_snippet_batch(
        manifest, snippet_fn=_good_snippet_fn,
        output_path=tmp_path / "out.json",
        checkpoint_path=ckpt,
        n_bins=N_BINS, resume=False,
    )
    assert ckpt.exists()
    data = json.loads(ckpt.read_text())
    assert "100" in data["completed_tic_ids"]


def test_resume_skips_completed(tmp_path):
    manifest = _make_manifest([100, 200])
    out = tmp_path / "out.json"
    ckpt = tmp_path / "ckpt.json"
    # First run
    build_snippet_batch(
        manifest, snippet_fn=_good_snippet_fn,
        output_path=out, checkpoint_path=ckpt,
        n_bins=N_BINS, resume=False,
    )
    # Second run with resume — nothing new to do
    result2 = build_snippet_batch(
        manifest, snippet_fn=_fail_snippet_fn,  # would fail if called
        output_path=out, checkpoint_path=ckpt,
        n_bins=N_BINS, resume=True,
    )
    assert result2.n_attempted == 0


def test_label_counts(tmp_path):
    manifest = _make_manifest([100, 200, 300], labels=[1, 0, 1])
    result = build_snippet_batch(
        manifest, snippet_fn=_good_snippet_fn,
        output_path=tmp_path / "out.json",
        checkpoint_path=tmp_path / "ckpt.json",
        n_bins=N_BINS, resume=False,
    )
    assert result.label_counts.get(1, 0) == 2
    assert result.label_counts.get(0, 0) == 1


def test_save_and_load_roundtrip(tmp_path):
    entries = [
        SnippetEntry(tic_id="100", label=1, source="toi",
                     phase=tuple(range(N_BINS)), flux=(1.0,) * N_BINS,
                     period_days=3.0, snr=10.0),
    ]
    path = tmp_path / "snippets.json"
    save_batch_output(entries, path)
    loaded = load_batch_output(path)
    assert len(loaded) == 1
    assert loaded[0].tic_id == "100"
    assert loaded[0].label == 1


def test_output_file_written(tmp_path):
    manifest = _make_manifest([100])
    out = tmp_path / "out.json"
    build_snippet_batch(
        manifest, snippet_fn=_good_snippet_fn,
        output_path=out, checkpoint_path=tmp_path / "ckpt.json",
        n_bins=N_BINS, resume=False,
    )
    assert out.exists()


def test_wrong_bin_count_triggers_failure(tmp_path):
    def _wrong_bins(tic_id, period, epoch, duration):
        return [0.0, 0.5], [1.0, 1.0]  # only 2 bins, not N_BINS=5

    manifest = _make_manifest([100])
    result = build_snippet_batch(
        manifest, snippet_fn=_wrong_bins,
        output_path=tmp_path / "out.json",
        checkpoint_path=tmp_path / "ckpt.json",
        n_bins=N_BINS, resume=False,
    )
    assert result.n_failed == 1


def test_formatter_contains_flag(tmp_path):
    manifest = _make_manifest([100])
    result = build_snippet_batch(
        manifest, snippet_fn=_good_snippet_fn,
        output_path=tmp_path / "out.json",
        checkpoint_path=tmp_path / "ckpt.json",
        n_bins=N_BINS, resume=False,
    )
    text = format_batch_result(result)
    assert "OK" in text
    assert len(text) > 0


def test_exception_in_snippet_fn_counts_as_failure(tmp_path):
    def _explode(tic_id, period, epoch, duration):
        raise RuntimeError("boom")

    manifest = _make_manifest([100])
    result = build_snippet_batch(
        manifest, snippet_fn=_explode,
        output_path=tmp_path / "out.json",
        checkpoint_path=tmp_path / "ckpt.json",
        n_bins=N_BINS, resume=False,
    )
    assert result.n_failed == 1
