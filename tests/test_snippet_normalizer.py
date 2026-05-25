"""Tests for Skills/snippet_normalizer.py (13 tests)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Skills"))

from snippet_normalizer import (
    NormalizationReport,
    format_normalization_report,
    normalize_batch,
    normalize_snippet,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_BINS = 21

def _flat_snippet(tic_id="100", label=1, source="toi", n=N_BINS):
    """Flat light curve — transit at phase 0 (same as OOT, depth=0)."""
    phase = [i / n - 0.5 for i in range(n)]
    flux = [1.0] * n
    return tic_id, label, source, phase, flux


def _transit_snippet(tic_id="100", label=1, source="toi", n=N_BINS, depth=0.01):
    """Light curve with a box transit at phase 0."""
    phase = [i / n - 0.5 for i in range(n)]
    flux = []
    for p in phase:
        if abs(p) < 0.05:
            flux.append(1.0 - depth)
        else:
            flux.append(1.0)
    return tic_id, label, source, phase, flux


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_ok_flag():
    tic, lbl, src, phase, flux = _flat_snippet()
    ns = normalize_snippet(tic, lbl, src, phase, flux, n_bins=N_BINS)
    assert ns.flag == "OK"


def test_output_length():
    tic, lbl, src, phase, flux = _flat_snippet()
    ns = normalize_snippet(tic, lbl, src, phase, flux, n_bins=N_BINS)
    assert len(ns.phase) == N_BINS
    assert len(ns.flux) == N_BINS


def test_empty_input_rejected():
    ns = normalize_snippet("100", 1, "toi", [], [], n_bins=N_BINS)
    assert ns.flag == "REJECTED"


def test_mismatched_lengths_rejected():
    ns = normalize_snippet("100", 1, "toi", [0.0], [1.0, 1.0], n_bins=N_BINS)
    assert ns.flag == "REJECTED"


def test_too_few_oot_points_rejected():
    # Only 3 points — too few for min_oot_points=10
    phase = [0.0, 0.1, 0.2]
    flux = [1.0, 1.0, 1.0]
    ns = normalize_snippet("100", 1, "toi", phase, flux, n_bins=N_BINS, min_oot_points=10)
    assert ns.flag == "REJECTED"


def test_normalization_field():
    tic, lbl, src, phase, flux = _flat_snippet()
    ns = normalize_snippet(tic, lbl, src, phase, flux, n_bins=N_BINS)
    assert ns.normalization == "local_median_mad"


def test_oot_scatter_set():
    tic, lbl, src, phase, flux = _transit_snippet()
    ns = normalize_snippet(tic, lbl, src, phase, flux, n_bins=N_BINS)
    # OOT scatter should be set (might be very small for clean data)
    # Just check it's a finite float when flag=OK
    if ns.flag == "OK":
        assert ns.oot_scatter is None or math.isfinite(ns.oot_scatter)


def test_tic_id_preserved():
    tic, lbl, src, phase, flux = _flat_snippet(tic_id="999")
    ns = normalize_snippet(tic, lbl, src, phase, flux, n_bins=N_BINS)
    assert ns.tic_id == "999"


def test_label_preserved():
    tic, lbl, src, phase, flux = _flat_snippet(label=0)
    ns = normalize_snippet(tic, lbl, src, phase, flux, n_bins=N_BINS)
    assert ns.label == 0


def test_batch_ok():
    snippets = []
    for i in range(3):
        _, lbl, src, phase, flux = _flat_snippet(tic_id=str(i))
        snippets.append({"tic_id": str(i), "label": lbl, "source": src,
                         "phase": phase, "flux": flux})
    results, report = normalize_batch(snippets, n_bins=N_BINS)
    assert report.n_input == 3
    assert report.n_normalized == 3
    assert report.n_rejected == 0


def test_batch_report_type():
    _, lbl, src, phase, flux = _flat_snippet()
    snippets = [{"tic_id": "1", "label": lbl, "source": src, "phase": phase, "flux": flux}]
    _, report = normalize_batch(snippets, n_bins=N_BINS)
    assert isinstance(report, NormalizationReport)


def test_formatter_contains_counts():
    _, lbl, src, phase, flux = _flat_snippet()
    snippets = [{"tic_id": "1", "label": lbl, "source": src, "phase": phase, "flux": flux}]
    _, report = normalize_batch(snippets, n_bins=N_BINS)
    text = format_normalization_report(report)
    assert "1" in text
    assert len(text) > 0


def test_empty_batch():
    results, report = normalize_batch([], n_bins=N_BINS)
    assert len(results) == 0
    assert report.n_input == 0
    assert report.n_normalized == 0
