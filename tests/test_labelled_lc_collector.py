"""Tests for Skills.labelled_lc_collector."""
from __future__ import annotations

import json
from pathlib import Path

from Skills.labelled_lc_collector import (
    LabelledDataset,
    LabelledSnippet,
    build_dataset,
    extract_snippet,
    format_dataset_summary,
)


def _make_row(tic_id=1, label=1, depth=0.01, n=500):
    epoch = 2458000.0
    period = 5.0
    dt = 2.0 / 1440.0
    time = [epoch - 5.0 + i * dt for i in range(n)]
    flux = []
    for t in time:
        ph = (t - epoch) % period
        if ph > period / 2:
            ph -= period
        flux.append(1.0 - depth if abs(ph) < 0.05 else 1.0)
    return {
        "tic_id": tic_id,
        "label": label,
        "period_days": period,
        "epoch_bjd": epoch,
        "time": time,
        "flux": flux,
        "source": "tess",
    }


class TestExtractSnippet:
    def test_returns_snippet(self) -> None:
        row = _make_row()
        s = extract_snippet(row["time"], row["flux"], row["period_days"], row["epoch_bjd"])
        assert isinstance(s, LabelledSnippet)

    def test_n_bins_respected(self) -> None:
        row = _make_row()
        s = extract_snippet(
            row["time"], row["flux"], row["period_days"], row["epoch_bjd"], n_bins=51
        )
        assert s is not None
        assert len(s.flux) == 51

    def test_phase_range(self) -> None:
        row = _make_row()
        s = extract_snippet(row["time"], row["flux"], row["period_days"], row["epoch_bjd"])
        assert s is not None
        assert all(-0.5 <= p < 0.5 for p in s.phase)

    def test_label_stored(self) -> None:
        row = _make_row(label=0)
        s = extract_snippet(row["time"], row["flux"], row["period_days"], row["epoch_bjd"], label=0)
        assert s is not None
        assert s.label == 0

    def test_insufficient_data_returns_none(self) -> None:
        s = extract_snippet([2458000.0, 2458001.0], [1.0, 1.0], 5.0, 2458000.0, n_bins=201)
        assert s is None

    def test_zero_period_returns_none(self) -> None:
        row = _make_row()
        s = extract_snippet(row["time"], row["flux"], 0.0, row["epoch_bjd"])
        assert s is None


class TestBuildDataset:
    def test_returns_dataset(self) -> None:
        rows = [_make_row(tic_id=i) for i in range(3)]
        d = build_dataset(rows)
        assert isinstance(d, LabelledDataset)

    def test_n_snippets_correct(self) -> None:
        rows = [_make_row(tic_id=i) for i in range(3)]
        d = build_dataset(rows)
        assert len(d.snippets) == 3

    def test_label_counts(self) -> None:
        rows = [_make_row(tic_id=i, label=i % 2) for i in range(4)]
        d = build_dataset(rows)
        assert d.label_counts.get(0, 0) + d.label_counts.get(1, 0) == 4

    def test_empty_rows(self) -> None:
        d = build_dataset([])
        assert len(d.snippets) == 0

    def test_saves_to_path(self, tmp_path: Path) -> None:
        rows = [_make_row()]
        out = tmp_path / "dataset.json"
        build_dataset(rows, output_path=out)
        assert out.exists()
        data = json.loads(out.read_text())
        assert "snippets" in data

    def test_generated_at_is_string(self) -> None:
        d = build_dataset([])
        assert isinstance(d.created_at, str)


class TestFormatDatasetSummary:
    def test_returns_string(self) -> None:
        d = build_dataset([_make_row()])
        assert isinstance(format_dataset_summary(d), str)

    def test_contains_total(self) -> None:
        rows = [_make_row(tic_id=i) for i in range(3)]
        d = build_dataset(rows)
        assert "3" in format_dataset_summary(d)
