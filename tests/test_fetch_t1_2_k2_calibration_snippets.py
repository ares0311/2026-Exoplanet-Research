"""Tests for Skills/fetch_t1_2_k2_calibration_snippets.py (offline only)."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Skills"))

from fetch_t1_2_k2_calibration_snippets import (  # noqa: E402
    _KEPLER_BJD_OFFSET,
    build_k2_calibration_snippet,
    build_k2_calibration_snippets,
)


def _sample_row(epic_id: int = 201092629, label: int = 1) -> dict:
    return {
        "epic_id": epic_id,
        "label": label,
        "period_days": 10.0,
        "epoch_bjd": 2457100.0,
    }


def _synthetic_transit_fetcher(epic_id: int) -> tuple[list[float], list[float]] | None:
    """Return a synthetic light curve with a real dip at phase 0.

    The fetcher contract (see ``_default_lc_fetcher``) returns time in full
    BJD, matching ``epoch_bjd`` from the manifest -- so this fixture builds
    raw BKJD-style timestamps and then adds ``_KEPLER_BJD_OFFSET`` back,
    exactly like the real fetcher does to lightkurve's raw ``.time.value``.
    """
    period = 10.0
    epoch_bjd = 2457100.0
    epoch_bkjd = epoch_bjd - _KEPLER_BJD_OFFSET  # raw lightkurve-style time
    n_points = 3000
    cadence_days = period * 3 / n_points  # 3 periods of coverage
    time_bkjd = [epoch_bkjd - 1.5 * period + i * cadence_days for i in range(n_points)]
    flux = []
    for i, t in enumerate(time_bkjd):
        phase = ((t - epoch_bkjd) % period) / period
        phase = phase - 1.0 if phase >= 0.5 else phase
        in_transit = abs(phase) < 0.01
        # Deterministic tiny wobble so out-of-transit flux isn't perfectly
        # flat (a perfectly flat baseline degenerates to MAD=0 in _normalise).
        noise = 0.0005 * math.sin(i * 0.37)
        flux.append((0.99 if in_transit else 1.0) + noise)
    time_bjd = [t + _KEPLER_BJD_OFFSET for t in time_bkjd]
    return time_bjd, flux


def _no_data_fetcher(epic_id: int) -> tuple[list[float], list[float]] | None:
    return None


def _short_fetcher(epic_id: int) -> tuple[list[float], list[float]] | None:
    return [1.0, 2.0, 3.0], [1.0, 1.0, 1.0]


def _error_fetcher(epic_id: int) -> tuple[list[float], list[float]] | None:
    raise RuntimeError("simulated network failure")


def test_build_k2_calibration_snippet_ok() -> None:
    result = build_k2_calibration_snippet(
        _sample_row(), lc_fetcher=_synthetic_transit_fetcher
    )
    assert result.flag == "OK"
    assert len(result.flux) == 201
    assert all(math.isfinite(v) for v in result.flux)
    # Transit dip should show up as a low-flux region near the centre bin,
    # compared against a well-sampled out-of-transit window away from the
    # phase-wrap edges (which can be undersampled in a finite time window).
    centre = result.flux[95:106]
    baseline = result.flux[40:80]
    assert sum(centre) / len(centre) < sum(baseline) / len(baseline)


def test_build_k2_calibration_snippet_no_data() -> None:
    result = build_k2_calibration_snippet(_sample_row(), lc_fetcher=_no_data_fetcher)
    assert result.flag == "NO_DATA"
    assert result.flux == ()


def test_build_k2_calibration_snippet_short() -> None:
    result = build_k2_calibration_snippet(_sample_row(), lc_fetcher=_short_fetcher)
    assert result.flag == "SHORT"


def test_build_k2_calibration_snippet_error() -> None:
    result = build_k2_calibration_snippet(_sample_row(), lc_fetcher=_error_fetcher)
    assert result.flag.startswith("ERROR:")


def test_build_k2_calibration_snippet_preserves_label_and_epic_id() -> None:
    result = build_k2_calibration_snippet(
        _sample_row(epic_id=42, label=0), lc_fetcher=_synthetic_transit_fetcher
    )
    assert result.epic_id == 42
    assert result.label == 0


def test_build_k2_calibration_snippets_writes_output_and_resumes(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    rows = [_sample_row(epic_id=i, label=i % 2) for i in range(1, 6)]
    manifest_path.write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    output_path = tmp_path / "snippets.jsonl"

    n_written = build_k2_calibration_snippets(
        manifest_path,
        output_path=output_path,
        lc_fetcher=_synthetic_transit_fetcher,
        commit_report=False,
    )
    assert n_written == 5
    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 5

    # Second run must skip all rows already written (resume).
    n_written_second = build_k2_calibration_snippets(
        manifest_path,
        output_path=output_path,
        lc_fetcher=_synthetic_transit_fetcher,
        commit_report=False,
    )
    assert n_written_second == 0
    lines_after = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines_after) == 5


def test_build_k2_calibration_snippets_records_terminal_failures(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    rows = [_sample_row(epic_id=i) for i in range(1, 4)]
    manifest_path.write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    output_path = tmp_path / "snippets.jsonl"

    n_written = build_k2_calibration_snippets(
        manifest_path,
        output_path=output_path,
        lc_fetcher=_no_data_fetcher,
        max_errors=100,
        commit_report=False,
    )
    assert n_written == 0
    failure_path = tmp_path / "snippets.jsonl.failures.jsonl"
    failure_lines = failure_path.read_text(encoding="utf-8").splitlines()
    assert len(failure_lines) == 3
    for line in failure_lines:
        rec = json.loads(line)
        assert rec["flag"] == "NO_DATA"

    # Ordinary rerun must skip terminal failures, not retry them.
    n_written_retry = build_k2_calibration_snippets(
        manifest_path,
        output_path=output_path,
        lc_fetcher=_synthetic_transit_fetcher,
        commit_report=False,
    )
    assert n_written_retry == 0

    # --retry-failures must re-attempt them.
    n_written_forced = build_k2_calibration_snippets(
        manifest_path,
        output_path=output_path,
        lc_fetcher=_synthetic_transit_fetcher,
        retry_failures=True,
        commit_report=False,
    )
    assert n_written_forced == 3


def test_build_k2_calibration_snippets_stops_early_on_max_errors(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    rows = [_sample_row(epic_id=i) for i in range(1, 11)]
    manifest_path.write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    output_path = tmp_path / "snippets.jsonl"

    build_k2_calibration_snippets(
        manifest_path,
        output_path=output_path,
        lc_fetcher=_no_data_fetcher,
        max_errors=2,
        workers=1,
        commit_report=False,
    )
    failure_path = tmp_path / "snippets.jsonl.failures.jsonl"
    failure_lines = failure_path.read_text(encoding="utf-8").splitlines()
    assert len(failure_lines) == 2
