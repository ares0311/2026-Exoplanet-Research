"""Tests for fetch_tess_kepler_overlap_snippets.py."""
from __future__ import annotations

import json
import sys
from io import StringIO
from types import SimpleNamespace

import Skills.fetch_tess_kepler_overlap_snippets as overlap
from Skills.fetch_tess_kepler_overlap_snippets import (
    KoiRow,
    _normalise,
    _phase_fold_bin,
    build_koi_tess_snippet,
    build_koi_tess_snippets,
    fetch_koi_table,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_CONFIRMED_ROW = KoiRow(
    kepid=757099,
    kepoi_name="K00001.01",
    disposition="CONFIRMED",
    period_days=2.4706,
    epoch_bkjd=120.7285,
)

_FP_ROW = KoiRow(
    kepid=1026032,
    kepoi_name="K00002.01",
    disposition="FALSE POSITIVE",
    period_days=2.2047,
    epoch_bkjd=65.9417,
)


def _make_lc_fetcher(n_points: int = 500):
    """Injectable fetcher returning a flat sinusoid in full BJD."""
    import math

    def fetcher(kepid: int, period: float, epoch_bjd: float):
        # Simple flat signal with a tiny transit dip at phase 0.
        time_bjd = [epoch_bjd + i * period / n_points for i in range(n_points)]
        flux = []
        for t in time_bjd:
            ph = ((t - epoch_bjd) % period) / period
            ph = ph - 1.0 if ph >= 0.5 else ph
            flux.append(1.0 - 0.01 * math.exp(-200 * ph**2))
        return time_bjd, flux

    return fetcher


def _no_data_fetcher(kepid: int, period: float, epoch_bjd: float):
    return None


def _error_fetcher(kepid: int, period: float, epoch_bjd: float):
    raise RuntimeError("network failure")


# ---------------------------------------------------------------------------
# _phase_fold_bin
# ---------------------------------------------------------------------------


def test_phase_fold_bin_length():
    time = [float(i) for i in range(100)]
    flux = [1.0] * 100
    result = _phase_fold_bin(time, flux, period=10.0, epoch=0.0, n_bins=20)
    assert len(result) == 20


def test_phase_fold_bin_skips_nonfinite():
    import math
    time = [0.0, float("nan"), 2.0]
    flux = [1.0, 1.0, 1.0]
    result = _phase_fold_bin(time, flux, period=5.0, epoch=0.0, n_bins=10)
    assert len(result) == 10
    assert all(math.isfinite(v) for v in result)


def test_phase_fold_bin_empty_bin_defaults_to_one():
    # Only one point; all other bins should default to 1.0.
    result = _phase_fold_bin([0.0], [0.5], period=10.0, epoch=0.0, n_bins=5)
    filled = [v for v in result if v != 1.0]
    empty = [v for v in result if v == 1.0]
    assert len(filled) == 1
    assert len(empty) == 4


# ---------------------------------------------------------------------------
# _normalise
# ---------------------------------------------------------------------------


def test_normalise_length():
    result = _normalise([1.0, 1.1, 0.9, 1.05, 0.95])
    assert len(result) == 5


def test_normalise_rejects_nonfinite():
    result = _normalise([1.0, float("nan"), 0.9])
    assert result == []


def test_normalise_constant_returns_zeros():
    result = _normalise([1.0] * 10)
    assert result == [0.0] * 10


# ---------------------------------------------------------------------------
# KoiRow label derivation
# ---------------------------------------------------------------------------


def test_confirmed_row_gives_label_1():
    fetcher = _make_lc_fetcher()
    r = build_koi_tess_snippet(_CONFIRMED_ROW, n_bins=201, lc_fetcher=fetcher)
    assert r.label == 1


def test_fp_row_gives_label_0():
    fetcher = _make_lc_fetcher()
    r = build_koi_tess_snippet(_FP_ROW, n_bins=201, lc_fetcher=fetcher)
    assert r.label == 0


# ---------------------------------------------------------------------------
# build_koi_tess_snippet — happy path
# ---------------------------------------------------------------------------


def test_snippet_ok_flag():
    r = build_koi_tess_snippet(_CONFIRMED_ROW, n_bins=201, lc_fetcher=_make_lc_fetcher())
    assert r.flag == "OK"


def test_snippet_correct_n_bins():
    r = build_koi_tess_snippet(_CONFIRMED_ROW, n_bins=201, lc_fetcher=_make_lc_fetcher())
    assert len(r.flux) == 201


def test_snippet_flux_all_finite():
    import math
    r = build_koi_tess_snippet(_CONFIRMED_ROW, n_bins=201, lc_fetcher=_make_lc_fetcher())
    assert all(math.isfinite(v) for v in r.flux)


def test_snippet_epoch_bjd_conversion():
    # epoch_bjd = epoch_bkjd + 2454833
    r = build_koi_tess_snippet(_CONFIRMED_ROW, n_bins=201, lc_fetcher=_make_lc_fetcher())
    expected_epoch = _CONFIRMED_ROW.epoch_bkjd + 2454833.0
    assert abs(r.epoch_bjd - expected_epoch) < 1e-6


def test_snippet_period_preserved():
    r = build_koi_tess_snippet(_CONFIRMED_ROW, n_bins=201, lc_fetcher=_make_lc_fetcher())
    assert abs(r.period_days - _CONFIRMED_ROW.period_days) < 1e-9


# ---------------------------------------------------------------------------
# build_koi_tess_snippet — failure paths
# ---------------------------------------------------------------------------


def test_snippet_no_data_flag():
    r = build_koi_tess_snippet(_CONFIRMED_ROW, n_bins=201, lc_fetcher=_no_data_fetcher)
    assert r.flag in {"NO_DATA", "NO_LIGHTKURVE"}


def test_snippet_error_flag():
    r = build_koi_tess_snippet(_CONFIRMED_ROW, n_bins=201, lc_fetcher=_error_fetcher)
    assert r.flag.startswith("ERROR:")


def test_snippet_short_flag():
    def short_fetcher(kepid, period, epoch_bjd):
        return [epoch_bjd], [1.0]  # only 1 point — not enough

    r = build_koi_tess_snippet(_CONFIRMED_ROW, n_bins=201, lc_fetcher=short_fetcher)
    assert r.flag == "SHORT"


# ---------------------------------------------------------------------------
# build_koi_tess_snippets — batch runner
# ---------------------------------------------------------------------------


def test_batch_writes_jsonl(tmp_path):
    out = tmp_path / "overlap.jsonl"
    rows = [_CONFIRMED_ROW, _FP_ROW]
    n = build_koi_tess_snippets(
        rows, n_bins=201, output_path=out, lc_fetcher=_make_lc_fetcher()
    )
    assert n == 2
    lines = [json.loads(ln) for ln in out.read_text().splitlines() if ln.strip()]
    assert len(lines) == 2


def test_batch_jsonl_fields(tmp_path):
    out = tmp_path / "overlap.jsonl"
    build_koi_tess_snippets(
        [_CONFIRMED_ROW], n_bins=201, output_path=out, lc_fetcher=_make_lc_fetcher()
    )
    obj = json.loads(out.read_text().strip())
    assert "tic_id" in obj
    assert "label" in obj
    assert "flux" in obj
    assert "source" in obj
    assert obj["source"] == "koi_tess_overlap"
    assert "kepoi_name" in obj


def test_batch_resume_skips_done(tmp_path):
    out = tmp_path / "overlap.jsonl"
    rows = [_CONFIRMED_ROW, _FP_ROW]
    # First run — writes 2 snippets.
    n1 = build_koi_tess_snippets(
        rows, n_bins=201, output_path=out, lc_fetcher=_make_lc_fetcher()
    )
    assert n1 == 2
    # Second run — both are already done; should write 0 new snippets.
    n2 = build_koi_tess_snippets(
        rows, n_bins=201, output_path=out, lc_fetcher=_make_lc_fetcher()
    )
    assert n2 == 0
    # Total lines should still be 2.
    lines = [ln for ln in out.read_text().splitlines() if ln.strip()]
    assert len(lines) == 2


def test_batch_max_errors_stops_early(tmp_path):
    out = tmp_path / "overlap.jsonl"
    # 5 rows, all will error, max_errors=3.
    rows = [_CONFIRMED_ROW] * 5
    n = build_koi_tess_snippets(
        rows, n_bins=201, output_path=out,
        lc_fetcher=_error_fetcher, max_errors=3,
    )
    assert n == 0


def test_batch_records_terminal_failures_in_sidecar(tmp_path):
    out = tmp_path / "overlap.jsonl"
    rows = [_CONFIRMED_ROW]
    n = build_koi_tess_snippets(
        rows, n_bins=201, output_path=out, lc_fetcher=_no_data_fetcher
    )
    assert n == 0
    failure_log = out.with_name(out.name + ".failures.jsonl")
    assert failure_log.exists()
    failures = [json.loads(ln) for ln in failure_log.read_text().splitlines()]
    assert len(failures) == 1
    assert failures[0]["kepoi_name"] == _CONFIRMED_ROW.kepoi_name
    assert failures[0]["flag"] in {"NO_DATA", "NO_LIGHTKURVE"}


def test_batch_resume_skips_terminal_failures(tmp_path):
    out = tmp_path / "overlap.jsonl"
    rows = [_CONFIRMED_ROW]
    build_koi_tess_snippets(
        rows, n_bins=201, output_path=out, lc_fetcher=_no_data_fetcher
    )

    n = build_koi_tess_snippets(
        rows, n_bins=201, output_path=out, lc_fetcher=_make_lc_fetcher()
    )

    assert n == 0
    assert not out.exists() or out.read_text() == ""


def test_batch_retry_failures_reprocesses_terminal_failures(tmp_path):
    out = tmp_path / "overlap.jsonl"
    rows = [_CONFIRMED_ROW]
    build_koi_tess_snippets(
        rows, n_bins=201, output_path=out, lc_fetcher=_no_data_fetcher
    )

    n = build_koi_tess_snippets(
        rows,
        n_bins=201,
        output_path=out,
        lc_fetcher=_make_lc_fetcher(),
        retry_failures=True,
    )

    assert n == 1
    lines = [json.loads(ln) for ln in out.read_text().splitlines() if ln.strip()]
    assert len(lines) == 1


def test_batch_groups_same_kic_into_one_fetch(tmp_path):
    out = tmp_path / "overlap.jsonl"
    sibling = KoiRow(
        kepid=_CONFIRMED_ROW.kepid,
        kepoi_name="K00001.02",
        disposition="FALSE POSITIVE",
        period_days=4.12,
        epoch_bkjd=121.0,
    )
    calls: list[int] = []
    base_fetcher = _make_lc_fetcher()

    def counting_fetcher(kepid: int, period: float, epoch_bjd: float):
        calls.append(kepid)
        return base_fetcher(kepid, period, epoch_bjd)

    n = build_koi_tess_snippets(
        [_CONFIRMED_ROW, sibling],
        n_bins=201,
        output_path=out,
        lc_fetcher=counting_fetcher,
        workers=2,
        request_delay=0,
    )

    assert n == 2
    assert calls == [_CONFIRMED_ROW.kepid]
    lines = [json.loads(ln) for ln in out.read_text().splitlines() if ln.strip()]
    assert {line["kepoi_name"] for line in lines} == {"K00001.01", "K00001.02"}


def test_batch_progress_survives_closed_stdout(tmp_path, monkeypatch):
    out = tmp_path / "overlap.jsonl"
    closed_stdout = StringIO()
    closed_stdout.close()
    monkeypatch.setattr(sys, "stdout", closed_stdout)

    n = build_koi_tess_snippets(
        [_CONFIRMED_ROW],
        n_bins=201,
        output_path=out,
        lc_fetcher=_make_lc_fetcher(),
        workers=1,
        request_delay=0,
    )

    assert n == 1
    assert len([ln for ln in out.read_text().splitlines() if ln.strip()]) == 1


def test_default_lc_fetcher_avoids_download_all(monkeypatch):
    calls = {"download_one": 0}

    class FakeTable:
        def __len__(self):
            return 1

        def __getitem__(self, key):
            return self

    class FakeSearchResult:
        table = FakeTable()

        def __len__(self):
            return 1

        def download_all(self):
            raise AssertionError("download_all mutates process-global stdout")

        def _download_one(self, *, table, quality_bitmask, download_dir, cutout_size):
            calls["download_one"] += 1
            return SimpleNamespace(
                time=SimpleNamespace(value=[0.0, 1.0, 2.0]),
                flux=SimpleNamespace(value=[1.0, 0.9, 1.1]),
                normalize=lambda: SimpleNamespace(
                    time=SimpleNamespace(value=[0.0, 1.0, 2.0]),
                    flux=SimpleNamespace(value=[1.0, 0.9, 1.1]),
                ),
            )

    class FakeCollection:
        def __init__(self, curves):
            self.curves = curves

        def stitch(self):
            return self.curves[0]

    fake_lk = SimpleNamespace(
        search_lightcurve=lambda *args, **kwargs: FakeSearchResult(),
        LightCurveCollection=FakeCollection,
    )
    monkeypatch.setitem(sys.modules, "lightkurve", fake_lk)

    raw = overlap._default_lc_fetcher(123, 2.0, 2454900.0)

    assert raw is not None
    assert calls == {"download_one": 1}


def test_batch_creates_parent_dirs(tmp_path):
    out = tmp_path / "subdir" / "nested" / "out.jsonl"
    build_koi_tess_snippets(
        [_CONFIRMED_ROW], n_bins=201, output_path=out, lc_fetcher=_make_lc_fetcher()
    )
    assert out.exists()


# ---------------------------------------------------------------------------
# fetch_koi_table — offline unit test with mock server response
# ---------------------------------------------------------------------------


def test_fetch_koi_table_parses_records(tmp_path):
    mock_data = json.dumps([
        {
            "kepid": 757099,
            "kepoi_name": "K00001.01",
            "koi_disposition": "CONFIRMED",
            "koi_period": 2.4706,
            "koi_time0bk": 120.7285,
        },
        {
            "kepid": 1026032,
            "kepoi_name": "K00002.01",
            "koi_disposition": "FALSE POSITIVE",
            "koi_period": 2.2047,
            "koi_time0bk": 65.9417,
        },
    ])
    # Write to a temp file and use a file:// URL.
    f = tmp_path / "koi.json"
    f.write_text(mock_data, encoding="utf-8")
    rows = fetch_koi_table(f"file://{f}")
    assert len(rows) == 2
    confirmed = [r for r in rows if r.disposition == "CONFIRMED"]
    fp = [r for r in rows if r.disposition == "FALSE POSITIVE"]
    assert len(confirmed) == 1
    assert len(fp) == 1


def test_fetch_koi_table_uses_certifi_ssl_context(monkeypatch):
    mock_data = json.dumps([
        {
            "kepid": 757099,
            "kepoi_name": "K00001.01",
            "koi_disposition": "CONFIRMED",
            "koi_period": 2.4706,
            "koi_time0bk": 120.7285,
        },
    ]).encode()
    captured: dict[str, object] = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return mock_data

    def fake_urlopen(url, *, timeout, context):  # noqa: ANN001, ANN202
        captured["url"] = url
        captured["timeout"] = timeout
        captured["context"] = context
        return _Response()

    monkeypatch.setattr(overlap, "urlopen", fake_urlopen)

    rows = fetch_koi_table("https://example.invalid/tap")

    assert len(rows) == 1
    assert captured["timeout"] == 120
    assert captured["context"] is not None


def test_fetch_koi_table_rejects_invalid_disposition(tmp_path):
    mock_data = json.dumps([
        {
            "kepid": 999,
            "kepoi_name": "K00099.01",
            "koi_disposition": "CANDIDATE",
            "koi_period": 5.0,
            "koi_time0bk": 100.0,
        },
    ])
    f = tmp_path / "koi.json"
    f.write_text(mock_data, encoding="utf-8")
    rows = fetch_koi_table(f"file://{f}")
    assert rows == []


def test_fetch_koi_table_rejects_missing_period(tmp_path):
    mock_data = json.dumps([
        {
            "kepid": 999,
            "kepoi_name": "K00099.01",
            "koi_disposition": "CONFIRMED",
            "koi_period": None,
            "koi_time0bk": 100.0,
        },
    ])
    f = tmp_path / "koi.json"
    f.write_text(mock_data, encoding="utf-8")
    rows = fetch_koi_table(f"file://{f}")
    assert rows == []
