"""Tests for the metadata-only representation baseline source verifier."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Skills"))

from verify_representation_baseline_sources import (
    format_status,
    load_contract,
    main,
    verify_sources,
)


def _contract() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "contract_id": "test",
        "packages": [
            {
                "name": "example-package",
                "version": "1.2.3",
                "requires_python": ">=3.11",
                "wheel": {
                    "filename": "example_package-1.2.3-py3-none-any.whl",
                    "size_bytes": 10,
                    "sha256": "a" * 64,
                },
            }
        ],
        "models": [
            {
                "name": "Example Model",
                "repo": "example/model",
                "commit": "b" * 40,
                "filename": "model.onnx",
                "size_bytes": 20,
                "sha256": "c" * 64,
                "license": "MIT",
                "embedding_dimension": 8,
                "max_observations": 16,
            }
        ],
        "aggregate_download_bytes": 30,
        "limitations": ["test contract"],
    }


def _write_contract(path: Path, *, mutate: Any = None) -> dict[str, Any]:
    contract = _contract()
    if mutate is not None:
        mutate(contract)
    path.write_text(json.dumps(contract), encoding="utf-8")
    return contract


def _fetch_json(url: str) -> dict[str, Any]:
    if "pypi.org" in url:
        return {
            "info": {"version": "1.2.3", "requires_python": ">=3.11"},
            "urls": [
                {
                    "filename": "example_package-1.2.3-py3-none-any.whl",
                    "size": 10,
                    "digests": {"sha256": "a" * 64},
                }
            ],
        }
    if "huggingface.co/api/models" in url:
        return {
            "id": "example/model",
            "sha": "b" * 40,
            "siblings": [
                {
                    "rfilename": "model.onnx",
                    "size": 20,
                    "lfs": {"sha256": "c" * 64},
                }
            ],
        }
    raise AssertionError(f"unexpected URL: {url}")


def _head(_url: str) -> dict[str, str]:
    return {
        "X-Repo-Commit": "b" * 40,
        "X-Linked-Size": "20",
        "X-Linked-ETag": f'"{"c" * 64}"',
    }


def test_load_contract_rejects_incorrect_aggregate(tmp_path: Path) -> None:
    path = tmp_path / "contract.json"
    _write_contract(path, mutate=lambda value: value.update(aggregate_download_bytes=31))
    with pytest.raises(ValueError, match="aggregate_download_bytes mismatch"):
        load_contract(path)


def test_verify_sources_writes_success_without_payload_downloads(tmp_path: Path) -> None:
    contract_path = tmp_path / "contract.json"
    output_path = tmp_path / "result.json"
    _write_contract(contract_path)
    reports: list[tuple[Any, Path]] = []

    def _report(report: Any, path: Path) -> bool:
        reports.append((report, path))
        return True

    result = verify_sources(
        contract_path,
        output_path,
        fetch_json_fn=_fetch_json,
        head_fn=_head,
        report_fn=_report,
    )

    assert output_path.is_file()
    assert result["status"] == "success"
    assert result["payload_bytes_downloaded"] == 0
    assert result["aggregate_download_bytes_if_installed"] == 30
    assert result["metadata_requests"] == 3
    assert result["packages"][0]["status"] == "verified"
    assert result["models"][0]["status"] == "verified"
    assert result["training_authorized"] is False
    assert len(reports) == 1
    assert reports[0][0].items_processed == 3


def test_verify_sources_fails_closed_on_package_hash_drift(tmp_path: Path) -> None:
    contract_path = tmp_path / "contract.json"
    output_path = tmp_path / "result.json"
    _write_contract(contract_path)

    def _drifted_fetch(url: str) -> dict[str, Any]:
        payload = _fetch_json(url)
        if "pypi.org" in url:
            payload["urls"][0]["digests"]["sha256"] = "d" * 64
        return payload

    with pytest.raises(ValueError, match="wheel metadata mismatch"):
        verify_sources(
            contract_path,
            output_path,
            fetch_json_fn=_drifted_fetch,
            head_fn=_head,
            report_fn=lambda *_args, **_kwargs: True,
        )
    assert not output_path.exists()


def test_verify_sources_fails_closed_on_repository_commit_drift(tmp_path: Path) -> None:
    contract_path = tmp_path / "contract.json"
    output_path = tmp_path / "result.json"
    _write_contract(contract_path)

    def _drifted_fetch(url: str) -> dict[str, Any]:
        payload = _fetch_json(url)
        if "huggingface.co/api/models" in url:
            payload["sha"] = "d" * 40
        return payload

    with pytest.raises(ValueError, match="repository mismatch"):
        verify_sources(
            contract_path,
            output_path,
            fetch_json_fn=_drifted_fetch,
            head_fn=_head,
            report_fn=lambda *_args, **_kwargs: True,
        )


def test_verify_sources_fails_closed_on_pinned_head_drift(tmp_path: Path) -> None:
    contract_path = tmp_path / "contract.json"
    output_path = tmp_path / "result.json"
    _write_contract(contract_path)
    headers = _head("")
    headers["X-Linked-ETag"] = '"drifted"'

    with pytest.raises(ValueError, match="pinned HEAD mismatch"):
        verify_sources(
            contract_path,
            output_path,
            fetch_json_fn=_fetch_json,
            head_fn=lambda _url: headers,
            report_fn=lambda *_args, **_kwargs: True,
        )


def test_format_status_reports_source_counts() -> None:
    text = format_status(
        {
            "status": "success",
            "packages": [{"name": "one"}],
            "models": [{"name": "two"}],
            "aggregate_download_bytes_if_installed": 30,
            "payload_bytes_downloaded": 0,
        }
    )
    assert "status=success" in text
    assert "packages=1" in text
    assert "models=1" in text
    assert "payload_bytes_downloaded=0" in text


def test_status_only_never_calls_network(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output_path = tmp_path / "result.json"
    output_path.write_text(
        json.dumps(
            {
                "status": "success",
                "packages": [],
                "models": [],
                "aggregate_download_bytes_if_installed": 0,
                "payload_bytes_downloaded": 0,
            }
        ),
        encoding="utf-8",
    )
    assert main(["--status-only", "--output", str(output_path)]) == 0
    assert "status=success" in capsys.readouterr().out
