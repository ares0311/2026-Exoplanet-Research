"""Offline tests for the Catalina stellar-variability label-source verifier."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Skills"))

from verify_stellar_variability_label_source import (  # noqa: E402
    format_status,
    load_contract,
    main,
    verify_source,
)


def _contract() -> dict[str, Any]:
    classes = {str(index): f"class_{index}" for index in range(1, 18)}
    counts = {str(index): 1 for index in range(1, 18)}
    return {
        "schema_version": 1,
        "contract_id": "test",
        "catalog": {
            "name": "Example publication catalog",
            "paper_doi": "10.example/paper",
            "catalog_doi": "10.example/catalog",
            "tap_endpoint": "https://example.invalid/tap",
            "table": "J/ApJS/213/9/table3",
            "gzip_url": "https://example.invalid/table3.dat.gz",
            "delivery": {
                "content_length_bytes": 1234,
                "etag": '"abc"',
                "last_modified": "Mon, 01 Sep 2014 14:55:23 GMT",
                "content_type": "application/x-gzip",
            },
            "row_count": 17,
            "required_columns": {
                "CRTS": "CHAR(16)",
                "RAJ2000": "DOUBLE",
                "DEJ2000": "DOUBLE",
                "Per": "DOUBLE",
                "Cl": "SMALLINT",
                "f_Cl": "CHAR(1)",
            },
            "classes": classes,
            "class_counts": counts,
            "ground_truth_basis": "publication reviewed",
        },
        "rejected_alternatives": [
            {"source": "automated", "reason": "model_outputs_not_ground_truth"},
            {"source": "large gated", "reason": "gated_and_storage_exceeds_limit"},
        ],
        "limitations": ["test only"],
    }


def _write_contract(path: Path, *, mutate: Any = None) -> dict[str, Any]:
    contract = _contract()
    if mutate is not None:
        mutate(contract)
    path.write_text(json.dumps(contract), encoding="utf-8")
    return contract


def _head(_url: str) -> dict[str, str]:
    return {
        "Content-Length": "1234",
        "ETag": '"abc"',
        "Last-Modified": "Mon, 01 Sep 2014 14:55:23 GMT",
        "Content-Type": "application/x-gzip",
    }


def _tap(_endpoint: str, query: str) -> dict[str, Any]:
    if "TAP_SCHEMA.columns" in query:
        types = {
            "CRTS": "CHAR(16)",
            "RAJ2000": "DOUBLE",
            "DEJ2000": "DOUBLE",
            "Per": "DOUBLE",
            "Cl": "SMALLINT",
            "f_Cl": "CHAR(1)",
        }
        return {
            "metadata": [{"name": "table_name"}],
            "data": [
                ['"J/ApJS/213/9/table3"', f'"{name}"', datatype, "description"]
                for name, datatype in types.items()
            ],
        }
    if "GROUP BY Cl" in query:
        return {
            "metadata": [{"name": "Cl"}, {"name": "n"}],
            "data": [[index, 1] for index in range(1, 18)],
        }
    if "COUNT(*)" in query:
        return {"metadata": [{"name": "n"}], "data": [[17]]}
    if "TOP 3" in query:
        return {
            "metadata": [{"name": "CRTS"}],
            "data": [
                ["CSS_A", 10.0, -20.0, 0.5, 1, None],
                ["CSS_B", 11.0, -21.0, 1.5, 2, "a"],
                ["CSS_C", 12.0, -22.0, 2.5, 3, "b"],
            ],
        }
    raise AssertionError(f"unexpected query: {query}")


def test_load_contract_rejects_class_count_sum_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "contract.json"

    def _mutate(value: dict[str, Any]) -> None:
        value["catalog"]["class_counts"]["17"] = 2

    _write_contract(path, mutate=_mutate)
    with pytest.raises(ValueError, match="must sum"):
        load_contract(path)


def test_verify_source_writes_bounded_success_artifact(tmp_path: Path) -> None:
    contract_path = tmp_path / "contract.json"
    output_path = tmp_path / "result.json"
    _write_contract(contract_path)
    reports: list[Any] = []

    def _report(report: Any, _path: Path) -> bool:
        reports.append(report)
        return True

    result = verify_source(
        contract_path,
        output_path,
        tap_fn=_tap,
        head_fn=_head,
        report_fn=_report,
    )

    assert output_path.is_file()
    assert result["status"] == "success"
    assert result["catalog"]["row_count"] == 17
    assert len(result["catalog"]["class_counts"]) == 17
    assert len(result["catalog"]["sample_rows"]) == 3
    assert result["full_catalog_payload_bytes_downloaded"] == 0
    assert result["source_identity_authorized"] is True
    assert result["crossmatch_authorized"] is False
    assert result["training_authorized"] is False
    assert reports[0].items_processed == 5


def test_verify_source_fails_closed_on_delivery_drift(tmp_path: Path) -> None:
    contract_path = tmp_path / "contract.json"
    _write_contract(contract_path)
    headers = _head("")
    headers["Content-Length"] = "1235"
    with pytest.raises(ValueError, match="delivery metadata mismatch"):
        verify_source(
            contract_path,
            tmp_path / "result.json",
            tap_fn=_tap,
            head_fn=lambda _url: headers,
            report_fn=lambda *_args, **_kwargs: True,
        )


def test_verify_source_fails_closed_on_schema_drift(tmp_path: Path) -> None:
    contract_path = tmp_path / "contract.json"
    _write_contract(contract_path)

    def _drifted_tap(endpoint: str, query: str) -> dict[str, Any]:
        payload = _tap(endpoint, query)
        if "TAP_SCHEMA.columns" in query:
            payload["data"][0][2] = "VARCHAR"
        return payload

    with pytest.raises(ValueError, match="schema mismatch"):
        verify_source(
            contract_path,
            tmp_path / "result.json",
            tap_fn=_drifted_tap,
            head_fn=_head,
            report_fn=lambda *_args, **_kwargs: True,
        )


def test_verify_source_fails_closed_on_class_count_drift(tmp_path: Path) -> None:
    contract_path = tmp_path / "contract.json"
    _write_contract(contract_path)

    def _drifted_tap(endpoint: str, query: str) -> dict[str, Any]:
        payload = _tap(endpoint, query)
        if "GROUP BY Cl" in query:
            payload["data"][0][1] = 2
        return payload

    with pytest.raises(ValueError, match="class-count mismatch"):
        verify_source(
            contract_path,
            tmp_path / "result.json",
            tap_fn=_drifted_tap,
            head_fn=_head,
            report_fn=lambda *_args, **_kwargs: True,
        )


def test_verify_source_fails_closed_on_bad_sample(tmp_path: Path) -> None:
    contract_path = tmp_path / "contract.json"
    _write_contract(contract_path)

    def _drifted_tap(endpoint: str, query: str) -> dict[str, Any]:
        payload = _tap(endpoint, query)
        if "TOP 3" in query:
            payload["data"][0][1] = 500.0
        return payload

    with pytest.raises(ValueError, match="invalid coordinates"):
        verify_source(
            contract_path,
            tmp_path / "result.json",
            tap_fn=_drifted_tap,
            head_fn=_head,
            report_fn=lambda *_args, **_kwargs: True,
        )


def test_format_status_reports_authorization_boundary() -> None:
    text = format_status(
        {
            "status": "success",
            "catalog": {"row_count": 17, "class_counts": {"1": 17}},
            "full_catalog_payload_bytes_downloaded": 0,
            "training_authorized": False,
        }
    )
    assert "rows=17" in text
    assert "classes=1" in text
    assert "full_catalog_payload_bytes_downloaded=0" in text
    assert "training_authorized=False" in text


def test_status_only_never_calls_network(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output_path = tmp_path / "result.json"
    output_path.write_text(
        json.dumps(
            {
                "status": "success",
                "catalog": {"row_count": 17, "class_counts": {"1": 17}},
                "full_catalog_payload_bytes_downloaded": 0,
                "training_authorized": False,
            }
        ),
        encoding="utf-8",
    )
    assert main(["--status-only", "--output", str(output_path)]) == 0
    assert "training_authorized=False" in capsys.readouterr().out
