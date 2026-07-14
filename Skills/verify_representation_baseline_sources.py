"""Verify pinned Phase 3 external-embedding sources without downloading model weights.

The committed contract names exact PyPI wheels and Hugging Face ONNX files.
This verifier checks current primary-source metadata and pinned-file HEAD
headers, then writes a small evidence artifact.  It never installs packages,
downloads model payloads, opens mission data, or authorizes model training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import ssl
import sys
import time
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from Skills.run_report import RunReport, report_path_for, run_and_commit_report  # noqa: E402

DEFAULT_CONTRACT_PATH = REPO_ROOT / "metadata/representation_baseline_source_contract_v1.json"
DEFAULT_OUTPUT_PATH = (
    REPO_ROOT / "artifacts/manifests/representation_baseline_source_verification_v1.json"
)
PYPI_JSON_TEMPLATE = "https://pypi.org/pypi/{name}/{version}/json"
HF_API_TEMPLATE = "https://huggingface.co/api/models/{repo}?blobs=true"
HF_RESOLVE_TEMPLATE = "https://huggingface.co/{repo}/resolve/{commit}/{filename}"

FetchJsonFn = Callable[[str], dict[str, Any]]
HeadFn = Callable[[str], Mapping[str, str]]
ReportFn = Callable[..., bool]


def _ssl_context() -> ssl.SSLContext | None:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return None


def _default_fetch_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=60, context=_ssl_context()) as response:  # noqa: S310
        value = json.load(response)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object from {url}")
    return value


def _default_head(url: str) -> Mapping[str, str]:
    request = urllib.request.Request(url, method="HEAD")  # noqa: S310
    with urllib.request.urlopen(  # noqa: S310
        request,
        timeout=60,
        context=_ssl_context(),
    ) as response:
        return {key.lower(): value for key, value in response.headers.items()}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _require_positive_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def load_contract(path: Path) -> dict[str, Any]:
    """Load and validate the immutable expected-source contract."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("representation baseline contract schema_version must be 1")
    packages = raw.get("packages")
    models = raw.get("models")
    if not isinstance(packages, list) or not packages:
        raise ValueError("representation baseline contract requires packages")
    if not isinstance(models, list) or not models:
        raise ValueError("representation baseline contract requires models")

    total = 0
    identities: set[str] = set()
    for index, package in enumerate(packages):
        if not isinstance(package, dict):
            raise ValueError(f"packages[{index}] must be an object")
        identity = _require_text(package.get("name"), f"packages[{index}].name")
        _require_text(package.get("version"), f"packages[{index}].version")
        _require_text(package.get("requires_python"), f"packages[{index}].requires_python")
        wheel = package.get("wheel")
        if not isinstance(wheel, dict):
            raise ValueError(f"packages[{index}].wheel must be an object")
        _require_text(wheel.get("filename"), f"packages[{index}].wheel.filename")
        _require_text(wheel.get("sha256"), f"packages[{index}].wheel.sha256")
        total += _require_positive_int(
            wheel.get("size_bytes"), f"packages[{index}].wheel.size_bytes"
        )
        if identity in identities:
            raise ValueError(f"duplicate source identity: {identity}")
        identities.add(identity)

    for index, model in enumerate(models):
        if not isinstance(model, dict):
            raise ValueError(f"models[{index}] must be an object")
        identity = _require_text(model.get("repo"), f"models[{index}].repo")
        _require_text(model.get("commit"), f"models[{index}].commit")
        _require_text(model.get("filename"), f"models[{index}].filename")
        _require_text(model.get("sha256"), f"models[{index}].sha256")
        total += _require_positive_int(model.get("size_bytes"), f"models[{index}].size_bytes")
        if identity in identities:
            raise ValueError(f"duplicate source identity: {identity}")
        identities.add(identity)

    expected_total = _require_positive_int(
        raw.get("aggregate_download_bytes"), "aggregate_download_bytes"
    )
    if total != expected_total:
        raise ValueError(
            f"aggregate_download_bytes mismatch: contract={expected_total} computed={total}"
        )
    return raw


def _pypi_result(package: Mapping[str, Any], payload: Mapping[str, Any]) -> dict[str, Any]:
    name = str(package["name"])
    expected_version = str(package["version"])
    info = payload.get("info")
    urls = payload.get("urls")
    if not isinstance(info, dict) or not isinstance(urls, list):
        raise ValueError(f"PyPI metadata for {name} is missing info or urls")
    if info.get("version") != expected_version:
        raise ValueError(
            f"{name} version mismatch: expected={expected_version} actual={info.get('version')}"
        )
    if info.get("requires_python") != package["requires_python"]:
        raise ValueError(
            f"{name} requires_python mismatch: expected={package['requires_python']} "
            f"actual={info.get('requires_python')}"
        )
    wheel = package["wheel"]
    selected = next(
        (
            item
            for item in urls
            if isinstance(item, dict) and item.get("filename") == wheel["filename"]
        ),
        None,
    )
    if selected is None:
        raise ValueError(f"{name} expected wheel is absent: {wheel['filename']}")
    actual_sha = (selected.get("digests") or {}).get("sha256")
    if selected.get("size") != wheel["size_bytes"] or actual_sha != wheel["sha256"]:
        raise ValueError(
            f"{name} wheel metadata mismatch: expected size/hash "
            f"{wheel['size_bytes']}/{wheel['sha256']} actual "
            f"{selected.get('size')}/{actual_sha}"
        )
    return {
        "name": name,
        "version": expected_version,
        "requires_python": info["requires_python"],
        "filename": selected["filename"],
        "size_bytes": selected["size"],
        "sha256": actual_sha,
        "status": "verified",
    }


def _model_result(
    model: Mapping[str, Any],
    payload: Mapping[str, Any],
    headers: Mapping[str, str],
) -> dict[str, Any]:
    repo = str(model["repo"])
    commit = str(model["commit"])
    filename = str(model["filename"])
    if payload.get("id") != repo or payload.get("sha") != commit:
        raise ValueError(
            f"{repo} repository mismatch: expected id/sha={repo}/{commit} "
            f"actual={payload.get('id')}/{payload.get('sha')}"
        )
    siblings = payload.get("siblings")
    if not isinstance(siblings, list):
        raise ValueError(f"{repo} metadata has no sibling list")
    selected = next(
        (
            item
            for item in siblings
            if isinstance(item, dict) and item.get("rfilename") == filename
        ),
        None,
    )
    if selected is None:
        raise ValueError(f"{repo} expected model file is absent: {filename}")
    actual_sha = (selected.get("lfs") or {}).get("sha256")
    if selected.get("size") != model["size_bytes"] or actual_sha != model["sha256"]:
        raise ValueError(
            f"{repo} model metadata mismatch: expected size/hash "
            f"{model['size_bytes']}/{model['sha256']} actual "
            f"{selected.get('size')}/{actual_sha}"
        )
    head_commit = headers.get("x-repo-commit")
    head_size = headers.get("x-linked-size")
    head_sha = headers.get("x-linked-etag", "").strip('"')
    if (
        head_commit != commit
        or head_size != str(model["size_bytes"])
        or head_sha != model["sha256"]
    ):
        raise ValueError(
            f"{repo} pinned HEAD mismatch: expected commit/size/hash "
            f"{commit}/{model['size_bytes']}/{model['sha256']} actual "
            f"{head_commit}/{head_size}/{head_sha}"
        )
    return {
        "repo": repo,
        "commit": commit,
        "filename": filename,
        "size_bytes": selected["size"],
        "sha256": actual_sha,
        "license": model["license"],
        "embedding_dimension": model["embedding_dimension"],
        "max_observations": model["max_observations"],
        "status": "verified",
    }


def _progress(index: int, total: int, label: str, started: float) -> None:
    elapsed = time.monotonic() - started
    rate = index / elapsed if elapsed else 0.0
    eta = (total - index) / rate if rate else 0.0
    print(f"  [{index}/{total}] {label} elapsed={elapsed:.1f}s ETA={eta:.1f}s", flush=True)


def _artifact_contract_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return path.name


def verify_sources(
    contract_path: Path,
    output_path: Path,
    *,
    fetch_json_fn: FetchJsonFn = _default_fetch_json,
    head_fn: HeadFn = _default_head,
    report_fn: ReportFn = run_and_commit_report,
) -> dict[str, Any]:
    """Verify all contract sources and write one metadata-only evidence artifact."""
    contract = load_contract(contract_path)
    started_at = datetime.now(UTC)
    started = time.monotonic()
    packages = contract["packages"]
    models = contract["models"]
    total_steps = len(packages) + 2 * len(models)
    print(
        "Representation baseline source verification startup: "
        f"packages={len(packages)} models={len(models)} metadata_requests={total_steps} "
        "payload_downloads=0 mode=metadata-only",
        flush=True,
    )

    package_results: list[dict[str, Any]] = []
    step = 0
    for package in packages:
        name = urllib.parse.quote(str(package["name"]), safe="")
        version = urllib.parse.quote(str(package["version"]), safe="")
        payload = fetch_json_fn(PYPI_JSON_TEMPLATE.format(name=name, version=version))
        package_results.append(_pypi_result(package, payload))
        step += 1
        _progress(step, total_steps, f"verified PyPI {package['name']}", started)

    model_results: list[dict[str, Any]] = []
    for model in models:
        repo = str(model["repo"])
        payload = fetch_json_fn(HF_API_TEMPLATE.format(repo=repo))
        step += 1
        _progress(step, total_steps, f"verified Hugging Face API {repo}", started)
        pinned_url = HF_RESOLVE_TEMPLATE.format(
            repo=repo,
            commit=model["commit"],
            filename=urllib.parse.quote(str(model["filename"]), safe=""),
        )
        headers = {key.lower(): value for key, value in head_fn(pinned_url).items()}
        model_results.append(_model_result(model, payload, headers))
        step += 1
        _progress(step, total_steps, f"verified pinned HEAD {repo}", started)

    verified_bytes = sum(item["size_bytes"] for item in package_results + model_results)
    if verified_bytes != contract["aggregate_download_bytes"]:
        raise ValueError(
            f"verified aggregate mismatch: expected={contract['aggregate_download_bytes']} "
            f"actual={verified_bytes}"
        )
    artifact: dict[str, Any] = {
        "schema_version": 1,
        "artifact_id": "representation_baseline_source_verification_v1",
        "verified_at_utc": datetime.now(UTC).isoformat(),
        "contract_path": _artifact_contract_path(contract_path),
        "contract_sha256": _sha256(contract_path),
        "runtime_python": sys.version.split()[0],
        "status": "success",
        "verification_mode": "primary_metadata_only",
        "metadata_requests": total_steps,
        "payload_bytes_downloaded": 0,
        "aggregate_download_bytes_if_installed": verified_bytes,
        "packages": package_results,
        "models": model_results,
        "training_authorized": False,
        "limitations": contract["limitations"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    elapsed = time.monotonic() - started
    report = RunReport(
        script="verify_representation_baseline_sources",
        status="success",
        started_at=started_at.isoformat(),
        completed_at=datetime.now(UTC).isoformat(),
        elapsed_seconds=elapsed,
        items_processed=total_steps,
        items_written=len(package_results) + len(model_results),
        output_paths=(str(output_path),),
        notes="primary metadata only; zero package, model, or mission-data payload downloads",
    )
    report_path = report_path_for("verify_representation_baseline_sources")
    if not report_fn(report, report_path):
        print(f"WARNING: Run Report push failed for {report_path}", flush=True)
    print(
        "Representation baseline source verification COMPLETE: "
        f"sources={len(package_results) + len(model_results)} "
        f"projected_bytes={verified_bytes} elapsed={elapsed:.1f}s",
        flush=True,
    )
    return artifact


def format_status(artifact: Mapping[str, Any]) -> str:
    """Format a durable source-verification artifact for operator inspection."""
    return (
        "Representation baseline source status: "
        f"status={artifact.get('status')} packages={len(artifact.get('packages', []))} "
        f"models={len(artifact.get('models', []))} "
        f"projected_bytes={artifact.get('aggregate_download_bytes_if_installed')} "
        f"payload_bytes_downloaded={artifact.get('payload_bytes_downloaded')}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--status-only", action="store_true")
    args = parser.parse_args(argv)
    if args.status_only:
        if not args.output.is_file():
            print(f"ERROR: verification artifact is missing: {args.output}", file=sys.stderr)
            return 1
        artifact = json.loads(args.output.read_text(encoding="utf-8"))
        print(format_status(artifact), flush=True)
        return 0
    try:
        artifact = verify_sources(args.contract, args.output)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: representation baseline source verification failed: {exc}", file=sys.stderr)
        return 1
    print(format_status(artifact), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
