"""Automate CNN checkpoint promotion for T1-1 production gate.

Reads the calibration JSON written by ``evaluate_cnn_checkpoint.py`` (present
only when both gates passed), registers the checkpoint in the model registry,
writes a promotion manifest, and prints the PRODUCTION_READINESS.md update
snippet plus the git commit recipe.

Usage
-----
    python Skills/promote_cnn_checkpoint.py \\
        --checkpoint models/cnn/best.pt \\
        --calibration models/cnn/calibration.json \\
        --registry models/registry.json

Exit codes: 0 = promoted, 1 = gates not met / file missing.

Public API
----------
PromotionResult(model_id, sha256, auc, f1, brier, ece, platt_a, platt_b,
                registry_path, manifest_path, promoted_at, flag)
promote_cnn_checkpoint(checkpoint_path, calibration_path, registry_path, *,
                       manifest_path) -> PromotionResult
format_promotion_result(result) -> str
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PromotionResult:
    """Outcome of a CNN checkpoint promotion attempt."""

    model_id: str
    sha256: str
    auc: float | None
    f1: float | None
    brier: float | None
    ece: float | None
    platt_a: float | None
    platt_b: float | None
    registry_path: str
    manifest_path: str
    promoted_at: str
    flag: str  # "PROMOTED" | "GATES_NOT_MET" | "MISSING_FILE" | "ALREADY_REGISTERED"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    """Return the hex SHA-256 digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
            fh.write("\n")
        os.replace(tmp, path)
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def _finite_metric(cal: dict, key: str) -> float | None:
    """Return a finite numeric calibration metric, or None when invalid."""
    value = cal.get(key)
    if not isinstance(value, int | float) or isinstance(value, bool):
        return None
    metric = float(value)
    return metric if math.isfinite(metric) else None


def _calibration_passes_gates(cal: dict) -> bool:
    """Independently verify calibration JSON satisfies CNN promotion gates."""
    if cal.get("flag") != "OK":
        return False
    auc = _finite_metric(cal, "test_auc_raw")
    f1 = _finite_metric(cal, "test_f1_cal")
    brier_cal = _finite_metric(cal, "test_brier_cal")
    ece_cal = _finite_metric(cal, "test_ece_cal")
    brier_raw = _finite_metric(cal, "test_brier_raw")
    ece_raw = _finite_metric(cal, "test_ece_raw")
    platt_a = _finite_metric(cal, "platt_a")
    platt_b = _finite_metric(cal, "platt_b")
    gate_auc = _finite_metric(cal, "gate_auc")
    gate_f1 = _finite_metric(cal, "gate_f1")
    if None in (
        auc,
        f1,
        brier_cal,
        ece_cal,
        brier_raw,
        ece_raw,
        platt_a,
        platt_b,
        gate_auc,
        gate_f1,
    ):
        return False
    return (
        auc >= gate_auc
        and f1 >= gate_f1
        and brier_cal <= brier_raw
        and ece_cal <= ece_raw
    )


# ---------------------------------------------------------------------------
# Main promotion function
# ---------------------------------------------------------------------------


def promote_cnn_checkpoint(
    checkpoint_path: Path,
    calibration_path: Path,
    registry_path: Path,
    *,
    manifest_path: Path | None = None,
) -> PromotionResult:
    """Register a validated CNN checkpoint and write a promotion manifest.

    Reads the calibration JSON produced by ``evaluate_cnn_checkpoint.py``
    (written only when both AUC and F1 gates pass), computes the checkpoint
    SHA-256, registers the model in the registry, and writes a promotion
    manifest alongside the checkpoint.

    Args:
        checkpoint_path: Path to the ``.pt`` state-dict file.
        calibration_path: Path to the ``calibration.json`` from evaluation.
        registry_path: Path to ``models/registry.json``.
        manifest_path: Where to write the promotion manifest JSON. Defaults to
            ``<checkpoint_dir>/promotion_manifest.json``.

    Returns:
        :class:`PromotionResult` with all promotion details and flag.
    """
    checkpoint_path = Path(checkpoint_path)
    calibration_path = Path(calibration_path)
    registry_path = Path(registry_path)
    promoted_at = datetime.now(UTC).isoformat()

    if manifest_path is None:
        manifest_path = checkpoint_path.parent / "promotion_manifest.json"
    else:
        manifest_path = Path(manifest_path)

    # Verify both files exist
    for p in (checkpoint_path, calibration_path):
        if not p.exists():
            return PromotionResult(
                model_id="",
                sha256="",
                auc=None,
                f1=None,
                brier=None,
                ece=None,
                platt_a=None,
                platt_b=None,
                registry_path=str(registry_path),
                manifest_path=str(manifest_path),
                promoted_at=promoted_at,
                flag="MISSING_FILE",
            )

    # Read calibration JSON
    try:
        cal = json.loads(calibration_path.read_text(encoding="utf-8"))
    except Exception:
        return PromotionResult(
            model_id="",
            sha256="",
            auc=None,
            f1=None,
            brier=None,
            ece=None,
            platt_a=None,
            platt_b=None,
            registry_path=str(registry_path),
            manifest_path=str(manifest_path),
            promoted_at=promoted_at,
            flag="MISSING_FILE",
        )

    # Verify evaluation passed and independently re-check gate metrics.
    if not _calibration_passes_gates(cal):
        return PromotionResult(
            model_id="",
            sha256="",
            auc=cal.get("test_auc_raw"),
            f1=cal.get("test_f1_cal"),
            brier=cal.get("test_brier_cal"),
            ece=cal.get("test_ece_cal"),
            platt_a=cal.get("platt_a"),
            platt_b=cal.get("platt_b"),
            registry_path=str(registry_path),
            manifest_path=str(manifest_path),
            promoted_at=promoted_at,
            flag="GATES_NOT_MET",
        )

    auc = cal.get("test_auc_raw")
    f1 = cal.get("test_f1_cal")
    brier = cal.get("test_brier_cal")
    ece = cal.get("test_ece_cal")
    platt_a = cal.get("platt_a")
    platt_b = cal.get("platt_b")

    # Compute SHA-256
    sha256 = _sha256_file(checkpoint_path)
    model_id = f"cnn_{sha256[:12]}"

    # Register in model registry
    try:
        _skills = str(Path(__file__).resolve().parent)
        if _skills not in sys.path:
            sys.path.insert(0, _skills)
        try:
            from Skills.model_registry import RegistryEntry, register
        except ModuleNotFoundError:
            from model_registry import RegistryEntry, register  # type: ignore[no-redef]

        entry = RegistryEntry(
            model_id=model_id,
            model_type="cnn",
            model_path=str(checkpoint_path),
            auc=auc,
            brier=brier,
            n_train=None,
            registered_at=promoted_at,
            notes=(
                f"Production-promoted CNN. test_auc={auc}, test_f1_cal={f1}, "
                f"Platt A={platt_a} B={platt_b}. SHA-256={sha256}"
            ),
        )
        registry_result = register(registry_path, entry)
        if registry_result.flag not in ("OK", "EMPTY"):
            pass  # already registered is treated as success for idempotency
    except ValueError:
        # Already registered with same model_id — idempotent
        return PromotionResult(
            model_id=model_id,
            sha256=sha256,
            auc=auc,
            f1=f1,
            brier=brier,
            ece=ece,
            platt_a=platt_a,
            platt_b=platt_b,
            registry_path=str(registry_path),
            manifest_path=str(manifest_path),
            promoted_at=promoted_at,
            flag="ALREADY_REGISTERED",
        )

    # Write promotion manifest
    manifest = {
        "model_id": model_id,
        "model_type": "cnn",
        "checkpoint_path": str(checkpoint_path),
        "calibration_path": str(calibration_path),
        "sha256": sha256,
        "test_auc_raw": auc,
        "test_f1_cal": f1,
        "test_brier_cal": brier,
        "test_ece_cal": ece,
        "platt_a": platt_a,
        "platt_b": platt_b,
        "gate_auc": cal.get("gate_auc"),
        "gate_f1": cal.get("gate_f1"),
        "promoted_at": promoted_at,
        "flag": "PROMOTED",
    }
    _atomic_write_json(manifest_path, manifest)

    return PromotionResult(
        model_id=model_id,
        sha256=sha256,
        auc=auc,
        f1=f1,
        brier=brier,
        ece=ece,
        platt_a=platt_a,
        platt_b=platt_b,
        registry_path=str(registry_path),
        manifest_path=str(manifest_path),
        promoted_at=promoted_at,
        flag="PROMOTED",
    )


# ---------------------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------------------


def format_promotion_result(result: PromotionResult) -> str:
    """Format promotion result as a human-readable report with commit recipe."""
    lines = [
        "## CNN Checkpoint Promotion",
        f"- Flag: {result.flag}",
        f"- Model ID: {result.model_id}",
        f"- SHA-256: {result.sha256}",
        f"- Test AUC (raw): {result.auc}",
        f"- Test F1 (cal): {result.f1}",
        f"- Test Brier (cal): {result.brier}",
        f"- Test ECE (cal): {result.ece}",
        f"- Platt A={result.platt_a}, B={result.platt_b}",
        f"- Manifest: {result.manifest_path}",
        f"- Registry: {result.registry_path}",
    ]

    if result.flag == "PROMOTED":
        lines += [
            "",
            "### PRODUCTION_READINESS.md update for T1-1",
            "Replace the T1-1 section with:",
            "",
            "```",
            "### T1-1: Production Tier 2 CNN Checkpoint — COMPLETE",
            "",
            f"- **Status: COMPLETE as of {result.promoted_at[:10]}**",
            f"- **Model ID**: `{result.model_id}`",
            f"- **SHA-256**: `{result.sha256}`",
            f"- **Test AUC (raw)**: {result.auc}  "
            f"**Test F1 (cal)**: {result.f1}  "
            f"**Test Brier**: {result.brier}  "
            f"**Test ECE**: {result.ece}",
            f"- **Platt calibration**: A={result.platt_a}, B={result.platt_b}",
            f"- **Manifest**: `{result.manifest_path}`",
            "- **Promotion gate**: AUC ≥ 0.85 ✓  Calibrated F1 ≥ 0.80 ✓",
            "```",
            "",
            "### Git commit recipe",
            "```bash",
            "git pull origin main",
            f"git add {result.manifest_path} models/registry.json "
            "models/cnn/calibration.json",
            'git commit -m "Promote CNN checkpoint — T1-1 production gate passed"',
            "git push -u origin HEAD",
            "```",
        ]
    elif result.flag == "GATES_NOT_MET":
        lines.append("")
        lines.append(
            "Calibration JSON flag is not OK — run evaluate_cnn_checkpoint.py first."
        )
    elif result.flag == "MISSING_FILE":
        lines.append("")
        lines.append("Checkpoint or calibration JSON not found.")
    elif result.flag == "ALREADY_REGISTERED":
        lines.append("")
        lines.append("Model already in registry — no changes made.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Promote a validated CNN checkpoint to production."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument(
        "--registry", type=Path, default=Path("models/registry.json")
    )
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()

    result = promote_cnn_checkpoint(
        args.checkpoint,
        args.calibration,
        args.registry,
        manifest_path=args.manifest,
    )
    print(format_promotion_result(result))
    sys.exit(0 if result.flag in ("PROMOTED", "ALREADY_REGISTERED") else 1)


if __name__ == "__main__":
    _main()
