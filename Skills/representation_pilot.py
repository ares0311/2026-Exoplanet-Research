"""Leakage-safe masked-reconstruction pilot for Kepler light-curve embeddings.

This is a bounded Phase 3 benchmark, not a production model trainer.  It
pretrains only on the predefined training split with labels hidden from the
reconstruction objective, freezes the encoder, fits a linear probe, and opens
the predefined test split exactly once for final comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from Skills.run_report import RunReport, report_path_for, run_and_commit_report  # noqa: E402
from Skills.train_cnn import _compute_auc  # noqa: E402

ReportFn = Callable[..., bool]


@dataclass(frozen=True)
class PilotConfig:
    """Validated settings for the bounded representation pilot."""

    n_bins: int = 201
    d_model: int = 32
    n_heads: int = 4
    n_layers: int = 2
    dim_feedforward: int = 64
    dropout: float = 0.1
    mask_fraction: float = 0.2
    batch_size: int = 128
    pretrain_epochs: int = 12
    probe_epochs: int = 80
    learning_rate: float = 0.001
    probe_learning_rate: float = 0.03
    weight_decay: float = 0.0001
    patience: int = 4
    probe_patience: int = 12
    seed: int = 42
    device: str = "auto"
    top_k: int = 100


@dataclass(frozen=True)
class SplitRows:
    """One predefined split loaded from the master JSONL corpus."""

    flux: tuple[tuple[float, ...], ...]
    labels: tuple[int, ...]
    groups: tuple[str, ...]
    tabular: tuple[tuple[float, ...], ...]


def load_config(path: Path) -> PilotConfig:
    """Load and validate a pilot configuration."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    config = PilotConfig(**raw)
    if config.n_bins < 8 or config.d_model < 4:
        raise ValueError("n_bins must be >= 8 and d_model must be >= 4")
    if config.d_model % config.n_heads:
        raise ValueError("d_model must be divisible by n_heads")
    if not 0.0 < config.mask_fraction < 1.0:
        raise ValueError("mask_fraction must be between zero and one")
    if config.device not in {"auto", "cpu", "mps", "cuda"}:
        raise ValueError("device must be auto, cpu, mps, or cuda")
    for name in ("batch_size", "pretrain_epochs", "probe_epochs", "patience", "top_k"):
        if int(getattr(config, name)) < 1:
            raise ValueError(f"{name} must be positive")
    return config


def _tabular_features(row: dict[str, Any], flux: tuple[float, ...]) -> tuple[float, ...]:
    """Return a small BLS-metadata/statistical baseline without label leakage."""
    ordered = sorted(flux)
    n = len(ordered)
    mean = sum(flux) / n
    variance = sum((value - mean) ** 2 for value in flux) / n
    return (
        math.log1p(max(float(row["period_days"]), 0.0)),
        math.log1p(max(float(row["duration_hours"]), 0.0)),
        mean,
        math.sqrt(variance),
        ordered[max(0, int(0.01 * (n - 1)))],
        ordered[max(0, int(0.05 * (n - 1)))],
        min(flux),
    )


def load_predefined_splits(path: Path, *, n_bins: int) -> dict[str, SplitRows]:
    """Load train/val/test rows and fail closed on group leakage."""
    buckets: dict[str, dict[str, list[Any]]] = {
        name: {"flux": [], "labels": [], "groups": [], "tabular": []}
        for name in ("train", "val", "test")
    }
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            split = str(row.get("split", ""))
            if split not in buckets:
                raise ValueError(f"line {line_number}: invalid predefined split {split!r}")
            flux = tuple(float(value) for value in row["flux"])
            if len(flux) != n_bins or not all(math.isfinite(value) for value in flux):
                raise ValueError(f"line {line_number}: flux must contain {n_bins} finite bins")
            label = int(row["label"])
            if label not in (0, 1):
                raise ValueError(f"line {line_number}: label must be zero or one")
            group = str(row["group_key"])
            if not group:
                raise ValueError(f"line {line_number}: group_key is required")
            bucket = buckets[split]
            bucket["flux"].append(flux)
            bucket["labels"].append(label)
            bucket["groups"].append(group)
            bucket["tabular"].append(_tabular_features(row, flux))

    group_sets = {name: set(values["groups"]) for name, values in buckets.items()}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = group_sets[left] & group_sets[right]
        if overlap:
            raise ValueError(f"group leakage between {left} and {right}: {sorted(overlap)[:3]}")
    if any(not buckets[name]["flux"] for name in buckets):
        raise ValueError("train, val, and test must all be non-empty")
    return {
        name: SplitRows(
            flux=tuple(values["flux"]),
            labels=tuple(values["labels"]),
            groups=tuple(values["groups"]),
            tabular=tuple(values["tabular"]),
        )
        for name, values in buckets.items()
    }


def _resolve_device(requested: str):  # noqa: ANN201
    import torch

    mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    cuda = bool(torch.cuda.is_available())
    if requested == "auto":
        return torch.device("mps" if mps else "cuda" if cuda else "cpu")
    if requested == "mps" and not mps:
        print("Warning: MPS unavailable; falling back to CPU.", flush=True)
        return torch.device("cpu")
    if requested == "cuda" and not cuda:
        print("Warning: CUDA unavailable; falling back to CPU.", flush=True)
        return torch.device("cpu")
    return torch.device(requested)


def _build_encoder(config: PilotConfig):  # noqa: ANN201
    import torch
    from torch import nn

    class MaskedLightCurveEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_projection = nn.Linear(1, config.d_model)
            self.position = nn.Parameter(torch.empty(1, config.n_bins, config.d_model))
            self.mask_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
            layer = nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.n_heads,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout,
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, config.n_layers)
            self.decoder = nn.Linear(config.d_model, 1)
            nn.init.normal_(self.position, std=0.02)
            # TransformerEncoder clones layers, so reinitialize every matrix explicitly.
            for parameter in self.encoder.parameters():
                if parameter.dim() > 1:
                    nn.init.xavier_uniform_(parameter)

        def encode(self, x, mask=None):  # noqa: ANN001, ANN201
            hidden = self.input_projection(x.unsqueeze(-1)) + self.position
            if mask is not None:
                hidden = torch.where(mask.unsqueeze(-1), self.mask_token + self.position, hidden)
            return self.encoder(hidden)

        def forward(self, x, mask):  # noqa: ANN001, ANN201
            return self.decoder(self.encode(x, mask)).squeeze(-1)

        def embed(self, x):  # noqa: ANN001, ANN201
            return self.encode(x).mean(dim=1)

    torch.manual_seed(config.seed)
    return MaskedLightCurveEncoder()


def _batches(values, batch_size: int, *, shuffle: bool, seed: int):  # noqa: ANN001, ANN201
    indices = list(range(len(values)))
    if shuffle:
        random.Random(seed).shuffle(indices)
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        yield [values[index] for index in batch_indices]


def _masked_loss(model, values, config: PilotConfig, device, *, train: bool, seed: int):  # noqa: ANN001, ANN201
    import torch

    losses: list[float] = []
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for rows in _batches(values, config.batch_size, shuffle=train, seed=seed):
            x = torch.tensor(rows, dtype=torch.float32, device=device)
            generator = torch.Generator(device="cpu").manual_seed(seed + len(losses))
            mask = torch.rand(x.shape, generator=generator).to(device) < config.mask_fraction
            mask[:, 0] = True
            reconstructed = model(x, mask)
            loss = ((reconstructed[mask] - x[mask]) ** 2).mean()
            losses.append(float(loss.detach().cpu()))
            if train:
                yield loss
    if not train:
        yield sum(losses) / len(losses)


def _extract_embeddings(model, rows: SplitRows, config: PilotConfig, device):  # noqa: ANN001, ANN201
    import torch

    output: list[list[float]] = []
    model.eval()
    with torch.no_grad():
        for values in _batches(rows.flux, config.batch_size, shuffle=False, seed=config.seed):
            x = torch.tensor(values, dtype=torch.float32, device=device)
            output.extend(model.embed(x).cpu().tolist())
    return output


def _standardize(train, val, test):  # noqa: ANN001, ANN201
    import torch

    train_tensor = torch.tensor(train, dtype=torch.float32)
    val_tensor = torch.tensor(val, dtype=torch.float32)
    test_tensor = torch.tensor(test, dtype=torch.float32)
    mean = train_tensor.mean(dim=0, keepdim=True)
    std = train_tensor.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (train_tensor - mean) / std, (val_tensor - mean) / std, (test_tensor - mean) / std


def _fit_probe(train_x, train_y, val_x, val_y, config: PilotConfig, device):  # noqa: ANN001, ANN201
    import torch
    from torch import nn

    torch.manual_seed(config.seed)
    model = nn.Linear(train_x.shape[1], 1).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.probe_learning_rate, weight_decay=config.weight_decay
    )
    criterion = nn.BCEWithLogitsLoss()
    best_auc = -1.0
    best_state: dict[str, Any] | None = None
    stale = 0
    train_x, train_y = train_x.to(device), torch.tensor(train_y, dtype=torch.float32, device=device)
    val_x = val_x.to(device)
    epochs_run = 0
    for _epoch in range(1, config.probe_epochs + 1):
        epochs_run += 1
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(train_x).squeeze(1), train_y)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            probabilities = torch.sigmoid(model(val_x).squeeze(1)).cpu().tolist()
        auc = _compute_auc(list(val_y), probabilities)
        if auc > best_auc + 1e-6:
            best_auc = auc
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= config.probe_patience:
            break
    if best_state is None:
        raise RuntimeError("linear probe produced no checkpoint")
    model.load_state_dict(best_state)
    return model, best_auc, epochs_run


def _evaluate_probe(model, x, labels, *, top_k: int, device):  # noqa: ANN001, ANN201
    import torch

    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(x.to(device)).squeeze(1)).cpu().tolist()
    k = min(top_k, len(labels))
    ranked = sorted(zip(probabilities, labels, strict=True), reverse=True)
    predictions = [int(value >= 0.5) for value in probabilities]
    tp = sum(pred == label == 1 for pred, label in zip(predictions, labels, strict=True))
    fp = sum(pred == 1 and label == 0 for pred, label in zip(predictions, labels, strict=True))
    fn = sum(pred == 0 and label == 1 for pred, label in zip(predictions, labels, strict=True))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "auc": _compute_auc(list(labels), probabilities),
        "f1_at_0_5": f1,
        "top_k": k,
        "top_k_positive_fraction": sum(label for _, label in ranked[:k]) / k,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_pilot(
    corpus_path: Path,
    config_path: Path,
    output_path: Path,
    checkpoint_path: Path,
    *,
    report_fn: ReportFn = run_and_commit_report,
) -> dict[str, Any]:
    """Train and evaluate the bounded representation pilot."""
    import torch

    started_at = datetime.now(UTC)
    started = time.monotonic()
    config = load_config(config_path)
    splits = load_predefined_splits(corpus_path, n_bins=config.n_bins)
    device = _resolve_device(config.device)
    print(
        "Representation pilot startup: "
        f"train={len(splits['train'].flux)} val={len(splits['val'].flux)} "
        f"test={len(splits['test'].flux)} batch={config.batch_size} "
        f"epochs={config.pretrain_epochs} patience={config.patience} device={device}",
        flush=True,
    )
    model = _build_encoder(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    best_val = math.inf
    best_state: dict[str, Any] | None = None
    best_epoch = 0
    stale = 0
    for epoch in range(1, config.pretrain_epochs + 1):
        model.train()
        train_losses: list[float] = []
        for loss in _masked_loss(
            model, splits["train"].flux, config, device, train=True, seed=config.seed + epoch
        ):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        val_loss = float(next(_masked_loss(
            model, splits["val"].flux, config, device, train=False, seed=config.seed
        )))
        train_loss = sum(train_losses) / len(train_losses)
        improved = val_loss < best_val - 1e-6
        if improved:
            best_val = val_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        elapsed = time.monotonic() - started
        rate = epoch / elapsed if elapsed else 0.0
        eta = (config.pretrain_epochs - epoch) / rate if rate else 0.0
        marker = "best" if improved else f"patience {stale}/{config.patience}"
        print(
            f"Epoch {epoch:2d}/{config.pretrain_epochs} train={train_loss:.6f} "
            f"val={val_loss:.6f} lr={config.learning_rate:.2e} {marker} ETA={eta:.0f}s",
            flush=True,
        )
        if stale >= config.patience:
            print(f"Early stop at epoch {epoch}; best epoch={best_epoch}.", flush=True)
            break
    if best_state is None:
        raise RuntimeError("masked pretraining produced no checkpoint")
    model.load_state_dict(best_state)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "config": asdict(config)}, checkpoint_path)

    embeddings = {
        name: _extract_embeddings(model, rows, config, device) for name, rows in splits.items()
    }
    emb_train, emb_val, emb_test = _standardize(
        embeddings["train"], embeddings["val"], embeddings["test"]
    )
    tab_train, tab_val, tab_test = _standardize(
        splits["train"].tabular, splits["val"].tabular, splits["test"].tabular
    )
    embedding_probe, embedding_val_auc, embedding_probe_epochs = _fit_probe(
        emb_train, splits["train"].labels, emb_val, splits["val"].labels, config, device
    )
    tabular_probe, tabular_val_auc, tabular_probe_epochs = _fit_probe(
        tab_train, splits["train"].labels, tab_val, splits["val"].labels, config, device
    )
    embedding_test = _evaluate_probe(
        embedding_probe, emb_test, splits["test"].labels, top_k=config.top_k, device=device
    )
    tabular_test = _evaluate_probe(
        tabular_probe, tab_test, splits["test"].labels, top_k=config.top_k, device=device
    )
    benchmark_auc = 0.957211
    result: dict[str, Any] = {
        "schema_version": 1,
        "benchmark_id": "kepler_masked_embedding_pilot_v1",
        "status": "pass" if embedding_test["auc"] > benchmark_auc else "does_not_beat_cnn",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "dataset_ids": {
            "pretrain": "t1_1_kepler_master_train",
            "validation": "t1_1_kepler_master_validation",
            "frozen_eval": "t1_1_kepler_master_frozen_eval",
        },
        "corpus_path": str(corpus_path),
        "corpus_sha256": _sha256(corpus_path),
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "resolved_device": str(device),
        "split_counts": {name: len(rows.labels) for name, rows in splits.items()},
        "pretraining": {"best_epoch": best_epoch, "best_val_masked_mse": best_val},
        "embedding_linear_probe": {
            "validation_auc": embedding_val_auc,
            "probe_epochs": embedding_probe_epochs,
            "test": embedding_test,
        },
        "tabular_linear_probe": {
            "features": [
                "log_period",
                "log_duration",
                "flux_mean",
                "flux_std",
                "flux_p01",
                "flux_p05",
                "flux_min",
            ],
            "validation_auc": tabular_val_auc,
            "probe_epochs": tabular_probe_epochs,
            "test": tabular_test,
        },
        "benchmark_cnn_v1": {"test_auc": benchmark_auc, "test_f1": 0.834688},
        "gate": "embedding test AUC must strictly exceed benchmark_cnn_v1 test AUC",
        "limitations": [
            "Pilot uses labeled Kepler snippets as unlabeled inputs; it is not broad "
            "unlabeled Kepler/TESS pretraining.",
            "No stellar-variability label benchmark is available in this corpus.",
            "No embedding-based injection-recovery comparison is included.",
            "The test split is opened once for this versioned pilot and must not be "
            "used for tuning.",
            "This pilot cannot promote or replace benchmark_cnn_v1 regardless of outcome.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    elapsed = time.monotonic() - started
    report = RunReport(
        script="representation_pilot",
        status="success",
        started_at=started_at.isoformat(),
        completed_at=datetime.now(UTC).isoformat(),
        elapsed_seconds=elapsed,
        items_processed=sum(len(rows.labels) for rows in splits.values()),
        items_written=1,
        output_paths=(str(output_path), str(checkpoint_path)),
        notes=f"bounded Phase 3 pilot; outcome={result['status']}",
    )
    report_path = report_path_for("representation_pilot")
    if not report_fn(report, report_path):
        print(f"WARNING: Run Report push failed for {report_path}", flush=True)
    print(
        f"Representation pilot COMPLETE: outcome={result['status']} "
        f"embedding_auc={embedding_test['auc']:.6f} tabular_auc={tabular_test['auc']:.6f} "
        f"elapsed={elapsed:.1f}s output={output_path}",
        flush=True,
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("data/processed/t1_1_kepler_master_combined.jsonl"),
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/representation_pilot_v1.json")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/manifests/representation_pilot_v1.json"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/representation_pilot_v1/best.pt"),
    )
    args = parser.parse_args(argv)
    run_pilot(args.corpus, args.config, args.output, args.checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
