"""Configuration loading for background search automation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path("configs/background_search_v0.json")
ALLOWED_REPORT_FORMATS = {"markdown", "html"}


class ConfigError(ValueError):
    """Raised when a background search config is missing required safe settings."""


@dataclass(frozen=True)
class BackgroundConfig:
    path: Path
    payload: dict[str, Any]
    fingerprint: str

    @property
    def version(self) -> str:
        return str(self.payload["version"])

    @property
    def schema_version(self) -> str:
        return str(self.payload["schema_version"])

    @property
    def target_pool(self) -> str:
        return str(self.payload["target_pool"])

    @property
    def priority_weights(self) -> dict[str, float]:
        return {key: float(value) for key, value in self.payload["priority_weights"].items()}

    @property
    def thresholds(self) -> dict[str, float]:
        return {key: float(value) for key, value in self.payload["thresholds"].items()}

    @property
    def scheduler(self) -> dict[str, float]:
        return {key: float(value) for key, value in self.payload["scheduler"].items()}

    @property
    def reports(self) -> dict[str, Any]:
        return dict(self.payload["reports"])

    def ledger_config(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "config_version": self.version,
            "config_fingerprint": self.fingerprint,
            "target_pool": self.target_pool,
            "network_access": bool(self.payload["network_access"]),
            "live_query_mode": bool(self.payload["live_query_mode"]),
            "approval_required_for_external_submission": bool(
                self.payload["approval_required_for_external_submission"]
            ),
        }


def load_background_config(path: Path = DEFAULT_CONFIG_PATH) -> BackgroundConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_background_config(payload)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    fingerprint = hashlib.sha256(canonical).hexdigest()
    return BackgroundConfig(path=path, payload=payload, fingerprint=fingerprint)


def validate_background_config(payload: dict[str, Any]) -> None:
    required_top_level = {
        "version",
        "schema_version",
        "target_pool",
        "network_access",
        "live_query_mode",
        "approval_required_for_external_submission",
        "priority_weights",
        "thresholds",
        "scheduler",
        "reports",
    }
    missing = sorted(required_top_level - set(payload))
    if missing:
        raise ConfigError(f"Missing config keys: {', '.join(missing)}")
    if payload["network_access"] is not False:
        raise ConfigError("Background search config must default network_access to false.")
    if payload["live_query_mode"] is not False:
        raise ConfigError("Background search config must default live_query_mode to false.")
    if payload["approval_required_for_external_submission"] is not True:
        raise ConfigError("External submission must require human approval.")

    required_weights = {
        "scientific_interest",
        "data_completeness",
        "false_positive_risk_inverse",
        "follow_up_feasibility",
        "calibration_confidence",
        "blocking_issue_inverse",
        "prior_review_penalty",
        "never_reviewed_boost",
    }
    _validate_numeric_map(
        payload["priority_weights"], required_weights, "priority_weights", 0.0, 1.0
    )

    required_thresholds = {
        "follow_up_priority_score",
        "low_false_positive_risk",
        "calibration_confidence_min",
        "blocking_issue_penalty_high",
    }
    _validate_numeric_map(payload["thresholds"], required_thresholds, "thresholds", 0.0, 1.0)

    _validate_numeric_map(
        payload["scheduler"], {"lock_wait_seconds", "lock_poll_seconds"}, "scheduler", 0.0, 60.0
    )
    if float(payload["scheduler"]["lock_poll_seconds"]) <= 0:
        raise ConfigError("scheduler.lock_poll_seconds must be positive.")

    reports = payload["reports"]
    if not isinstance(reports, dict):
        raise ConfigError("reports must be an object.")
    if not isinstance(reports.get("export_dir"), str) or not reports["export_dir"]:
        raise ConfigError("reports.export_dir must be a non-empty string.")
    formats = reports.get("formats")
    if not isinstance(formats, list) or not formats:
        raise ConfigError("reports.formats must be a non-empty list.")
    unknown_formats = sorted(set(formats) - ALLOWED_REPORT_FORMATS)
    if unknown_formats:
        raise ConfigError(f"Unknown report formats: {', '.join(unknown_formats)}")


def _validate_numeric_map(
    value: Any, required_keys: set[str], section: str, lower: float, upper: float
) -> None:
    if not isinstance(value, dict):
        raise ConfigError(f"{section} must be an object.")
    missing = sorted(required_keys - set(value))
    if missing:
        raise ConfigError(f"Missing {section} keys: {', '.join(missing)}")
    for key in required_keys:
        item = value[key]
        if not isinstance(item, int | float):
            raise ConfigError(f"{section}.{key} must be numeric.")
        if not lower <= float(item) <= upper:
            raise ConfigError(f"{section}.{key} must be between {lower} and {upper}.")
