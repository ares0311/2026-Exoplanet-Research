"""Centralised YAML/JSON pipeline configuration loader with schema validation.

Loads pipeline parameters from a YAML or JSON file, validates required keys
and value ranges, and provides typed access.  Supports dot-notation access
and environment-variable overrides (``EXO_<KEY>``).

Public API
----------
PipelineConfig — dict-like config with attribute access
load_config(path, *, env_prefix) -> PipelineConfig
validate_config(cfg) -> list[str]   # list of validation errors (empty = OK)
default_config() -> PipelineConfig
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_DEFAULTS: dict[str, Any] = {
    "mission":       "TESS",
    "min_snr":       5.0,
    "max_peaks":     5,
    "scorer":        "bayesian",
    "model_path":    None,
    "calibration_path": None,
    "cache_dir":     "data/lc_cache",
    "log_dir":       "logs",
    "output_dir":    "results",
    "tmag_min":      10.0,
    "tmag_max":      14.0,
    "n_targets":     100,
    "period_min":    0.5,
    "period_max":    27.0,
    "sigma_clip":    4.0,
    "flatten_window": 101,
    "scan_log":      "data/scan_log.json",
}

_REQUIRED_KEYS: tuple[str, ...] = ("mission", "min_snr", "scorer")

_VALID_MISSIONS = {"TESS", "Kepler", "K2"}
_VALID_SCORERS  = {"bayesian", "xgboost", "ensemble"}

_RANGE_CHECKS: list[tuple[str, float, float]] = [
    ("min_snr",       0.1,  100.0),
    ("max_peaks",       1,   50),
    ("tmag_min",      6.0,  20.0),
    ("tmag_max",      6.0,  20.0),
    ("n_targets",       1, 10000),
    ("period_min",    0.1,  100.0),
    ("period_max",    0.1, 1000.0),
    ("sigma_clip",    1.0,   10.0),
    ("flatten_window", 5,   2001),
]


class PipelineConfig:
    """Dict-like config object with attribute access and env-var overrides.

    Args:
        data: Configuration dictionary.
    """

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = dict(data)

    # ------------------------------------------------------------------
    # Dict-like interface
    # ------------------------------------------------------------------

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def keys(self) -> Any:
        return self._data.keys()

    def items(self) -> Any:
        return self._data.items()

    def as_dict(self) -> dict[str, Any]:
        return dict(self._data)

    # ------------------------------------------------------------------
    # Attribute access
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self._data[name]
        except KeyError:
            raise AttributeError(f"PipelineConfig has no attribute '{name}'") from None

    def __repr__(self) -> str:
        return f"PipelineConfig({self._data!r})"


def _parse_file(path: Path) -> dict[str, Any]:
    text = path.read_text()
    if path.suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore[import]
            return yaml.safe_load(text) or {}
        except ImportError:
            # Minimal YAML subset: key: value lines
            data: dict[str, Any] = {}
            for line in text.splitlines():
                line = line.split("#", 1)[0].strip()
                if ":" not in line:
                    continue
                k, _, v = line.partition(":")
                data[k.strip()] = _coerce(v.strip())
            return data
    else:
        return json.loads(text)


def _coerce(v: str) -> Any:
    """Convert a YAML-style string to Python type."""
    if v.lower() == "null" or v == "~" or v == "":
        return None
    if v.lower() == "true":
        return True
    if v.lower() == "false":
        return False
    try:
        if "." in v:
            return float(v)
        return int(v)
    except ValueError:
        return v.strip("'\"")


def load_config(
    path: Path | str,
    *,
    env_prefix: str = "EXO",
) -> PipelineConfig:
    """Load a YAML or JSON config file and apply environment-variable overrides.

    Environment variables of the form ``EXO_<KEY>`` (upper-case) override the
    corresponding config key.

    Args:
        path: Path to config file.
        env_prefix: Prefix for environment variable overrides.

    Returns:
        :class:`PipelineConfig`.
    """
    data = dict(_DEFAULTS)
    data.update(_parse_file(Path(path)))

    # Apply env-var overrides
    prefix = env_prefix.upper() + "_"
    for env_key, env_val in os.environ.items():
        if env_key.startswith(prefix):
            cfg_key = env_key[len(prefix):].lower()
            data[cfg_key] = _coerce(env_val)

    return PipelineConfig(data)


def validate_config(cfg: PipelineConfig) -> list[str]:
    """Validate a PipelineConfig; return list of error strings (empty = OK)."""
    errors: list[str] = []

    for key in _REQUIRED_KEYS:
        if key not in cfg or cfg[key] is None:
            errors.append(f"Missing required key: '{key}'")

    mission = cfg.get("mission", "")
    if mission not in _VALID_MISSIONS:
        errors.append(f"Invalid mission '{mission}'; must be one of {_VALID_MISSIONS}")

    scorer = cfg.get("scorer", "")
    if scorer not in _VALID_SCORERS:
        errors.append(f"Invalid scorer '{scorer}'; must be one of {_VALID_SCORERS}")

    for key, lo, hi in _RANGE_CHECKS:
        val = cfg.get(key)
        if val is not None:
            try:
                v = float(val)
                if not (lo <= v <= hi):
                    errors.append(f"'{key}' = {v} out of range [{lo}, {hi}]")
            except (TypeError, ValueError):
                errors.append(f"'{key}' must be numeric, got {val!r}")

    if "tmag_min" in cfg and "tmag_max" in cfg:
        lo_t = cfg.get("tmag_min", 0)
        hi_t = cfg.get("tmag_max", 0)
        try:
            if float(lo_t) >= float(hi_t):
                errors.append(f"tmag_min ({lo_t}) must be < tmag_max ({hi_t})")
        except (TypeError, ValueError):
            pass

    return errors


def default_config() -> PipelineConfig:
    """Return a PipelineConfig with all default values."""
    return PipelineConfig(dict(_DEFAULTS))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(
        prog="config_manager",
        description="Load, validate, and display pipeline configuration.",
    )
    parser.add_argument("config", nargs="?", default=None,
                        help="Path to YAML/JSON config file.")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--dump-defaults", action="store_true")
    args = parser.parse_args(argv)

    if args.dump_defaults:
        print(json.dumps(_DEFAULTS, indent=2))
        return 0

    if args.config is None:
        parser.print_help()
        return 1

    cfg = load_config(args.config)
    errors = validate_config(cfg)

    if args.validate:
        if errors:
            print("Config errors:")
            for e in errors:
                print(f"  - {e}")
            return 1
        print("Config is valid.")
        return 0

    print(json.dumps(cfg.as_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
