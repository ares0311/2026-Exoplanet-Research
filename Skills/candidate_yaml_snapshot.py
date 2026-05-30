"""Serialize a candidate dict to a compact YAML snapshot for archiving.

Public API:
    YamlSnapshotResult  -- frozen dataclass
    build_yaml_snapshot(candidate, *, fields) -> YamlSnapshotResult
    write_yaml_snapshot(result, path) -> str
    format_yaml_preview(result) -> str
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

_DEFAULT_FIELDS = [
    "tic_id", "period_days", "epoch_bjd", "depth_ppm", "duration_hours",
    "fpp", "snr", "pathway", "disposition", "run_at",
]


@dataclass(frozen=True)
class YamlSnapshotResult:
    tic_id: str
    n_fields: int
    yaml_text: str
    flag: str


def _to_yaml_value(v: object) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    if v is None:
        return "null"
    return f'"{v}"'


def build_yaml_snapshot(
    candidate: dict[str, object],
    *,
    fields: list[str] | None = None,
) -> YamlSnapshotResult:
    tic_id = str(candidate.get("tic_id", "unknown"))
    use_fields = fields if fields is not None else _DEFAULT_FIELDS
    lines = ["---"]
    n_fields = 0
    for key in use_fields:
        if key in candidate:
            lines.append(f"{key}: {_to_yaml_value(candidate[key])}")
            n_fields += 1
    yaml_text = "\n".join(lines)
    flag = "OK" if n_fields > 0 else "NO_FIELDS"
    return YamlSnapshotResult(tic_id=tic_id, n_fields=n_fields, yaml_text=yaml_text, flag=flag)


def write_yaml_snapshot(result: YamlSnapshotResult, path: str | Path) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(result.yaml_text + "\n")
    return str(p)


def format_yaml_preview(result: YamlSnapshotResult) -> str:
    lines = [
        f"## YAML Snapshot — TIC {result.tic_id}",
        "",
        f"**Fields:** {result.n_fields}  **Flag:** {result.flag}",
        "",
        "```yaml",
        result.yaml_text,
        "```",
    ]
    return "\n".join(lines)


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Build a YAML candidate snapshot.")
    parser.add_argument("candidate_file", help="JSON file with candidate dict.")
    parser.add_argument("--output", default=None, help="Output YAML file path.")
    args = parser.parse_args()
    with open(args.candidate_file) as fh:
        candidate = json.load(fh)
    result = build_yaml_snapshot(candidate)
    if args.output:
        write_yaml_snapshot(result, args.output)
    print(format_yaml_preview(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
