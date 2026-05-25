"""Compare two config JSON files field-by-field and report changes.

Recurses into nested dicts using dot-separated key paths. Non-dict values
are treated as leaf nodes.

Public API
----------
ConfigDiffEntry(key, old_value, new_value, change_type)
ConfigDiffResult(n_added, n_removed, n_changed, n_unchanged, entries, flag)
diff_configs(old, new, *, prefix, include_unchanged) -> ConfigDiffResult
load_and_diff_configs(old_path, new_path, **kwargs) -> ConfigDiffResult
format_config_diff(result) -> str
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ConfigDiffEntry:
    key: str
    old_value: object
    new_value: object
    change_type: str  # "added" | "removed" | "changed" | "unchanged"


@dataclass(frozen=True)
class ConfigDiffResult:
    n_added: int
    n_removed: int
    n_changed: int
    n_unchanged: int
    entries: tuple[ConfigDiffEntry, ...]
    flag: str  # "OK" | "NO_CHANGE" | "INVALID"


def _flatten(d: dict, prefix: str = "") -> dict[str, object]:
    """Recursively flatten a dict with dot-separated keys."""
    out: dict[str, object] = {}
    for k, v in d.items():
        full_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, prefix=full_key))
        else:
            out[full_key] = v
    return out


def diff_configs(
    old: dict,
    new: dict,
    *,
    prefix: str = "",
    include_unchanged: bool = False,
) -> ConfigDiffResult:
    """Compare two config dicts field-by-field.

    Args:
        old: Older config dict.
        new: Newer config dict.
        prefix: Optional dot-prefix to prepend all keys (for sub-dict recursion).
        include_unchanged: If True, include unchanged fields in entries.

    Returns:
        :class:`ConfigDiffResult`.
    """
    if not isinstance(old, dict) or not isinstance(new, dict):
        return ConfigDiffResult(0, 0, 0, 0, (), "INVALID")

    try:
        old_flat = _flatten(old, prefix=prefix)
        new_flat = _flatten(new, prefix=prefix)
    except Exception:
        return ConfigDiffResult(0, 0, 0, 0, (), "INVALID")

    entries: list[ConfigDiffEntry] = []
    n_added = n_removed = n_changed = n_unchanged = 0

    all_keys = sorted(set(old_flat) | set(new_flat))
    for key in all_keys:
        if key in old_flat and key not in new_flat:
            n_removed += 1
            entries.append(ConfigDiffEntry(
                key=key,
                old_value=old_flat[key],
                new_value=None,
                change_type="removed",
            ))
        elif key not in old_flat and key in new_flat:
            n_added += 1
            entries.append(ConfigDiffEntry(
                key=key,
                old_value=None,
                new_value=new_flat[key],
                change_type="added",
            ))
        else:
            old_v = old_flat[key]
            new_v = new_flat[key]
            if old_v == new_v:
                n_unchanged += 1
                if include_unchanged:
                    entries.append(ConfigDiffEntry(
                        key=key,
                        old_value=old_v,
                        new_value=new_v,
                        change_type="unchanged",
                    ))
            else:
                n_changed += 1
                entries.append(ConfigDiffEntry(
                    key=key,
                    old_value=old_v,
                    new_value=new_v,
                    change_type="changed",
                ))

    flag = "NO_CHANGE" if n_added == 0 and n_removed == 0 and n_changed == 0 else "OK"

    return ConfigDiffResult(
        n_added=n_added,
        n_removed=n_removed,
        n_changed=n_changed,
        n_unchanged=n_unchanged,
        entries=tuple(entries),
        flag=flag,
    )


def load_and_diff_configs(
    old_path: str | Path,
    new_path: str | Path,
    **kwargs,
) -> ConfigDiffResult:
    """Load two JSON config files and diff them."""
    try:
        old = json.loads(Path(old_path).read_text())
        new = json.loads(Path(new_path).read_text())
    except Exception:
        return ConfigDiffResult(0, 0, 0, 0, (), "INVALID")
    if not isinstance(old, dict) or not isinstance(new, dict):
        return ConfigDiffResult(0, 0, 0, 0, (), "INVALID")
    return diff_configs(old, new, **kwargs)


def format_config_diff(result: ConfigDiffResult) -> str:
    """Format config diff result as Markdown."""
    lines = [
        "## Config Diff Tool",
        "",
        f"- Added: {result.n_added}",
        f"- Removed: {result.n_removed}",
        f"- Changed: {result.n_changed}",
        f"- Unchanged: {result.n_unchanged}",
        f"- **Flag: {result.flag}**",
        "",
    ]
    non_unchanged = [e for e in result.entries if e.change_type != "unchanged"]
    if non_unchanged:
        lines.append("### Changes")
        lines.append("")
        lines.append("| Key | Old | New | Type |")
        lines.append("|-----|-----|-----|------|")
        for e in non_unchanged:
            lines.append(
                f"| {e.key} | {e.old_value} | {e.new_value} | {e.change_type} |"
            )
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="config_diff_tool",
        description="Compare two JSON config files field-by-field.",
    )
    parser.add_argument("old_path")
    parser.add_argument("new_path")
    parser.add_argument("--include-unchanged", action="store_true")
    args = parser.parse_args(argv)

    result = load_and_diff_configs(
        args.old_path,
        args.new_path,
        include_unchanged=args.include_unchanged,
    )
    print(format_config_diff(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
