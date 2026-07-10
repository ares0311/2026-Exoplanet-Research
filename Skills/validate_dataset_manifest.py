"""Validate one or more versioned dataset manifests and their local artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from exo_toolkit.dataset_manifest import validate_dataset_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", nargs="+", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--skip-checksum",
        action="store_true",
        help="Validate the contract and local path without hashing the artifact.",
    )
    args = parser.parse_args(argv)

    results = [
        validate_dataset_manifest(
            path,
            repo_root=args.repo_root,
            verify_checksum=not args.skip_checksum,
        )
        for path in args.manifests
    ]
    for result in results:
        print(json.dumps(result.model_dump(mode="json"), sort_keys=True), flush=True)
    return 0 if all(result.ok for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
