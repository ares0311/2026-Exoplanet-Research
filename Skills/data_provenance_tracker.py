"""Track data provenance for reproducibility of pipeline outputs.

Records the origin, version, and transformation chain for each artifact
so results can be traced back to their source data.

Public API
----------
ProvenanceRecord(artifact, source, version, transform_chain,
                 recorded_at, checksum, flag)
ProvenanceLog(path)
    .record(artifact, source, *, version, transform_chain) -> ProvenanceRecord
    .get(artifact) -> ProvenanceRecord | None
    .history(artifact) -> list[ProvenanceRecord]
    .summary() -> dict
format_provenance_report(log) -> str
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProvenanceRecord:
    artifact: str
    source: str
    version: str
    transform_chain: tuple[str, ...]
    recorded_at: float
    checksum: str | None   # MD5 of artifact file if it exists
    flag: str  # "OK" | "UNVERIFIED"


def _file_checksum(path: str) -> str | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        h = hashlib.md5()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


class ProvenanceLog:
    """Persistent provenance log backed by JSON."""

    def __init__(self, log_path: str | Path) -> None:
        self._path = Path(log_path)
        self._records: list[dict] = self._load()

    def _load(self) -> list[dict]:
        if not self._path.exists():
            return []
        try:
            return json.loads(self._path.read_text())
        except Exception:
            return []

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = None
        try:
            fd, tmp = tempfile.mkstemp(dir=self._path.parent, suffix=".tmp")
            with os.fdopen(fd, "w") as fh:
                json.dump(self._records, fh, indent=2)
            os.replace(tmp, self._path)
        except Exception:
            with contextlib.suppress(OSError):
                if tmp:
                    os.unlink(tmp)
            raise

    def record(
        self,
        artifact: str,
        source: str,
        *,
        version: str = "",
        transform_chain: list[str] | None = None,
    ) -> ProvenanceRecord:
        """Record provenance for an artifact.

        Args:
            artifact: Name or path of the output artifact.
            source: Origin data source description.
            version: Optional version string.
            transform_chain: List of transformation steps applied.

        Returns:
            ProvenanceRecord for the recorded entry.
        """
        checksum = _file_checksum(artifact)
        flag = "OK" if checksum is not None else "UNVERIFIED"
        entry = {
            "artifact": artifact,
            "source": source,
            "version": version,
            "transform_chain": transform_chain or [],
            "recorded_at": time.time(),
            "checksum": checksum,
            "flag": flag,
        }
        self._records.append(entry)
        self._save()
        return ProvenanceRecord(
            artifact=artifact,
            source=source,
            version=version,
            transform_chain=tuple(transform_chain or []),
            recorded_at=entry["recorded_at"],
            checksum=checksum,
            flag=flag,
        )

    def get(self, artifact: str) -> ProvenanceRecord | None:
        """Get the most recent provenance record for an artifact."""
        matches = [r for r in self._records if r["artifact"] == artifact]
        if not matches:
            return None
        r = matches[-1]
        return ProvenanceRecord(
            artifact=r["artifact"],
            source=r["source"],
            version=r.get("version", ""),
            transform_chain=tuple(r.get("transform_chain", [])),
            recorded_at=r["recorded_at"],
            checksum=r.get("checksum"),
            flag=r.get("flag", "UNVERIFIED"),
        )

    def history(self, artifact: str) -> list[ProvenanceRecord]:
        """Get all provenance records for an artifact."""
        return [
            ProvenanceRecord(
                artifact=r["artifact"],
                source=r["source"],
                version=r.get("version", ""),
                transform_chain=tuple(r.get("transform_chain", [])),
                recorded_at=r["recorded_at"],
                checksum=r.get("checksum"),
                flag=r.get("flag", "UNVERIFIED"),
            )
            for r in self._records
            if r["artifact"] == artifact
        ]

    def summary(self) -> dict:
        """Return summary statistics for the provenance log."""
        artifacts = {r["artifact"] for r in self._records}
        n_verified = sum(1 for r in self._records if r.get("flag") == "OK")
        return {
            "n_records": len(self._records),
            "n_artifacts": len(artifacts),
            "n_verified": n_verified,
        }


def format_provenance_report(log: ProvenanceLog) -> str:
    """Format provenance log as Markdown.

    Args:
        log: ProvenanceLog to format.

    Returns:
        Markdown string.
    """
    s = log.summary()
    lines = [
        "## Data Provenance Report\n",
        f"**Records**: {s['n_records']} | "
        f"**Artifacts**: {s['n_artifacts']} | "
        f"**Verified**: {s['n_verified']}\n",
    ]
    artifacts = {r["artifact"] for r in log._records}
    if not artifacts:
        lines.append("\n_No provenance records._")
        return "\n".join(lines)

    lines += [
        "",
        "| Artifact | Source | Version | Steps | Checksum | Flag |",
        "|---|---|---|---|---|---|",
    ]
    for artifact in sorted(artifacts):
        rec = log.get(artifact)
        if rec is None:
            continue
        chk = rec.checksum[:8] if rec.checksum else "—"
        steps = len(rec.transform_chain)
        lines.append(
            f"| {Path(rec.artifact).name} | {rec.source[:30]} | "
            f"{rec.version or '—'} | {steps} | `{chk}` | `{rec.flag}` |"
        )
    return "\n".join(lines)


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Track data provenance.")
    sub = parser.add_subparsers(dest="cmd")

    rec_p = sub.add_parser("record")
    rec_p.add_argument("--log", required=True)
    rec_p.add_argument("--artifact", required=True)
    rec_p.add_argument("--source", required=True)
    rec_p.add_argument("--version", default="")
    rec_p.add_argument("--steps", nargs="*", default=[])

    rep_p = sub.add_parser("report")
    rep_p.add_argument("--log", required=True)

    args = parser.parse_args(argv)
    if args.cmd == "record":
        plog = ProvenanceLog(args.log)
        r = plog.record(args.artifact, args.source, version=args.version,
                        transform_chain=args.steps)
        print(f"Recorded: {r.artifact} ({r.flag})")
    elif args.cmd == "report":
        plog = ProvenanceLog(args.log)
        print(format_provenance_report(plog))
    else:
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
