"""Track the evolution of a candidate across multiple pipeline runs over time.

Records FPP, period, pathway, and other key metrics per run and exposes trend
analysis and Markdown rendering.

Public API
----------
TimelineEntry  (dataclass)
CandidateTimeline(path)
    .record(row, *, note)
    .entries() -> list[TimelineEntry]
    .latest() -> TimelineEntry | None
    .summary() -> dict
    .to_markdown() -> str
"""
from __future__ import annotations

import datetime
import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class TimelineEntry:
    run_at: str           # ISO-8601 UTC
    period_days: float
    fpp: float
    planet_posterior: float
    pathway: str
    scorer: str
    note: str = ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class CandidateTimeline:
    """Persistent per-candidate timeline stored as a JSON file per candidate.

    Args:
        path: Directory that holds per-candidate JSON files.
    """

    def __init__(self, path: Path) -> None:
        self._dir = Path(path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _candidate_path(self, candidate_id: str) -> Path:
        return self._dir / f"{candidate_id}.json"

    def _load(self, candidate_id: str) -> list[dict[str, Any]]:
        p = self._candidate_path(candidate_id)
        if not p.exists():
            return []
        return json.loads(p.read_text())

    def _save(self, candidate_id: str, data: list[dict[str, Any]]) -> None:
        p = self._candidate_path(candidate_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write so a crash never leaves a partial file.
        fd, tmp = tempfile.mkstemp(dir=p.parent, suffix=".tmp")
        try:
            with open(fd, "w") as f:
                json.dump(data, f, indent=2)
            Path(tmp).rename(p)
        except Exception:
            Path(tmp).unlink(missing_ok=True)
            raise

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def record(self, row: dict[str, Any], *, note: str = "") -> None:
        """Append a new pipeline run result to the timeline.

        Args:
            row: Output dict from ``run_pipeline`` or ``exo --output``.
            note: Optional free-text annotation for this run.
        """
        candidate_id = str(row.get("candidate_id", "unknown"))

        fpp = (
            row.get("scores", {}).get("false_positive_probability")
            or row.get("best_fpp")
            or 0.0
        )
        planet_posterior = row.get("posterior", {}).get("planet_candidate") or 0.0
        pathway = row.get("pathway", "unknown")
        scorer = row.get("meta", {}).get("scorer", "bayesian")
        period_days = float(row.get("period_days") or 0.0)

        entry = TimelineEntry(
            run_at=datetime.datetime.now(datetime.UTC).isoformat(),
            period_days=period_days,
            fpp=float(fpp),
            planet_posterior=float(planet_posterior),
            pathway=str(pathway),
            scorer=str(scorer),
            note=note,
        )

        data = self._load(candidate_id)
        data.append(asdict(entry))
        self._save(candidate_id, data)

    def entries(self, candidate_id: str = "unknown") -> list[TimelineEntry]:
        """Return all timeline entries for a candidate.

        Args:
            candidate_id: Candidate identifier (matches filename stem).

        Returns:
            List of :class:`TimelineEntry` in chronological order.
        """
        return [TimelineEntry(**d) for d in self._load(candidate_id)]

    def latest(self, candidate_id: str = "unknown") -> TimelineEntry | None:
        """Return the most recent entry, or ``None`` if no entries exist."""
        data = self._load(candidate_id)
        return TimelineEntry(**data[-1]) if data else None

    def summary(self, candidate_id: str = "unknown") -> dict[str, Any]:
        """Return aggregate statistics across all recorded runs.

        Returns:
            Dict with keys ``n_runs``, ``first_seen``, ``last_seen``,
            ``trend_fpp`` (last FPP − first FPP; negative means improving).
        """
        data = self._load(candidate_id)
        if not data:
            return {
                "n_runs": 0,
                "first_seen": None,
                "last_seen": None,
                "trend_fpp": None,
            }
        entries = [TimelineEntry(**d) for d in data]
        trend = entries[-1].fpp - entries[0].fpp if len(entries) >= 2 else None
        return {
            "n_runs": len(entries),
            "first_seen": entries[0].run_at,
            "last_seen": entries[-1].run_at,
            "trend_fpp": trend,
        }

    def to_markdown(self, candidate_id: str = "unknown") -> str:
        """Return a Markdown table of all timeline entries."""
        entries = self.entries(candidate_id)
        if not entries:
            return "_No entries._\n"

        header = "| Run | Period (d) | FPP | Planet Posterior | Pathway | Scorer | Note |"
        sep    = "| --- | --- | --- | --- | --- | --- | --- |"
        lines  = [header, sep]
        for i, e in enumerate(entries, start=1):
            lines.append(
                f"| {i} | {e.period_days:.4f} | {e.fpp:.4f} | {e.planet_posterior:.4f}"
                f" | {e.pathway} | {e.scorer} | {e.note} |"
            )
        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli(argv: list[str] | None = None) -> int:
    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser(
        prog="candidate_timeline",
        description="Show or summarise the pipeline run history for a candidate.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    show_p = sub.add_parser("show", help="Print Markdown timeline table.")
    show_p.add_argument("dir", type=Path, metavar="DIR")
    show_p.add_argument("candidate_id", metavar="CANDIDATE_ID")

    sum_p = sub.add_parser("summary", help="Print summary statistics.")
    sum_p.add_argument("dir", type=Path, metavar="DIR")
    sum_p.add_argument("candidate_id", metavar="CANDIDATE_ID")

    args = parser.parse_args(argv)
    tl = CandidateTimeline(args.dir)

    if args.cmd == "show":
        print(tl.to_markdown(args.candidate_id), end="")
    else:
        import json as _json  # noqa: PLC0415
        print(_json.dumps(tl.summary(args.candidate_id), indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
