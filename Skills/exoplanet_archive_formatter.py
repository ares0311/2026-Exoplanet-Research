"""Format a candidate dict into a structured ExoFOP/NExSci submission record."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ArchiveRecord:
    tic_id: str
    period_days: float | None
    epoch_bjd: float | None
    depth_ppm: float | None
    duration_hours: float | None
    fpp: float | None
    pathway: str | None
    disposition: str
    notes: str
    flag: str


_REQUIRED_FIELDS = ("tic_id",)


def format_archive_record(candidate: dict) -> ArchiveRecord:
    """
    Extract and normalise fields from a pipeline output dict into an
    ExoFOP-style submission record.
    """
    tic = (
        str(candidate.get("tic_id", candidate.get("target_id", "")))
        .strip()
    )
    if not tic:
        return ArchiveRecord(
            tic_id="", period_days=None, epoch_bjd=None,
            depth_ppm=None, duration_hours=None,
            fpp=None, pathway=None,
            disposition="UNKNOWN", notes="", flag="MISSING_TIC_ID",
        )

    signal = candidate.get("signal", {}) or {}
    scores = candidate.get("scores", {}) or {}

    def _float(keys: list[str], src: dict = candidate) -> float | None:
        for k in keys:
            v = src.get(k)
            if v is not None:
                try:
                    return float(v)
                except (TypeError, ValueError):
                    pass
        return None

    period = _float(["period_days"], signal) or _float(["period_days"])
    epoch = _float(["epoch_bjd", "t0_bjd"], signal) or _float(["epoch_bjd"])
    depth = _float(["depth_ppm"], signal) or _float(["depth_ppm"])
    dur = _float(["duration_hours"], signal) or _float(["duration_hours"])
    fpp = (
        _float(["false_positive_probability"], scores)
        or _float(["false_positive_probability", "best_fpp"])
    )
    pathway = candidate.get("pathway") or candidate.get("submission_pathway")

    # Disposition
    if fpp is not None:
        if fpp < 0.10:
            disposition = "PC"  # planet candidate
        elif fpp < 0.50:
            disposition = "APC"  # ambiguous PC
        else:
            disposition = "FP"
    else:
        disposition = "UNKNOWN"

    notes_parts = []
    if pathway:
        notes_parts.append(f"pathway={pathway}")
    meta = candidate.get("meta")
    scorer = meta.get("scorer") if isinstance(meta, dict) else None
    if scorer:
        notes_parts.append(f"scorer={scorer}")

    return ArchiveRecord(
        tic_id=tic,
        period_days=round(period, 6) if period is not None else None,
        epoch_bjd=round(epoch, 6) if epoch is not None else None,
        depth_ppm=round(depth, 2) if depth is not None else None,
        duration_hours=round(dur, 4) if dur is not None else None,
        fpp=round(fpp, 6) if fpp is not None else None,
        pathway=str(pathway) if pathway else None,
        disposition=disposition,
        notes="; ".join(notes_parts),
        flag="OK",
    )


def format_archive_record_markdown(r: ArchiveRecord) -> str:
    def _fmt(v: object) -> str:
        return str(v) if v is not None else "N/A"

    return (
        f"| Field | Value |\n"
        f"|---|---|\n"
        f"| TIC ID | {r.tic_id} |\n"
        f"| Period (days) | {_fmt(r.period_days)} |\n"
        f"| Epoch (BJD) | {_fmt(r.epoch_bjd)} |\n"
        f"| Depth (ppm) | {_fmt(r.depth_ppm)} |\n"
        f"| Duration (hours) | {_fmt(r.duration_hours)} |\n"
        f"| FPP | {_fmt(r.fpp)} |\n"
        f"| Disposition | {r.disposition} |\n"
        f"| Pathway | {_fmt(r.pathway)} |\n"
        f"| Notes | {r.notes or 'N/A'} |\n"
        f"| Flag | {r.flag} |\n"
    )


def _cli() -> int:
    p = argparse.ArgumentParser(description="Format a candidate dict as an archive record.")
    p.add_argument("input", help="JSON file with candidate dict")
    p.add_argument("--json", action="store_true", help="Output JSON instead of Markdown")
    args = p.parse_args()
    data = json.loads(Path(args.input).read_text())
    r = format_archive_record(data)
    if args.json:
        import dataclasses
        print(json.dumps(dataclasses.asdict(r), indent=2))
    else:
        print(format_archive_record_markdown(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
