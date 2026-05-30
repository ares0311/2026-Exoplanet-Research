"""Format a candidate as an ExoFOP-style TOI alert payload.

Public API:
    ToiAlertResult  -- frozen dataclass
    format_toi_alert(tic_id, period_days, epoch_bjd, depth_ppm, duration_hours,
                     fpp, disposition) -> ToiAlertResult
    format_toi_text(result) -> str
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class ToiAlertResult:
    tic_id: str
    period_days: float
    epoch_bjd: float
    depth_ppm: float
    duration_hours: float
    fpp: float
    disposition: str
    flag: str


def format_toi_alert(
    tic_id: str,
    period_days: float,
    epoch_bjd: float,
    depth_ppm: float,
    duration_hours: float,
    fpp: float,
    disposition: str = "PC",
) -> ToiAlertResult:
    if period_days <= 0:
        return ToiAlertResult(
            tic_id=tic_id, period_days=period_days, epoch_bjd=epoch_bjd,
            depth_ppm=depth_ppm, duration_hours=duration_hours,
            fpp=fpp, disposition=disposition, flag="INVALID_PERIOD",
        )
    if depth_ppm <= 0:
        return ToiAlertResult(
            tic_id=tic_id, period_days=period_days, epoch_bjd=epoch_bjd,
            depth_ppm=depth_ppm, duration_hours=duration_hours,
            fpp=fpp, disposition=disposition, flag="INVALID_DEPTH",
        )
    if not 0.0 <= fpp <= 1.0:
        return ToiAlertResult(
            tic_id=tic_id, period_days=period_days, epoch_bjd=epoch_bjd,
            depth_ppm=depth_ppm, duration_hours=duration_hours,
            fpp=fpp, disposition=disposition, flag="INVALID_FPP",
        )
    return ToiAlertResult(
        tic_id=tic_id, period_days=period_days, epoch_bjd=epoch_bjd,
        depth_ppm=depth_ppm, duration_hours=duration_hours,
        fpp=fpp, disposition=disposition, flag="OK",
    )


def format_toi_text(result: ToiAlertResult) -> str:
    lines = [
        f"## TOI Alert — TIC {result.tic_id}",
        "",
        "| Field | Value |",
        "|-------|-------|",
        f"| TIC ID | {result.tic_id} |",
        f"| Period (days) | {result.period_days:.6f} |",
        f"| Epoch (BJD) | {result.epoch_bjd:.6f} |",
        f"| Depth (ppm) | {result.depth_ppm:.1f} |",
        f"| Duration (hours) | {result.duration_hours:.3f} |",
        f"| FPP | {result.fpp:.4f} |",
        f"| Disposition | {result.disposition} |",
        f"| Flag | {result.flag} |",
    ]
    return "\n".join(lines)


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Format a TOI alert payload.")
    parser.add_argument("tic_id")
    parser.add_argument("period_days", type=float)
    parser.add_argument("epoch_bjd", type=float)
    parser.add_argument("depth_ppm", type=float)
    parser.add_argument("duration_hours", type=float)
    parser.add_argument("fpp", type=float)
    parser.add_argument("--disposition", default="PC")
    args = parser.parse_args()
    result = format_toi_alert(
        args.tic_id, args.period_days, args.epoch_bjd,
        args.depth_ppm, args.duration_hours, args.fpp, args.disposition,
    )
    print(format_toi_text(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
