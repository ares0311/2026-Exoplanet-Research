"""Tests for Skills.candidate_html_export."""
from __future__ import annotations

from pathlib import Path

from Skills.candidate_html_export import to_html_gallery, write_html_gallery


def _row(
    cid: str = "TIC1-001",
    target: str = "TIC 1",
    period: float = 10.0,
    fpp: float = 0.10,
    pathway: str = "tfop_ready",
) -> dict:
    return {
        "candidate_id": cid,
        "target_id": target,
        "period_days": period,
        "best_fpp": fpp,
        "best_pathway": pathway,
    }


class TestToHtmlGallery:
    def test_empty_rows_returns_no_candidates_message(self) -> None:
        result = to_html_gallery([])
        assert "No candidates" in result

    def test_html_contains_doctype(self) -> None:
        result = to_html_gallery([_row()])
        assert "<!DOCTYPE html>" in result

    def test_html_contains_title(self) -> None:
        result = to_html_gallery([_row()], title="My Gallery")
        assert "My Gallery" in result

    def test_html_contains_candidate_id(self) -> None:
        result = to_html_gallery([_row(cid="TIC99-001")])
        assert "TIC99-001" in result

    def test_html_contains_sortable_table_headers(self) -> None:
        result = to_html_gallery([_row()])
        assert "sortTable" in result
        assert "<th" in result

    def test_fpp_in_output(self) -> None:
        result = to_html_gallery([_row(fpp=0.1234)])
        assert "0.1234" in result

    def test_n_candidates_count_shown(self) -> None:
        rows = [_row(cid="A"), _row(cid="B"), _row(cid="C")]
        result = to_html_gallery(rows)
        assert "3" in result

    def test_candidates_sorted_by_fpp_ascending(self) -> None:
        rows = [_row(cid="High", fpp=0.90), _row(cid="Low", fpp=0.05)]
        result = to_html_gallery(rows)
        assert result.index("Low") < result.index("High")

    def test_html_body_tags_present(self) -> None:
        result = to_html_gallery([_row()])
        assert "<body" in result
        assert "</body>" in result

    def test_candidate_card_section_present(self) -> None:
        result = to_html_gallery([_row()])
        assert "candidate-card" in result

    def test_missing_plot_file_does_not_crash(self, tmp_path: Path) -> None:
        result = to_html_gallery([_row(cid="X")], plot_dir=tmp_path)
        assert "X" in result

    def test_embedded_plot_present_when_png_exists(self, tmp_path: Path) -> None:
        png = tmp_path / "TIC1-001.png"
        png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)  # minimal fake PNG
        result = to_html_gallery([_row(cid="TIC1-001")], plot_dir=tmp_path)
        assert "base64" in result


class TestWriteHtmlGallery:
    def test_writes_html_file(self, tmp_path: Path) -> None:
        out = tmp_path / "gallery.html"
        write_html_gallery([_row()], out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_written_file_contains_candidate_id(self, tmp_path: Path) -> None:
        out = tmp_path / "gallery.html"
        write_html_gallery([_row(cid="TIC77-001")], out)
        assert "TIC77-001" in out.read_text()
