"""Tests for the shared EXO-Hunter CLI interaction contract.

Covers the requirements in ``docs/CLI_UX_SPEC.md`` that are testable without a
live terminal, plus the semantic golden assertions required by §13. Golden files
are compared semantically (stable elements only); no test asserts on
timing-dependent animation frames, which §13 explicitly forbids.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from exo_toolkit.hunter_ux import (
    COMMAND_SPECS,
    DEFAULT_RESULT_COLUMNS,
    Column,
    GuidedEntry,
    ValidationError,
    command_index,
    filter_commands,
    render_action_preview,
    render_palette,
    render_results_table,
    select_columns,
    truncate_cell,
    validate_choice,
    validate_positive_float,
    validate_target_count,
    validate_target_reference,
)

GOLDEN_DIR = Path(__file__).parent / "golden"


def _assert_golden(name: str, actual: str) -> None:
    """Compare against a stored golden file, writing it when absent."""
    path = GOLDEN_DIR / name
    if not path.exists():  # pragma: no cover - first-run bootstrap
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(actual, encoding="utf-8")
    expected = path.read_text(encoding="utf-8")
    assert actual == expected, f"golden mismatch for {name}"


class TestValidateTargetCount:
    """UX-IN-03 live validity sentinels."""

    def test_accepts_positive_integer(self) -> None:
        assert validate_target_count("20") == 20
        assert validate_target_count("  7 ") == 7

    def test_rejects_non_numeric_with_spec_sentinel(self) -> None:
        with pytest.raises(ValidationError) as exc:
            validate_target_count("twenty")
        assert str(exc.value) == "Invalid - enter a positive whole number."

    def test_rejects_zero_with_spec_sentinel(self) -> None:
        with pytest.raises(ValidationError) as exc:
            validate_target_count("0")
        assert str(exc.value) == "Invalid - targets must be greater than zero."

    def test_rejects_negative(self) -> None:
        with pytest.raises(ValidationError):
            validate_target_count("-3")

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValidationError):
            validate_target_count("   ")

    def test_rejects_float_string(self) -> None:
        with pytest.raises(ValidationError):
            validate_target_count("2.5")


class TestOtherValidators:
    def test_positive_float_accepts_and_rejects(self) -> None:
        assert validate_positive_float("1.5") == 1.5
        with pytest.raises(ValidationError):
            validate_positive_float("0")
        with pytest.raises(ValidationError):
            validate_positive_float("abc")

    def test_target_reference_accepts_rank_and_identifier(self) -> None:
        assert validate_target_reference("3") == "3"
        assert validate_target_reference("TIC 12345") == "TIC 12345"

    def test_target_reference_rejects_zero_rank_and_empty(self) -> None:
        with pytest.raises(ValidationError):
            validate_target_reference("0")
        with pytest.raises(ValidationError):
            validate_target_reference("")

    def test_choice_validation(self) -> None:
        assert validate_choice("open", ("open", "all")) == "open"
        with pytest.raises(ValidationError) as exc:
            validate_choice("bogus", ("open", "all"))
        assert "choose one of" in str(exc.value)


class TestCommandRegistry:
    """CLI-02 and UX-CMD-02."""

    def test_every_contract_required_command_is_registered(self) -> None:
        required = {
            "/New-Search",
            "/Follow-Up-Search",
            "/Run-Search",
            "/Show-Follow-Ups",
            "/Inspect-Target",
            "/Help",
            "/Exit",
        }
        assert required <= {spec.name for spec in COMMAND_SPECS}

    def test_every_command_has_a_nonempty_description(self) -> None:
        for spec in COMMAND_SPECS:
            assert spec.summary.strip(), f"{spec.name} has no description"

    def test_index_resolves_aliases(self) -> None:
        index = command_index()
        assert index["/run-search"].name == "/Run-Search"
        assert index["/run-new-search"].name == "/Run-Search"

    def test_required_and_optional_parameter_shapes_are_reported(self) -> None:
        index = command_index()
        new_search = index["/new-search"]
        assert new_search.required_names() == "targets"
        assert "max-download-gb" in new_search.optional_names()
        assert index["/help"].required_names() == "none"


class TestFilterCommands:
    """UX-CMD-03 live filtering."""

    def test_empty_query_returns_everything(self) -> None:
        assert filter_commands("") == COMMAND_SPECS

    def test_prefix_match_wins(self) -> None:
        assert [spec.name for spec in filter_commands("/foll")] == ["/Follow-Up-Search"]

    def test_leading_slash_is_optional(self) -> None:
        assert filter_commands("foll") == filter_commands("/foll")

    def test_falls_back_to_summary_substring(self) -> None:
        names = [spec.name for spec in filter_commands("freeze")]
        assert "/New-Search" in names

    def test_unmatched_query_returns_empty(self) -> None:
        assert filter_commands("zzzznope") == ()


class TestTruncateAndColumns:
    """UX-TABLE-01 width awareness."""

    def test_truncate_marks_elision(self) -> None:
        assert truncate_cell("abcdefgh", 5) == "abcd…"
        assert truncate_cell("abc", 5) == "abc"
        assert truncate_cell("abc", 0) == ""

    def test_narrow_width_drops_low_priority_columns_first(self) -> None:
        chosen = select_columns(DEFAULT_RESULT_COLUMNS, 40)
        names = [column.header for column in chosen]
        assert "Rank" in names and "Target" in names
        assert "Reason" not in names

    def test_rank_and_identity_survive_extreme_narrowing(self) -> None:
        chosen = select_columns(DEFAULT_RESULT_COLUMNS, 5)
        names = [column.header for column in chosen]
        assert names == ["Rank", "Target"]

    def test_wide_width_keeps_every_column(self) -> None:
        chosen = select_columns(DEFAULT_RESULT_COLUMNS, 140)
        assert len(chosen) == len(DEFAULT_RESULT_COLUMNS)

    def test_no_rendered_line_exceeds_terminal_width(self) -> None:
        rows = [
            {
                "rank": 1,
                "target_id": "TIC 237884073",
                "ranking_score": 100.0,
                "search_status": "follow-up",
                "object_classification": "star",
                "selection_reason": "x" * 400,
            }
        ]
        for width in (40, 80, 140):
            rendered = render_results_table(rows, terminal_width=width)
            for line in rendered.splitlines():
                assert len(line) <= width, f"line exceeded width {width}: {len(line)}"

    def test_priority_zero_columns_are_never_dropped(self) -> None:
        columns = (Column("A", "a", 10, priority=0), Column("B", "b", 10, priority=5))
        assert [c.header for c in select_columns(columns, 1)] == ["A"]


class TestGuidedEntry:
    """UX-IN-01/UX-IN-02/UX-IN-03."""

    def _spec(self):
        return command_index()["/new-search"]

    def test_valid_value_is_stored(self) -> None:
        entry = GuidedEntry(self._spec())
        entry.set_value("targets", "12")
        assert entry.values["targets"] == "12"
        assert entry.is_executable()

    def test_invalid_value_raises_and_is_not_stored(self) -> None:
        entry = GuidedEntry(self._spec())
        with pytest.raises(ValidationError):
            entry.set_value("targets", "twenty")
        assert "targets" not in entry.values

    def test_unknown_field_is_rejected(self) -> None:
        entry = GuidedEntry(self._spec())
        with pytest.raises(ValidationError):
            entry.set_value("nonsense", "1")

    def test_default_satisfies_required_field(self) -> None:
        # targets carries a visible default of 20, so the command is executable
        # before the operator types anything (UX-IN-02 "defaults are visible").
        assert GuidedEntry(self._spec()).is_executable()

    def test_required_field_without_default_blocks_execution(self) -> None:
        entry = GuidedEntry(command_index()["/inspect-target"])
        assert not entry.is_executable()
        assert entry.missing_required() == ("rank-or-id",)
        entry.set_value("rank-or-id", "3")
        assert entry.is_executable()

    def test_to_argv_matches_the_scriptable_interface(self) -> None:
        entry = GuidedEntry(self._spec())
        entry.set_value("targets", "5")
        entry.set_value("max-download-gb", "2.5")
        assert entry.to_argv() == ["--targets", "5", "--max-download-gb", "2.5"]

    def test_to_argv_applies_visible_default(self) -> None:
        assert GuidedEntry(self._spec()).to_argv() == ["--targets", "20"]

    def test_clearing_optional_field_removes_it(self) -> None:
        entry = GuidedEntry(self._spec())
        entry.set_value("max-download-gb", "2.5")
        entry.set_value("max-download-gb", "")
        assert "max-download-gb" not in entry.values


class TestActionPreview:
    """CLI/UX spec §8."""

    def test_only_supplied_fields_are_shown(self) -> None:
        rendered = render_action_preview({"mode": "new", "requested_targets": 5})
        assert "Mode:" in rendered and "Requested targets:" in rendered
        assert "Estimated storage:" not in rendered

    def test_unresolved_value_is_labelled_not_invented(self) -> None:
        rendered = render_action_preview({"mode": "new", "estimated_storage": None})
        assert "not resolved" in rendered


class TestGoldenUx:
    """CLI/UX spec §13 semantic golden assertions."""

    def test_command_palette_golden(self) -> None:
        _assert_golden("command_palette.txt", render_palette(terminal_width=80))

    def test_new_search_fields_golden(self) -> None:
        entry = GuidedEntry(command_index()["/new-search"])
        _assert_golden("new_search_fields.txt", entry.render())

    def test_invalid_targets_golden(self) -> None:
        messages = []
        for bad in ("twenty", "0"):
            try:
                validate_target_count(bad)
            except ValidationError as exc:
                messages.append(f"Targets: {bad}\n{exc}")
        _assert_golden("invalid_targets.txt", "\n\n".join(messages))

    def test_action_preview_golden(self) -> None:
        preview = {
            "mode": "new",
            "requested_targets": 5,
            "scientific_constraints": "none",
            "primary_sources": "TIC via MAST",
            "source_freshness": "not resolved",
            "history_freshness": "not resolved",
            "estimated_universe": "not resolved",
            "estimated_storage": "not resolved",
            "estimated_compute": "not resolved",
            "output_behavior": "durable manifest + terminal table",
        }
        _assert_golden("action_preview.txt", render_action_preview(preview))

    def _rows(self) -> list[dict[str, object]]:
        return [
            {
                "rank": 1,
                "target_id": "TIC 237884073",
                "ranking_score": 100.0,
                "search_status": "follow-up",
                "object_classification": "star",
                "selection_reason": "Highest-value acceptance follow-up with corrected composite",
            },
            {
                "rank": 2,
                "target_id": "TIC 38846515",
                "ranking_score": 92.5,
                "search_status": "novel",
                "object_classification": "star",
                "selection_reason": "Adaptive expansion recovered this candidate outside the "
                "initial sample",
            },
        ]

    def test_results_table_80_columns_golden(self) -> None:
        _assert_golden(
            "results_table_80_columns.txt",
            render_results_table(self._rows(), terminal_width=80),
        )

    def test_results_table_140_columns_golden(self) -> None:
        _assert_golden(
            "results_table_140_columns.txt",
            render_results_table(self._rows(), terminal_width=140),
        )
