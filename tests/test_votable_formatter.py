"""Tests for Skills/votable_formatter.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from votable_formatter import (
    format_as_votable,
    format_votable_result,
)


class TestFormatAsVotable:
    def test_empty_records_returns_empty(self):
        result = format_as_votable([], ["ra", "dec"])
        assert result.flag == "EMPTY"
        assert result.n_rows == 0

    def test_basic_records(self):
        records = [{"ra": 12.3, "dec": -45.6}]
        result = format_as_votable(records, ["ra", "dec"])
        assert result.flag == "OK"
        assert result.n_rows == 1

    def test_xml_string_is_string(self):
        records = [{"ra": 12.3}]
        result = format_as_votable(records, ["ra"])
        assert isinstance(result.xml_string, str)

    def test_xml_contains_votable_tag(self):
        records = [{"ra": 1.0}]
        result = format_as_votable(records, ["ra"])
        assert "VOTABLE" in result.xml_string

    def test_xml_contains_field_name(self):
        records = [{"tic_id": 12345}]
        result = format_as_votable(records, ["tic_id"])
        assert "tic_id" in result.xml_string

    def test_xml_contains_row_value(self):
        records = [{"tmag": 11.5}]
        result = format_as_votable(records, ["tmag"])
        assert "11.5" in result.xml_string

    def test_multiple_rows(self):
        records = [{"ra": i * 10.0} for i in range(5)]
        result = format_as_votable(records, ["ra"])
        assert result.n_rows == 5

    def test_string_column_type(self):
        records = [{"name": "TOI-700"}]
        result = format_as_votable(records, ["name"])
        assert result.flag == "OK"
        assert "char" in result.xml_string.lower() or "TD" in result.xml_string

    def test_integer_column_type(self):
        records = [{"sector": 5}]
        result = format_as_votable(records, ["sector"])
        assert result.flag == "OK"

    def test_columns_reported(self):
        records = [{"ra": 1.0, "dec": 2.0}]
        result = format_as_votable(records, ["ra", "dec"])
        assert "ra" in result.columns
        assert "dec" in result.columns

    def test_custom_table_name(self):
        records = [{"ra": 1.0}]
        result = format_as_votable(records, ["ra"], table_name="my_table")
        assert "my_table" in result.xml_string

    def test_invalid_records_type(self):
        result = format_as_votable("not a list", ["ra"])
        assert result.flag == "INVALID"

    def test_no_columns(self):
        records = [{"ra": 1.0}]
        result = format_as_votable(records, [])
        assert result.flag in ("OK", "INVALID")

    def test_missing_value_handled(self):
        records = [{"ra": 1.0, "dec": None}]
        result = format_as_votable(records, ["ra", "dec"])
        assert result.flag == "OK"


class TestFormatVotableResult:
    def test_returns_string(self):
        records = [{"ra": 1.0}]
        result = format_as_votable(records, ["ra"])
        text = format_votable_result(result)
        assert isinstance(text, str)

    def test_contains_flag(self):
        records = [{"ra": 1.0}]
        result = format_as_votable(records, ["ra"])
        text = format_votable_result(result)
        assert "OK" in text

    def test_contains_row_count(self):
        records = [{"ra": 1.0}, {"ra": 2.0}]
        result = format_as_votable(records, ["ra"])
        text = format_votable_result(result)
        assert "2" in text
