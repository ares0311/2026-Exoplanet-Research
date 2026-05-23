"""Serialise candidate dicts to a VO-compliant VOTable XML string.

Uses only the stdlib ``xml.etree.ElementTree`` module.  The output is a
minimal VOTable 1.4 document with one RESOURCE/TABLE containing one FIELD
per requested column.  Suitable for upload to CDS/VizieR or import into
TOPCAT.

Public API
----------
VOTableResult(n_rows, columns, xml_string, warnings, flag)
format_as_votable(records, columns, *, table_name) -> VOTableResult
format_votable_result(result) -> str
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass

# VOTable datatype mapping: Python type name → VOTable datatype
_TYPE_MAP: dict[str, str] = {
    "int": "int",
    "float": "double",
    "str": "char",
    "bool": "boolean",
    "NoneType": "char",
}


@dataclass(frozen=True)
class VOTableResult:
    n_rows: int
    columns: tuple[str, ...]
    xml_string: str
    warnings: tuple[str, ...]
    flag: str  # "OK" | "EMPTY" | "INVALID"


def _infer_votable_type(values: list) -> str:
    """Infer a VOTable datatype from a list of sample values."""
    for v in values:
        if v is None:
            continue
        t = type(v).__name__
        return _TYPE_MAP.get(t, "char")
    return "char"


def _value_to_str(v) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "T" if v else "F"
    if isinstance(v, float):
        return repr(v)
    return str(v)


def format_as_votable(
    records: list[dict],
    columns: list[str] | None = None,
    *,
    table_name: str = "candidates",
) -> VOTableResult:
    """Serialise a list of candidate dicts to a VOTable XML string.

    Args:
        records: List of dicts (one per candidate row).
        columns: Ordered column names to include.  Defaults to all keys
            found in the first record.
        table_name: VOTable TABLE name attribute.

    Returns:
        :class:`VOTableResult`.
    """
    if not isinstance(records, list):
        return VOTableResult(0, (), "", (), "INVALID")
    if not records:
        return VOTableResult(0, (), "", (), "EMPTY")

    if columns is None:
        cols: list[str] = list(records[0].keys())
    else:
        cols = list(columns)

    if not cols:
        return VOTableResult(0, (), "", (), "INVALID")

    warnings: list[str] = []

    # Build type map from first non-null value per column
    col_types: dict[str, str] = {}
    for col in cols:
        sample = [r.get(col) for r in records if r.get(col) is not None]
        col_types[col] = _infer_votable_type(sample)

    # Build XML tree
    root = ET.Element("VOTABLE", {"version": "1.4",
                                   "xmlns": "http://www.ivoa.net/xml/VOTable/v1.3"})
    resource = ET.SubElement(root, "RESOURCE", {"name": table_name})
    table = ET.SubElement(resource, "TABLE", {"name": table_name})

    for col in cols:
        dt = col_types[col]
        attrib = {"name": col, "datatype": dt}
        if dt == "char":
            attrib["arraysize"] = "*"
        ET.SubElement(table, "FIELD", attrib)

    data = ET.SubElement(table, "DATA")
    tabledata = ET.SubElement(data, "TABLEDATA")

    for rec in records:
        tr = ET.SubElement(tabledata, "TR")
        for col in cols:
            td = ET.SubElement(tr, "TD")
            td.text = _value_to_str(rec.get(col))

    # Check for columns requested but missing from all records
    all_keys: set[str] = set()
    for r in records:
        all_keys.update(r.keys())
    for col in cols:
        if col not in all_keys:
            warnings.append(f"Column '{col}' not found in any record")

    xml_string = ET.tostring(root, encoding="unicode", xml_declaration=False)

    return VOTableResult(
        n_rows=len(records),
        columns=tuple(cols),
        xml_string=xml_string,
        warnings=tuple(warnings),
        flag="OK",
    )


def format_votable_result(result: VOTableResult) -> str:
    """Format VOTable result summary as Markdown."""
    lines = [
        "## VOTable Formatter",
        "",
        f"- Rows: {result.n_rows}",
        f"- Columns: {len(result.columns)} ({', '.join(result.columns[:5])}"
        f"{'…' if len(result.columns) > 5 else ''})",
        f"- XML length: {len(result.xml_string)} chars",
        f"- Warnings: {len(result.warnings)}",
        f"- **Flag: {result.flag}**",
    ]
    if result.warnings:
        for w in result.warnings:
            lines.append(f"  - ⚠ {w}")
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(
        prog="votable_formatter",
        description="Serialise candidate dicts to VOTable XML.",
    )
    parser.add_argument("--json", type=str, default=None, help="JSON array string")
    parser.add_argument("--columns", type=str, default=None, help="Comma-separated columns")
    args = parser.parse_args(argv)

    records = json.loads(args.json) if args.json else []
    cols = args.columns.split(",") if args.columns else None
    result = format_as_votable(records, cols)
    print(format_votable_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
