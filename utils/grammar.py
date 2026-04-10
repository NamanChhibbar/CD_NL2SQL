"""Grammar building utilities for SQL guided decoding."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from .enums import DatasetNames


def escape_grammar_atom(text: str) -> str:
    """Escape one literal so it can be embedded in an EBNF grammar rule."""
    escaped_text = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"\\"{escaped_text}\\""'


def build_dynamic_grammar(
    base_grammar: str,
    tables: list[str],
    columns: list[str],
) -> str:
    """Inject table and column names into the base grammar template.

    ``__TABLE_REF__`` is replaced with an alternation of *tables* and
    ``__COLUMN_REF__`` with an alternation of *columns*.
    """
    table_rule = " | ".join(escape_grammar_atom(t) for t in tables)
    column_rule = " | ".join(escape_grammar_atom(c) for c in columns)

    grammar = base_grammar.replace("__TABLE_REF__", table_rule)
    return grammar.replace("__COLUMN_REF__", column_rule)


def read_grammar_template(path: Path) -> str:
    """Read the base grammar template from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Grammar file not found: {path}")
    return path.read_text(encoding="utf-8")


_CREATE_TABLE_RE = re.compile(
    r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?[`\"']?(\w+)[`\"']?\s*\((.*?)\)",
    re.IGNORECASE | re.DOTALL,
)
_CONSTRAINT_RE = re.compile(
    r"^\s*(PRIMARY|FOREIGN|UNIQUE|CHECK|CONSTRAINT)\s",
    re.IGNORECASE,
)


def parse_sqale_schema(schema_str: str) -> tuple[list[str], list[str]]:
    """Extract table names and column names from SQaLe CREATE TABLE schemas.

    Returns:
        A tuple ``(table_names, column_names)`` where both are lists of
        strings.  Constraint lines (PRIMARY KEY, FOREIGN KEY, etc.) are
        skipped.
    """
    tables: list[str] = []
    columns: list[str] = []

    for match in _CREATE_TABLE_RE.finditer(schema_str):
        tables.append(match.group(1))
        for col_def in match.group(2).split(","):
            col_def = col_def.strip()
            if not col_def or _CONSTRAINT_RE.match(col_def):
                continue
            col_match = re.match(r"[`\"']?(\w+)[`\"']?", col_def)
            if col_match:
                columns.append(col_match.group(1))

    return tables, columns


def extract_schema_info(
    item: dict[str, Any], dataset_name: DatasetNames
) -> tuple[list[str], list[str]]:
    """Return ``(table_names, column_names)`` for a single dataset item.

    * **WikiSQL** - fixed table name ``"table"`` with column headers.
    * **SQaLe** - parsed from the ``schema`` field.
    """
    if dataset_name == DatasetNames.WIKISQL:
        return ["table"], list(item["table"]["header"])

    tables, columns = parse_sqale_schema(item["schema"])
    if not tables:
        tables = ["table"]
    if not columns:
        raise ValueError(f"Could not parse columns from SQaLe schema: {item['schema'][:200]}")
    return tables, columns
