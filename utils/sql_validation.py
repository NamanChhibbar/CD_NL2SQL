"""Shared SQL syntax validation helpers backed by sqlglot."""

from __future__ import annotations

import re

from sqlglot import Dialect, exp, parse_one
from sqlglot.errors import ParseError, TokenError

SQL_FENCE_RE = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
SQL_VALIDATION_DIALECT = "sqlite"
SQLITE_DIALECT = Dialect.get_or_raise(SQL_VALIDATION_DIALECT)


def extract_sql_from_response(text: str) -> str:
    """Normalize model output down to SQL text."""
    stripped = text.strip()
    if not stripped:
        return stripped

    fence_match = SQL_FENCE_RE.search(stripped)
    if fence_match:
        return fence_match.group(1).strip()

    return stripped


def normalized_sql_token_types(sql: str) -> list[str]:
    """Return the SQLite token stream shape, ignoring trailing semicolons."""
    return [
        token.token_type.name
        for token in SQLITE_DIALECT.tokenize(sql)
        if token.token_type.name != "SEMICOLON"
    ]


def validate_sql_with_sqlglot(sql_text: str) -> tuple[bool, str | None]:
    """Use sqlglot's SQLite parser to catch syntax errors in generated SQL."""
    sql = extract_sql_from_response(sql_text)
    if not sql:
        return False, "The response was empty."

    try:
        parsed = parse_one(sql, dialect=SQL_VALIDATION_DIALECT)
    except (ParseError, TokenError) as exc:
        return False, str(exc)

    if isinstance(parsed, exp.Select) and not parsed.expressions:
        return False, "SELECT statements must include at least one projection."

    regenerated_sql = parsed.sql(dialect=SQL_VALIDATION_DIALECT)
    if normalized_sql_token_types(sql) != normalized_sql_token_types(regenerated_sql):
        return False, "The SQL could not be parsed cleanly by sqlglot."

    return True, None
