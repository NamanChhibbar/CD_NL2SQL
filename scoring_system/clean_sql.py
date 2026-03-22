"""
Normalize raw SQL text before parsing.

Goals: lowercasing, consistent spacing around operators, stripping noise
(semicolons, backticks, table aliases like ``t1.``), and a stable ``count(*)``
form. This keeps ``sql_parser`` regexes simpler and comparisons fairer.
"""

import re


def clean(sql_text: str) -> str:
    """
    Return a single-line, lowercased SQL string with normalized punctuation and whitespace.

    Parameters
    ----------
    sql_text
        Raw SQL (may include newlines, extra spaces, markdown fences are handled elsewhere).
    """
    normalized = sql_text.lower()
    normalized = normalized.replace("\n", " ")

    # Pad comparison operators so tokens split predictably for downstream regexes.
    normalized = re.sub(r"(>=|<=|!=|=|>|<)", r" \1 ", normalized)

    normalized = re.sub(r"\s*,\s*", ", ", normalized)
    normalized = re.sub(r"\(\s*", "(", normalized)
    normalized = re.sub(r"\s*\)", ")", normalized)

    # Canonical form for COUNT(*).
    normalized = re.sub(r"count\s*\(\s*\*\s*\)", "count(*)", normalized)

    normalized = normalized.replace(";", "")
    normalized = normalized.replace("`", "")

    # Strip simple table aliases (e.g. t1.column -> column) for alignment with gold SQL.
    normalized = re.sub(r"\bt\d+\.", "", normalized)

    # Default missing ASC/DESC to ASC so equivalent orderings match.
    normalized = re.sub(
        r"order by (\w+)(?!\s+(asc|desc))", r"order by \1 asc", normalized
    )

    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized
