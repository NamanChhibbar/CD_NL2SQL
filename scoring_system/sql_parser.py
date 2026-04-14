"""
Parse cleaned SQL strings into ``ParsedSQL`` objects.

SELECT (with optional DISTINCT and common aggregates), WHERE (AND-separated), GROUP BY, ORDER BY, and LIMIT. 
Strings are normalized (dates to ISO, numbers without commas).
"""

import re
from typing import List
from dateutil import parser as date_parser
from clean_sql import clean
from schema import Condition, OrderItem, ParsedSQL, SelectClause

SUPPORTED_AGGREGATES = {"count", "min", "max", "sum", "avg"} # lowercase aggregate names recognized in the SELECT list


def normalize_value(raw: str) -> str:
    """
    Normalize a literal or expression fragment from the right-hand side of a WHERE comparison.

    Strips outer quotes, parses obvious dates to ``YYYY-MM-DD``, and strips commas from
    digit-like values. Falls back to lowercased string if no rule applies.
    """
    if raw is None: 
        return raw

    value = raw.strip().lower()

    if (value.startswith("'") and value.endswith("'")) or (
        value.startswith('"') and value.endswith('"')
    ):
        value = value[1:-1].strip() # removing the surrounding quotes

    if looks_like_date(value): # if the value is a date
        try:
            parsed_date = date_parser.parse(value)
            return parsed_date.strftime("%Y-%m-%d")
        except (ValueError, TypeError, OverflowError):
            pass

    if looks_like_number(value): # if the value is a number
        return value.replace(",", "").replace(" ", "")

    return value


def looks_like_number(value: str) -> bool:
    """True for simple integer-like strings that may contain commas or spaces."""
    return bool(re.fullmatch(r"[\d,\s]+", value))


def looks_like_date(value: str) -> bool:
    """
    Heuristic: treat as date only if a month name or a 4-digit year appears.

    Avoids misclassifying values like ``11-8`` as dates.
    """
    lower = value.lower()
    has_month_word = bool(
        re.search(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b", lower)
    )
    has_four_digit_year = bool(re.search(r"\b\d{4}\b", lower))
    return has_month_word or has_four_digit_year


def normalize_column_name(column: str): 
    """Trim whitespace from a column identifier; pass through None."""
    if column is None:
        return None
    return column.strip()


def parse_sql(sql: str) -> ParsedSQL:
    """
    Parse ``sql`` into a ``ParsedSQL`` after running ``clean_sql.clean``.

    Raises nothing on malformed SQL: missing clauses yield empty lists / None where appropriate.
    """
    sql = clean(sql)

    # --- SELECT ---
    distinct = False
    aggregators: List[str] = []
    columns: List[str] = []

    select_match = re.search(r"select (.*?)( from|$)", sql)
    if select_match:
        select_fragment = select_match.group(1).strip()

        if select_fragment.startswith("distinct"):
            distinct = True
            select_fragment = select_fragment[len("distinct") :].strip()

        select_items = [piece.strip() for piece in select_fragment.split(",")]

        for select_item in select_items:
            aggregate_name = "none"
            column_name = None

            paren_aggregate_match = re.match(
                r"^(count|min|max|sum|avg)\s*\(\s*(.*?)\s*\)$", select_item
            )

            if paren_aggregate_match:
                aggregate_name = paren_aggregate_match.group(1)
                column_name = paren_aggregate_match.group(2).strip()
            else:
                tokens = select_item.split()
                if len(tokens) >= 2 and tokens[0] in SUPPORTED_AGGREGATES:
                    aggregate_name = tokens[0]
                    column_name = " ".join(tokens[1:]).strip()
                else:
                    column_name = select_item

            if aggregate_name == "count" and (column_name == "*" or column_name == ""):
                column_name = "__all__"

            aggregators.append(aggregate_name)
            columns.append(normalize_column_name(column_name))

    select_clause = SelectClause(
        distinct=distinct, aggregators=aggregators, columns=columns
    )

    # --- WHERE ---
    where_conditions: List[Condition] = []

    where_match = re.search(r"where (.*?)(group by|order by|limit|$)", sql)
    if where_match:
        where_fragment = where_match.group(1)
        and_parts = re.split(r"\band\b", where_fragment)

        for part in and_parts:
            part = part.strip()
            operator_match = re.search(r"(>=|<=|!=|=|>|<)", part)
            if operator_match:
                operator = operator_match.group(1)
                lhs_rhs = part.split(operator)
                if len(lhs_rhs) == 2:
                    column_name = normalize_column_name(lhs_rhs[0])
                    rhs = lhs_rhs[1].strip()
                    rhs = normalize_value(rhs)
                    where_conditions.append(
                        Condition(column=column_name, operator=operator, value=rhs)
                    )

    # --- GROUP BY ---
    group_by_columns: List[str] = []
    group_match = re.search(r"group by (.*?)(order by|limit|$)", sql)
    if group_match:
        group_fragment = group_match.group(1).strip()
        group_by_columns = [
            normalize_column_name(column) for column in group_fragment.split(",")
        ]

    # --- ORDER BY ---
    order_by_items: List[OrderItem] = []
    order_match = re.search(r"order by (.*?)(limit|$)", sql)
    if order_match:
        order_fragment = order_match.group(1).strip()
        order_parts = [piece.strip() for piece in order_fragment.split(",")]

        for order_part in order_parts:
            direction = "asc"
            if order_part.endswith(" desc"):
                direction = "desc"
                column_name = order_part[:-5].strip()
            elif order_part.endswith(" asc"):
                direction = "asc"
                column_name = order_part[:-4].strip()
            else:
                column_name = order_part

            order_by_items.append(
                OrderItem(column=normalize_column_name(column_name), direction=direction)
            )

    # --- LIMIT ---
    limit_value = None
    limit_match = re.search(r"limit (\d+)", sql)
    if limit_match:
        limit_value = int(limit_match.group(1))

    return ParsedSQL(
        select=select_clause,
        where=where_conditions,
        group_by=group_by_columns,
        order_by=order_by_items,
        limit=limit_value,
    )
