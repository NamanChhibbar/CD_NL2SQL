"""Evaluate predicted SQL against gold (reference) SQL."""

from difflib import SequenceMatcher
import json
from pathlib import Path
import re

from schema import ParsedSQL
from sql_parser import parse_sql


def multiset_match_under_value_equality(
    predicted_values: list[str], gold_values: list[str]
) -> bool:
    """Return True if each predicted value can be paired with a distinct gold value such that ``values_are_equivalent`` holds (order-independent).

    Used for WHERE columns and values where clause order may differ.
    """
    if len(predicted_values) != len(gold_values):
        return False

    gold_index_used = [False] * len(gold_values)

    for predicted in predicted_values:
        matched = False
        for gold_index, gold in enumerate(gold_values):
            if not gold_index_used[gold_index] and values_are_equivalent(predicted, gold):
                gold_index_used[gold_index] = True
                matched = True
                break
        if not matched:
            return False

    return True


def is_numeric_string(text: str) -> bool:
    """True if text is a plain integer or decimal (allows $ and commas)."""
    stripped = text.replace("$", "").replace(",", "")
    return bool(re.fullmatch(r"\d+(\.\d+)?", stripped))


def parse_numeric_string(text: str) -> float:
    """Parse a numeric string after removing currency and thousands separators."""
    stripped = text.replace("$", "").replace(",", "")
    return float(stripped)


def string_similarity_ratio(left: str, right: str) -> float:
    """Normalized edit-distance style similarity in [0, 1]."""
    return SequenceMatcher(None, left, right).ratio()


def values_are_equivalent(left: str, right: str) -> bool:
    """Lenient equality for comparing literal values in WHERE and similar places.

    Rules (first match wins): exact string match; numeric equality ignoring $/commas;
    substring containment; fuzzy string similarity above 0.8.
    """
    if left == right:  # if the left and right values are the same
        return True
    elif is_numeric_string(left) and is_numeric_string(
        right
    ):  # if the left and right values are numeric
        return parse_numeric_string(left) == parse_numeric_string(right)
    elif (
        left in right or right in left
    ):  # if the left value is a substring of the right value or the right value is a substring of the left value
        return True
    elif (  # noqa: SIM103
        string_similarity_ratio(left, right) > 0.8
    ):  # if the string similarity ratio is greater than 0.8
        return True
    else:
        return False


def extract_sql_from_response(model_text: str) -> str:
    """If the model wrapped SQL in a ```sql fence, return the inner SQL; otherwise strip the whole text."""
    fence_match = re.search(r"```sql\n(.*?)```", model_text, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()
    return model_text.strip()


def normalize_identifier_or_literal(token: str) -> str:
    """Lowercase, strip, and remove a single layer of surrounding quotes if present."""
    token = token.strip().lower()  # stripping and lowercasing the token

    if (
        token.startswith("'") and token.endswith("'")
    ) or (  # if the token starts with a single quote and ends with a single quote
        token.startswith('"')
        and token.endswith(
            '"'
        )  # if the token starts with a double quote and ends with a double quote
    ):
        token = token[1:-1]  # removing the surrounding quotes

    return token


def canonicalize_parse(parsed: ParsedSQL) -> dict:
    """Build an order-insensitive dict representation of a parse."""
    return {
        "select": sorted(zip(parsed.select.aggregators, parsed.select.columns, strict=True)),
        "where": sorted(
            (condition.column, condition.operator, str(condition.value))
            for condition in parsed.where
        ),
        "group_by": sorted(parsed.group_by),
        "order_by": sorted((item.column, item.direction) for item in parsed.order_by),
        "limit": parsed.limit,
    }


def evaluate(jsonl_path: Path) -> dict[str, float]:
    """Read one example per line from ``jsonl_path`` and return component-wise accuracy rates.

    Returned keys are fractions in [0, 1]: ``agg``, ``select``, ``distinct``,
    ``where_col``, ``where_op``, ``where_val``, ``group_by``, ``order_by``,
    ``limit``, and ``logical_form`` (strict all-or-nothing match).

    Also prints an error breakdown: for failed logical-form examples, which clause
    was the first to disagree (hierarchical blame).
    """
    total_examples = 0

    aggregate_matches = 0
    select_list_matches = 0
    distinct_matches = 0

    where_column_matches = 0
    where_operator_matches = 0
    where_value_matches = 0

    group_by_matches = 0
    order_by_matches = 0
    limit_matches = 0

    logical_form_matches = 0

    # Counts how often each clause is the first failure when logical form is wrong.
    first_failure_counts = {
        "agg_only": 0,
        "col_only": 0,
        "both": 0,
        "select": 0,
        "distinct": 0,
        "where": 0,
        "group_by": 0,
        "order_by": 0,
        "limit": 0,
    }

    with open(jsonl_path, encoding="utf-8") as jsonl_file:
        for line in jsonl_file:
            select_clause_matches = True
            distinct_clause_matches = True
            where_clause_matches = True
            group_by_clause_matches = True
            order_by_clause_matches = True
            limit_clause_matches = True

            record = json.loads(line)

            predicted_sql = extract_sql_from_response(
                record["response"]
            )  # extracting the predicted sql from the response
            gold_sql = record["human_sql"]  # extracting the gold sql from the record

            parsed_predicted = parse_sql(predicted_sql)  # parsing the predicted sql
            parsed_gold = parse_sql(gold_sql)  # parsing the gold sql

            total_examples += 1  # incrementing the total number of examples

            # --- SELECT: aggregators and bare columns (WikiSQL-style) ---
            predicted_aggs = sorted(parsed_predicted.select.aggregators)
            gold_aggs = sorted(parsed_gold.select.aggregators)

            predicted_cols = sorted(
                normalize_identifier_or_literal(column)
                for column in parsed_predicted.select.columns
            )
            gold_cols = sorted(
                normalize_identifier_or_literal(column) for column in parsed_gold.select.columns
            )

            if predicted_cols == gold_cols and predicted_aggs == gold_aggs:
                aggregate_matches += 1
                select_list_matches += 1
            elif predicted_cols == gold_cols:
                select_clause_matches = False
                first_failure_counts["col_only"] += 1
            elif predicted_aggs == gold_aggs:
                select_clause_matches = False
                first_failure_counts["agg_only"] += 1
            else:
                first_failure_counts["both"] += 1
                select_clause_matches = False

            if parsed_predicted.select.distinct == parsed_gold.select.distinct:
                distinct_matches += 1
            else:
                distinct_clause_matches = False

            # --- WHERE: columns, operators, and values scored separately ---
            predicted_where_tuples = sorted(
                (condition.column, condition.operator, str(condition.value))
                for condition in parsed_predicted.where
            )
            gold_where_tuples = sorted(
                (condition.column, condition.operator, str(condition.value))
                for condition in parsed_gold.where
            )

            predicted_where_columns = [
                normalize_identifier_or_literal(column) for column, _, _ in predicted_where_tuples
            ]
            gold_where_columns = [
                normalize_identifier_or_literal(column) for column, _, _ in gold_where_tuples
            ]

            if multiset_match_under_value_equality(predicted_where_columns, gold_where_columns):
                where_column_matches += 1
            else:
                where_clause_matches = False

            predicted_where_operators = [
                normalize_identifier_or_literal(operator)
                for _, operator, _ in predicted_where_tuples
            ]
            gold_where_operators = [
                normalize_identifier_or_literal(operator) for _, operator, _ in gold_where_tuples
            ]

            if predicted_where_operators == gold_where_operators:
                where_operator_matches += 1
            else:
                where_clause_matches = False

            predicted_where_values_raw = [str(value) for _, _, value in predicted_where_tuples]
            gold_where_values_raw = [str(value) for _, _, value in gold_where_tuples]

            predicted_where_values_norm = [
                normalize_identifier_or_literal(value) for value in predicted_where_values_raw
            ]
            gold_where_values_norm = [
                normalize_identifier_or_literal(value) for value in gold_where_values_raw
            ]

            if multiset_match_under_value_equality(
                sorted(predicted_where_values_norm), sorted(gold_where_values_norm)
            ):
                where_value_matches += 1
            else:
                where_clause_matches = False

            # --- GROUP BY ---
            if sorted(parsed_predicted.group_by) == sorted(parsed_gold.group_by):
                group_by_matches += 1
            else:
                group_by_clause_matches = False

            # --- ORDER BY ---
            predicted_order = sorted(
                (item.column, item.direction) for item in parsed_predicted.order_by
            )
            gold_order = sorted((item.column, item.direction) for item in parsed_gold.order_by)

            if predicted_order == gold_order:
                order_by_matches += 1
            else:
                order_by_clause_matches = False

            # --- LIMIT ---
            if parsed_predicted.limit == parsed_gold.limit:
                limit_matches += 1
            else:
                limit_clause_matches = False

            # --- Logical form: assign first-failure bucket for diagnostics ---
            if not select_clause_matches:  # if the select clause is not matched
                first_failure_counts["select"] += 1
            elif not distinct_clause_matches:  # if the distinct clause is not matched
                first_failure_counts["distinct"] += 1
            elif not where_clause_matches:  # if the where clause is not matched
                first_failure_counts["where"] += 1
            elif not group_by_clause_matches:  # if the group by clause is not matched
                first_failure_counts["group_by"] += 1
            elif not order_by_clause_matches:  # if the order by clause is not matched
                first_failure_counts["order_by"] += 1
            elif not limit_clause_matches:  # if the limit clause is not matched
                first_failure_counts["limit"] += 1

            if (  # if all the clauses are matched
                select_clause_matches
                and distinct_clause_matches
                and where_clause_matches
                and group_by_clause_matches
                and order_by_clause_matches
                and limit_clause_matches
            ):
                logical_form_matches += 1  # incrementing the logical form matches

    denominator = total_examples
    print("Error Breakdown (fraction of all examples):")
    for failure_key, count in first_failure_counts.items():
        print(f"{failure_key}: {count / denominator:.3f}")  # printing the failure key and count

    return {
        "agg": aggregate_matches / denominator,
        "select": select_list_matches / denominator,
        "distinct": distinct_matches / denominator,
        "where_col": where_column_matches / denominator,
        "where_op": where_operator_matches / denominator,
        "where_val": where_value_matches / denominator,
        "group_by": group_by_matches / denominator,
        "order_by": order_by_matches / denominator,
        "limit": limit_matches / denominator,
        "logical_form": logical_form_matches / denominator,
    }  # returning the scores
