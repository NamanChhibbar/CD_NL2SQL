"""Defined prompts used in LLM calls."""

WIKISQL_PROMPT = """Help the user write an SQL statement for their question.
You will be given a representation of the SQL database's tables and the users query.
Only generate valid SQL output.

Table: {table}

Query: {query}

SQL Query:
"""

SQALE_PROMPT = """Help the user write an SQL statement for their question.
You will be given a representation of the SQL database's schema and the users query.
Only generate valid SQL output.

Schema: {table}

Query: {query}

SQL Query:
"""
