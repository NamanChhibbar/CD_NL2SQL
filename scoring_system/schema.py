"""
Structured representation of a parsed SQL query.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class Condition:
    """One predicate in a WHERE clause: ``column operator value``."""

    column: str
    operator: str
    value: Any


@dataclass
class SelectClause:
    """The SELECT list: DISTINCT flag, one aggregator per selected expression, and column names."""

    distinct: bool
    aggregators: List[str]
    columns: List[str]


@dataclass
class OrderItem:
    """A single column in ORDER BY with sort direction."""

    column: str
    direction: str  # "asc" or "desc"


@dataclass
class ParsedSQL:
    """Full parse of a single SQL string (subset of SQL supported by this project)."""

    select: SelectClause
    where: List[Condition] = field(default_factory=list)
    group_by: List[str] = field(default_factory=list)
    having: List[Condition] = field(default_factory=list)
    order_by: List[OrderItem] = field(default_factory=list)
    limit: Optional[int] = None
