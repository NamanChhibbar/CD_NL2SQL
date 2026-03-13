"""Common shared logic for working with files."""

from collections.abc import Iterable
import json
from os import PathLike
from pathlib import Path
from typing import Any


def read_jsonl(path: PathLike | str) -> Iterable[Any]:
    """Read data in JSON Lines format.

    Args:
        path: Path to file.
    """
    with open(path, encoding="utf-8") as jsonl_in:
        for line in jsonl_in:
            yield json.loads(line)


def write_jsonl(rows: Iterable[Any], path: Path, **kwargs: Any) -> None:
    """Write data in JSON Lines format.

    Args:
        rows: List of JSON-serializable values to write to file.
        path: Path to file.
        kwargs: Additional parameters to pass to `json.dumps()`.
    """
    # `separators` is used here to write JSON compactly
    lines = (json.dumps(row, ensure_ascii=False, separators=(",", ":"), **kwargs) for row in rows)
    output = "\n".join(lines) + "\n"
    path.write_text(output, encoding="utf-8")
