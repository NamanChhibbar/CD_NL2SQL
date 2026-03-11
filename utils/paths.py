"""A set of structured paths for file access."""

from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
WIKISQL_DATA = DATA_DIR / "wikisql"

WIKISQL_DATA_TRAIN = WIKISQL_DATA / "train.jsonl"
WIKISQL_DATA_VAL = WIKISQL_DATA / "validation.jsonl"
WIKISQL_DATA_TEST = WIKISQL_DATA / "test.jsonl"

for path in [DATA_DIR, WIKISQL_DATA]:
    path.mkdir(exist_ok=True, parents=True)
