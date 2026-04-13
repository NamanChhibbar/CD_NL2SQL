"""run evaluator.evaluate on every *.jsonl file under a target directory."""

import argparse
from pathlib import Path

from evaluator import evaluate


def main() -> None:
    """Entry point of script."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "jsonl_dir",
        nargs="?",
        type=Path,
        default="data",
        help="Directory containing JSONL files to evaluate.",
    )
    args = parser.parse_args()
    data_dir: Path = args.jsonl_dir

    jsonl_paths = sorted(data_dir.glob("*.jsonl"))
    for path in jsonl_paths:  # for each jsonl file
        print(f"\n{path}")
        scores = evaluate(path)
        for metric_name, value in scores.items():
            print(f"{metric_name:12s}: {value:.3f}")  # printing the metric name and value


if __name__ == "__main__":
    main()
