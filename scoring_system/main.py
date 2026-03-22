"""
run evaluator.evaluate on every *.jsonl file under data/.
"""

import glob
from evaluator import evaluate


def main() -> None:
    jsonl_paths = sorted(glob.glob("data/*.jsonl"))
    for path in jsonl_paths: # for each jsonl file
        print(f"\n{path}")
        scores = evaluate(path)
        for metric_name, value in scores.items(): 
            print(f"{metric_name:12s}: {value:.3f}") # printing the metric name and value


if __name__ == "__main__":
    main()
