"""Run evaluator.evaluate on JSONL files and render logical-form comparison charts."""

import argparse
from collections import defaultdict
import csv
from pathlib import Path
import re
import sys

SCORING_SYSTEM_DIR = Path(__file__).resolve().parent
if str(SCORING_SYSTEM_DIR) not in sys.path:
    sys.path.insert(0, str(SCORING_SYSTEM_DIR))

RUN_SPLITS = ("validation", "test")
VARIANT_BASE = "base"
VARIANT_GUIDED = "guided"
DATASET_NAMES = {"wikisql": "wikisql", "sqale": "SQaLe"}
MODEL_SIZE_PATTERN = re.compile(r"(\d+(?:\.\d+)?)([bm])", re.IGNORECASE)


def parse_result_filename(jsonl_path: Path) -> tuple[str, str, str, str] | None:
    """Infer model name, dataset, run split, and variant from a JSONL filename."""
    parts = jsonl_path.stem.split("_")

    split_name = next((run_split for run_split in RUN_SPLITS if run_split in parts), None)
    if split_name is None:
        return None

    dataset_name = next(
        (DATASET_NAMES[part.lower()] for part in parts if part.lower() in DATASET_NAMES),
        None,
    )
    if dataset_name is None:
        return None

    filtered_parts = parts.copy()
    filtered_parts.remove(split_name)
    dataset_token = next(part for part in filtered_parts if part.lower() in DATASET_NAMES)
    filtered_parts.remove(dataset_token)

    variant_name = VARIANT_BASE
    if "guided" in filtered_parts:
        filtered_parts.remove("guided")
        variant_name = VARIANT_GUIDED

    model_name = "_".join(filtered_parts)
    if not model_name:
        return None

    return model_name, dataset_name, split_name, variant_name


def load_evaluator():
    """Import the evaluator only when the full scoring pipeline is executed."""
    from evaluator import evaluate

    return evaluate


def load_pyplot():
    """Import matplotlib lazily and configure a non-interactive backend."""
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def model_sort_key(model_name: str) -> tuple[float, str]:
    """Sort models by parameter size, then name."""
    size_match = MODEL_SIZE_PATTERN.search(model_name)
    if size_match is None:
        return float("inf"), model_name

    size_value = float(size_match.group(1))
    size_unit = size_match.group(2).lower()
    size_in_millions = size_value if size_unit == "m" else size_value * 1000
    return size_in_millions, model_name


def variant_sort_key(variant_name: str) -> int:
    """Place base rows before guided rows in exports."""
    if variant_name == VARIANT_BASE:
        return 0
    if variant_name == VARIANT_GUIDED:
        return 1
    return 2


def save_stats_csv(rows: list[dict[str, str | float]], csv_path: Path) -> None:
    """Write one CSV row per evaluated model file with all reported metrics."""
    if not rows:
        print("No evaluation rows available for CSV export; skipping CSV.")
        return

    fieldnames = list(rows[0].keys())
    sorted_rows = sorted(
        rows,
        key=lambda row: (
            str(row["dataset"]).lower(),
            RUN_SPLITS.index(row["split"]) if row["split"] in RUN_SPLITS else len(RUN_SPLITS),
            model_sort_key(str(row["model"])),
            variant_sort_key(str(row["variant"])),
            str(row["filename"]),
        ),
    )

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted_rows)

    print(f"Saved CSV stats to {csv_path}")


def save_logical_form_chart(
    dataset_name: str,
    split_name: str,
    split_scores: dict[str, dict[str, float]],
    output_dir: Path,
) -> bool:
    """Render one grouped bar chart comparing base and guided logical_form."""
    paired_models = [
        model_name
        for model_name, variant_scores in sorted(
            split_scores.items(), key=lambda item: model_sort_key(item[0])
        )
        if VARIANT_BASE in variant_scores and VARIANT_GUIDED in variant_scores
    ]
    if not paired_models:
        print(f"No paired base/guided files found for {split_name}; skipping chart.")
        return False

    pyplot = load_pyplot()
    base_scores = [split_scores[model_name][VARIANT_BASE] for model_name in paired_models]
    guided_scores = [split_scores[model_name][VARIANT_GUIDED] for model_name in paired_models]

    figure_width = max(8, len(paired_models) * 1.6)
    figure, axis = pyplot.subplots(figsize=(figure_width, 6))
    positions = list(range(len(paired_models)))
    bar_width = 0.36

    base_positions = [position - bar_width / 2 for position in positions]
    guided_positions = [position + bar_width / 2 for position in positions]

    axis.bar(
        base_positions,
        base_scores,
        width=bar_width,
        color="#4c78a8",
        label="Base logical_form",
    )
    axis.bar(
        guided_positions,
        guided_scores,
        width=bar_width,
        color="#54a24b",
        label="Guided logical_form",
    )

    for position, score in zip(base_positions, base_scores, strict=True):
        axis.text(
            position,
            min(score + 0.015, 0.99),
            f"{score:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    for position, score in zip(guided_positions, guided_scores, strict=True):
        axis.text(
            position,
            min(score + 0.015, 0.99),
            f"{score:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    axis.set_title(f"Logical Form Performance: {dataset_name} {split_name.capitalize()}")
    axis.set_xlabel("Model")
    axis.set_ylabel("Logical form accuracy")
    axis.set_xticks(positions)
    axis.set_xticklabels(paired_models, rotation=20, ha="right")
    axis.set_ylim(0, 1.02)
    axis.grid(axis="y", linestyle="--", alpha=0.35)
    axis.legend()
    figure.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    chart_path = output_dir / f"logical_form_{dataset_name.lower()}_{split_name}.png"
    figure.savefig(chart_path, dpi=200, bbox_inches="tight")
    pyplot.close(figure)
    print(f"Saved {dataset_name} {split_name} chart to {chart_path}")
    return True


def main() -> None:
    """Evaluate all JSONL files in a directory and emit logical-form charts."""
    evaluate = load_evaluator()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "jsonl_dir",
        nargs="?",
        type=Path,
        default=Path("data"),
        help="Directory containing JSONL files to evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save generated chart images. Defaults to <jsonl_dir>/charts.",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=None,
        help="Path to save per-model evaluation stats as CSV. Defaults to <output_dir>/model_stats.csv.",
    )
    args = parser.parse_args()
    data_dir: Path = args.jsonl_dir
    output_dir = args.output_dir or data_dir / "charts"
    csv_path = args.csv_path or output_dir / "model_stats.csv"

    jsonl_paths = sorted(data_dir.glob("*.jsonl"))
    if not jsonl_paths:
        raise SystemExit(f"No JSONL files found in {data_dir}")

    logical_form_scores: dict[str, dict[str, dict[str, dict[str, float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict))
    )
    csv_rows: list[dict[str, str | float]] = []

    for path in jsonl_paths:
        print(f"\n{path}")
        scores = evaluate(path)
        for metric_name, value in scores.items():
            print(f"{metric_name:12s}: {value:.3f}")

        parsed_name = parse_result_filename(path)
        if parsed_name is None:
            print(
                f"Skipping chart aggregation for {path.name}: filename does not encode dataset and split."
            )
            continue

        model_name, dataset_name, split_name, variant_name = parsed_name
        logical_form_scores[dataset_name][split_name][model_name][variant_name] = scores[
            "logical_form"
        ]
        csv_rows.append(
            {
                "filename": path.name,
                "model": model_name,
                "dataset": dataset_name,
                "split": split_name,
                "variant": variant_name,
                **scores,
            }
        )

    for dataset_name, dataset_scores in sorted(logical_form_scores.items()):
        for split_name in RUN_SPLITS:
            save_logical_form_chart(
                dataset_name,
                split_name,
                dataset_scores.get(split_name, {}),
                output_dir,
            )

    save_stats_csv(csv_rows, csv_path)


if __name__ == "__main__":
    main()
