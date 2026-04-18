"""Run evaluator.evaluate on JSONL files and render logical-form comparison charts."""

import argparse
from collections import defaultdict
import csv
import math
from pathlib import Path
import re
import sys

SCORING_SYSTEM_DIR = Path(__file__).resolve().parent
if str(SCORING_SYSTEM_DIR) not in sys.path:
    sys.path.insert(0, str(SCORING_SYSTEM_DIR))

RUN_SPLITS = ("validation", "test")
VARIANT_BASE = "base"
VARIANT_GUIDED = "guided"
VARIANT_AGENT_CRITIC = "agent_critic"
VARIANT_FT = "ft"
VARIANT_ORDER = (VARIANT_BASE, VARIANT_GUIDED, VARIANT_AGENT_CRITIC, VARIANT_FT)
VARIANT_LABELS = {
    VARIANT_BASE: "Base",
    VARIANT_GUIDED: "Guided",
    VARIANT_AGENT_CRITIC: "Agent Critic",
    VARIANT_FT: "FT",
}
VARIANT_COLORS = {
    VARIANT_BASE: "#4c78a8",
    VARIANT_GUIDED: "#54a24b",
    VARIANT_AGENT_CRITIC: "#b279a2",
    VARIANT_FT: "#f58518",
}
FALLBACK_VARIANT_COLORS = ("#e45756", "#72b7b2", "#b279a2", "#ff9da6")
DATASET_NAMES = {"wikisql": "wikisql", "sqale": "SQaLe"}
MODEL_SIZE_PATTERN = re.compile(r"(\d+(?:\.\d+)?)([bm])", re.IGNORECASE)
EXPLICIT_VARIANT_SUFFIXES = (
    (("agent", "critic"), VARIANT_AGENT_CRITIC),
    (("guided",), VARIANT_GUIDED),
)


def extract_model_name_and_variants(model_parts: list[str]) -> tuple[str, list[str]]:
    """Strip embedded variant tokens from model parts and return the canonical model name."""
    extracted_variants: list[str] = []
    cleaned_parts: list[str] = []

    for part in model_parts:
        if part.lower() == VARIANT_FT:
            extracted_variants.append(VARIANT_FT)
            continue

        subparts = part.split("-")
        cleaned_subparts = []
        for subpart in subparts:
            if subpart.lower() == VARIANT_FT:
                extracted_variants.append(VARIANT_FT)
                continue
            cleaned_subparts.append(subpart)

        cleaned_part = "-".join(cleaned_subparts)
        if cleaned_part:
            cleaned_parts.append(cleaned_part)

    return "_".join(cleaned_parts), extracted_variants


def combine_variants(variant_markers: list[str]) -> str:
    """Create a stable variant name from extracted markers."""
    if not variant_markers:
        return VARIANT_BASE

    unique_markers = set(variant_markers)
    ordered_markers = [variant for variant in VARIANT_ORDER if variant in unique_markers]
    extra_markers = sorted(unique_markers.difference(VARIANT_ORDER))
    return "+".join([*ordered_markers, *extra_markers])


def extract_explicit_variant_suffixes(parts: list[str]) -> tuple[list[str], list[str]]:
    """Strip known variant suffixes from the end of a filename token list."""
    remaining_parts = parts.copy()
    variant_markers: list[str] = []

    while remaining_parts:
        for suffix_parts, variant_name in EXPLICIT_VARIANT_SUFFIXES:
            if (
                tuple(part.lower() for part in remaining_parts[-len(suffix_parts) :])
                != suffix_parts
            ):
                continue
            del remaining_parts[-len(suffix_parts) :]
            variant_markers.append(variant_name)
            break
        else:
            break

    return remaining_parts, variant_markers


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

    remaining_parts, variant_markers = extract_explicit_variant_suffixes(filtered_parts)
    model_name, embedded_variants = extract_model_name_and_variants(remaining_parts)
    if not model_name:
        return None

    variant_name = combine_variants([*variant_markers, *embedded_variants])
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


def variant_sort_key(variant_name: str) -> tuple[int, str]:
    """Place known variants first and keep other labels stable."""
    if variant_name in VARIANT_ORDER:
        return VARIANT_ORDER.index(variant_name), variant_name
    return len(VARIANT_ORDER), variant_name


def variant_label(variant_name: str) -> str:
    """Return a human-readable label for charts."""
    return " + ".join(
        VARIANT_LABELS.get(part, part.upper() if len(part) <= 3 else part.title())
        for part in variant_name.split("+")
    )


def variant_color(variant_name: str, variant_index: int) -> str:
    """Pick a consistent color for each charted variant."""
    return VARIANT_COLORS.get(
        variant_name,
        FALLBACK_VARIANT_COLORS[variant_index % len(FALLBACK_VARIANT_COLORS)],
    )


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
    """Render one grouped bar chart comparing all logical-form variants."""
    if not split_scores:
        print(f"No files found for {dataset_name} {split_name}; skipping chart.")
        return False

    chart_models = [
        model_name
        for model_name, _variant_scores in sorted(
            split_scores.items(), key=lambda item: model_sort_key(item[0])
        )
    ]
    observed_variants = sorted(
        {
            variant_name
            for variant_scores in split_scores.values()
            for variant_name in variant_scores
        },
        key=variant_sort_key,
    )
    if not observed_variants:
        print(
            f"No logical-form variant data found for {dataset_name} {split_name}; skipping chart."
        )
        return False

    pyplot = load_pyplot()
    figure_width = max(8, len(chart_models) * 1.6)
    figure, axis = pyplot.subplots(figsize=(figure_width, 6))
    positions = list(range(len(chart_models)))
    bar_width = 0.8 / len(observed_variants)

    for variant_index, variant_name in enumerate(observed_variants):
        offset = (variant_index - (len(observed_variants) - 1) / 2) * bar_width
        variant_positions = [position + offset for position in positions]
        variant_scores = [
            split_scores[model_name].get(variant_name, math.nan) for model_name in chart_models
        ]

        axis.bar(
            variant_positions,
            variant_scores,
            width=bar_width,
            color=variant_color(variant_name, variant_index),
            label=f"{variant_label(variant_name)} logical_form",
        )

        for position, score in zip(variant_positions, variant_scores, strict=True):
            if math.isnan(score):
                continue
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
    axis.set_xticklabels(chart_models, rotation=20, ha="right")
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
