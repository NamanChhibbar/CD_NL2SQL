"""Submit Slurm array jobs for fine-tuning one or more Gemma models."""

import argparse
from datetime import UTC, datetime
import os
from pathlib import Path
import subprocess

from utils.enums import DatasetNames, GemmaModels

REPO_ROOT = Path(__file__).resolve().parents[1]
SBATCH_SCRIPT = REPO_ROOT / "scripts" / "train.sbatch"


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model_names",
        nargs="+",
        choices=[str(model) for model in GemmaModels],
        help="One or more model names to fine-tune.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        choices=[str(dataset) for dataset in DatasetNames],
        default=str(DatasetNames.WIKISQL),
        help="Dataset name to use for fine-tuning.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs"),
        help="Base directory for all fine-tuning outputs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Per-device train batch size.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--num-train-epochs",
        type=int,
        default=1,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length.",
    )
    parser.add_argument(
        "--run-stamp",
        type=str,
        default=datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S"),
        help="Run identifier shared by all array tasks.",
    )
    parser.add_argument(
        "--sbatch-arg",
        dest="sbatch_args",
        action="append",
        default=[],
        help="Extra argument to forward to sbatch. Can be specified multiple times.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the sbatch command without submitting it.",
    )
    return parser


def build_env(args: argparse.Namespace) -> dict[str, str]:
    """Build the environment exported to sbatch."""
    env = os.environ.copy()
    env.update(
        {
            "DATASET_NAME": args.dataset_name,
            "OUTPUT_ROOT": str(args.output_root),
            "RUN_STAMP": args.run_stamp,
            "BATCH_SIZE": str(args.batch_size),
            "GRADIENT_ACCUMULATION_STEPS": str(args.gradient_accumulation_steps),
            "NUM_TRAIN_EPOCHS": str(args.num_train_epochs),
            "LEARNING_RATE": str(args.learning_rate),
            "MAX_SEQ_LENGTH": str(args.max_seq_length),
        }
    )
    return env


def build_command(model_names: list[str], extra_sbatch_args: list[str]) -> list[str]:
    """Construct the sbatch command for the requested models."""
    array_arg = f"--array=0-{len(model_names) - 1}"
    return ["sbatch", array_arg, *extra_sbatch_args, str(SBATCH_SCRIPT), *model_names]


def main() -> None:
    """Submit the Slurm array job."""
    parser = build_parser()
    args = parser.parse_args()

    if not SBATCH_SCRIPT.is_file():
        raise FileNotFoundError(f"Could not find sbatch script at {SBATCH_SCRIPT}")

    command = build_command(args.model_names, args.sbatch_args)
    env = build_env(args)

    print("Submitting fine-tuning array job:")
    print(f"  Models: {', '.join(args.model_names)}")
    print(f"  Dataset: {args.dataset_name}")
    print(f"  Run stamp: {args.run_stamp}")
    print(f"  Command: {' '.join(command)}")

    if args.dry_run:
        return

    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
