#!/bin/bash

#SBATCH --job-name=train_gemma_array
#SBATCH --partition=medium-lg
#SBATCH --account=eclair
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=50G
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --time=24:00:00
#SBATCH --output=slurm/train_%A_%a.out
#SBATCH --error=slurm/train_%A_%a.err

set -euo pipefail

MODEL_NAMES=("$@")

if [ "${#MODEL_NAMES[@]}" -eq 0 ] && [ -n "${MODEL_NAMES_CSV:-}" ]; then
  IFS=',' read -r -a MODEL_NAMES <<<"${MODEL_NAMES_CSV}"
fi

if [ "${#MODEL_NAMES[@]}" -eq 0 ] && [ -n "${MODEL_NAME:-}" ]; then
  MODEL_NAMES=("${MODEL_NAME}")
fi

if [ "${#MODEL_NAMES[@]}" -eq 0 ]; then
  MODEL_NAMES=("google/gemma-3-4b-it")
fi

TASK_INDEX="${SLURM_ARRAY_TASK_ID:-0}"

if ! [[ "${TASK_INDEX}" =~ ^[0-9]+$ ]]; then
  echo "SLURM_ARRAY_TASK_ID must be an integer, got '${TASK_INDEX}'." >&2
  exit 1
fi

if [ "${#MODEL_NAMES[@]}" -gt 1 ] && [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
  echo "Received ${#MODEL_NAMES[@]} model names but this job was not submitted as an array." >&2
  echo "Submit with: sbatch --array=0-$((${#MODEL_NAMES[@]} - 1)) scripts/train.sbatch \"\${model_names[@]}\"" >&2
  exit 1
fi

if [ "${TASK_INDEX}" -ge "${#MODEL_NAMES[@]}" ]; then
  echo "SLURM_ARRAY_TASK_ID=${TASK_INDEX} is out of range for ${#MODEL_NAMES[@]} model names." >&2
  exit 1
fi

MODEL_NAME="${MODEL_NAMES[${TASK_INDEX}]}"
DATASET_NAME="${DATASET_NAME:-wikisql}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
RUN_STAMP="${RUN_STAMP:-$(date +'%Y%m%d_%H%M%S')}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"

MODEL_SLUG="$(printf '%s' "${MODEL_NAME}" | sed 's#[/:]#-#g')"
OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_STAMP}/${MODEL_SLUG}"

mkdir -p slurm "${OUTPUT_DIR}"

if [ ! -f ".venv/bin/activate" ]; then
  echo "Virtual environment activation script not found at .venv/bin/activate." >&2
  exit 1
fi

source .venv/bin/activate

echo "Starting training job..."
echo "Array Task Index: ${TASK_INDEX}"
echo "Model: ${MODEL_NAME}"
echo "Dataset: ${DATASET_NAME}"
echo "Output Directory: ${OUTPUT_DIR}"

python3 training/train.py \
  --model-name "${MODEL_NAME}" \
  --dataset-name "${DATASET_NAME}" \
  --output-dir "${OUTPUT_DIR}" \
  --batch-size "${BATCH_SIZE}" \
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --num-train-epochs "${NUM_TRAIN_EPOCHS}" \
  --learning-rate "${LEARNING_RATE}"
